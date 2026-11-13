"""Stage 4B audit: does the continuing-range fix remove pathological all-ins, and what does it cost?

    python scripts/audit_ev_decisions.py --spots 300

Not a match or tournament. A fixed sample of decision situations from the
Phase 3 **validation** split is rebuilt (public history and board, plus the
deciding player's own hole cards, which a real player knows) and each one is
decided by four configurations with the same seed:

    {Frequency EV, ML EV} x {old rule: random-hand equity when called,
                             new rule: equity against the continuing range}

The ML agent has no opponent history here (history = training priors).
Caches are warmed before timing. Writes results/phase4/stage4b_rule_audit.md
and .json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from pokerkit import Card, Deck

from hybridpoker.agents.ev_agent import EVAgent, FrequencyResponseModel, MLResponseModel
from hybridpoker.agents.strength import preflop_equity
from hybridpoker.dataset_generation import preflop_classes
from hybridpoker.dataset_loader import load_dataset
from hybridpoker.dataset_manifest import DEFAULT_MANIFEST_DIR, MAIN_MANIFEST_NAME, load_manifest
from hybridpoker.features import parse_history
from hybridpoker.observation import PublicAction, observe
from hybridpoker.actions import ActionCategory, LegalAction, apply_action
from hybridpoker.game import create_heads_up_state
from hybridpoker.reporting import md_table
from hybridpoker.seeding import derive_seed
from hybridpoker.state_coverage import load_coverage

RAW = Path("data/raw/simulations/phase3_s3000")
OUT = Path("results/phase4")
ACTIONS = ("fold", "check_call", "raise_half_pot", "raise_pot", "all_in")


def rebuild(row):
    """The deciding player's observation: public history and board, plus its own hole cards.

    The public history is replayed in PokerKit with fake hole cards (as in
    hybridpoker.counterfactual); only then are the player's own cards added.
    """
    board = list(Card.parse(row["obs_board"])) if row["obs_board"] else []
    history = tuple(PublicAction(street, position, ActionCategory(category), bet_to, 0, False)
                    for street, position, category, bet_to in parse_history(row["obs_history"]))
    fake = iter([c for c in Deck.STANDARD if c not in set(board)])
    board_iter = iter(board)
    state = create_heads_up_state()

    def deal() -> None:
        while True:
            if state.can_deal_hole():
                state.deal_hole(next(fake))
            elif state.can_burn_card():
                state.burn_card(next(fake))
            elif state.can_deal_board():
                state.deal_board(next(board_iter))
            else:
                return

    for action in history:
        deal()
        apply_action(state, LegalAction(action.category, action.bet_to, 0, False))
    deal()
    return dataclasses.replace(observe(state, history), hole_cards=tuple(Card.parse(row["hidden_actor_hole_cards"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spots", type=int, default=300)
    args = parser.parse_args()

    manifest = load_manifest(DEFAULT_MANIFEST_DIR / MAIN_MANIFEST_NAME)
    frame = load_dataset(manifest, RAW)
    validation = frame[frame["id_split"] == "validation"]
    sample = validation.sample(n=args.spots, random_state=derive_seed(3000, "stage4b_audit") % 2**32)

    # Warm caches so timings are steady-state (500: hero equity; 200: preflop range ranking).
    for samples in (200, 500):
        for c in preflop_classes():
            preflop_equity(c, samples)

    coverage = load_coverage()
    frequency = FrequencyResponseModel.load()
    ml = MLResponseModel.load()
    configs = {}
    for model_name, model in (("frequency", frequency), ("ml", ml)):
        configs[f"{model_name}_old"] = EVAgent(f"{model_name}_old", model, coverage=coverage,
                                               called_equity_fn=lambda o, e, fs, n, r: [e for _ in fs])
        configs[f"{model_name}_new"] = EVAgent(f"{model_name}_new", model, coverage=coverage)

    rows = []
    for index, (_, row) in enumerate(sample.iterrows()):
        obs = rebuild(row)
        assert obs.pot_size == row["obs_pot_size"] and obs.to_call == row["obs_to_call"], "rebuild mismatch"
        seed = derive_seed(3000, "stage4b_audit_spot", index)
        for name, agent in configs.items():
            agent.act(obs, random.Random(seed))
            d = agent.diagnostics[-1]
            rows.append({"spot": index, "config": name, "street": d.street, "equity": d.equity, "chosen": d.chosen,
                         "facing_bet": obs.to_call > 0,
                         "all_in_ev": d.action_evs.get("all_in", np.nan), "check_call_ev": d.action_evs.get("check_call", np.nan),
                         "equity_ms": 1000 * d.equity_seconds, "model_ms": 1000 * d.model_seconds,
                         "range_ms": 1000 * d.range_seconds, "total_ms": 1000 * d.total_seconds,
                         "all_in_called_equity": d.called_equity.get("all_in", np.nan),
                         "all_in_continue": d.continue_fraction.get("all_in", np.nan),
                         "all_in_low_coverage": d.coverage.get("all_in", {}).get("low", False)})
    df = pd.DataFrame(rows)

    summary = []
    for name in configs:
        part = df[df["config"] == name]
        entry = {"configuration": name, "spots": len(part)}
        for a in ACTIONS:
            entry[a] = f"{(part['chosen'] == a).mean():.1%}"
        entry["all-in with equity < 0.70"] = f"{((part['chosen'] == 'all_in') & (part['equity'] < 0.70)).mean():.1%}"
        summary.append(entry)
    timing = []
    for name in ("frequency_new", "ml_new", "frequency_old", "ml_old"):
        part = df[df["config"] == name]
        for column in ("equity_ms", "model_ms", "range_ms", "total_ms"):
            timing.append({"configuration": name, "component": column.replace("_ms", ""),
                           "median ms": float(part[column].median()), "95th pct ms": float(part[column].quantile(0.95))})
    by_street = df[df["config"].str.endswith("_new")].groupby(["config", "street"])["total_ms"].agg(
        ["median", lambda s: s.quantile(0.95)]).reset_index()
    by_street.columns = ["configuration", "street", "median ms", "95th pct ms"]
    changed = {}
    for model_name in ("frequency", "ml"):
        old = df[df["config"] == f"{model_name}_old"].set_index("spot")["chosen"]
        new = df[df["config"] == f"{model_name}_new"].set_index("spot")["chosen"]
        changed[model_name] = float((old != new).mean())
    agree = float((df[df["config"] == "frequency_new"].set_index("spot")["chosen"]
                   == df[df["config"] == "ml_new"].set_index("spot")["chosen"]).mean())

    md = ["# Stage 4B: rule audit on fixed validation-split spots", "",
          f"{args.spots} decision situations sampled (seeded) from the Phase 3 validation split, rebuilt from public "
          "history and board plus the deciding player's own cards. Each decided by four configurations with the "
          "same seed. Not a match; no opponent history for the ML agent (training priors). Caches warmed.", "",
          "## Chosen actions", "", md_table(pd.DataFrame(summary)), "",
          f"Decisions changed by the fix: Frequency {changed['frequency']:.1%}, ML {changed['ml']:.1%}. "
          f"Frequency and ML (new rule) choose the same action in {agree:.1%} of spots.", "",
          "## Decision latency (warm caches)", "", md_table(pd.DataFrame(timing), "{:.1f}"), "",
          "By street (new rule, total):", "", md_table(by_street, "{:.1f}"), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage4b_rule_audit.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "stage4b_rule_audit.json").write_text(json.dumps(
        {"summary": summary, "timing": timing, "changed": changed, "agreement_new": agree,
         "spots": args.spots}, indent=2) + "\n", encoding="utf-8")
    df.to_csv(OUT / "stage4b_rule_audit_rows.csv", index=False, lineterminator="\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
