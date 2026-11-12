"""Leakage and reproducibility audits L1-L11 (``docs/phase3_plan.md`` section 10).

Each audit returns an :class:`AuditResult`. The unit tests run them on
small synthetic matches, and ``scripts/pilot_report.py`` runs them on the
pilot dataset.

Labels (``label_``) are read here only to *verify* that nothing leaks,
never to build features.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pokerkit import Card, Deck

from hybridpoker.actions import ActionCategory
from hybridpoker.dataset_manifest import (
    HELDOUT_SPLIT,
    PILOT_HELDOUT_SPLIT,
    build_manifest,
    build_pilot_manifest,
    configs_by_name,
    manifest_text,
)
from hybridpoker.dataset_loader import prior_rows
from hybridpoker.features import CATEGORIES, RAISE_VALUES, feature_set_a, parse_history
from hybridpoker.game import create_heads_up_state
from hybridpoker.observation import observe
from hybridpoker.opponent_configs import PARAMETER_NAMES, STYLE_PRESETS
from hybridpoker.opponent_history import HISTORY_KEYS, HistoryPriors, fit_priors, history_features, player_role
from hybridpoker.pot_odds import pot_odds

FORBIDDEN_PREFIXES = ("id_", "hidden_", "label_", "out_", "act_", "obs_")
ALLOWED_FEATURE_PREFIXES = ("a_", "b_")
FORBIDDEN_TERMS = (
    *STYLE_PRESETS, *PARAMETER_NAMES, "hole", "hidden", "label", "agent", "role", "seed",
    "match", "split", "equity", "reason", "net", "showdown", "final", "config",
)
PRIVATE_PREFIXES = ("hidden_", "label_", "out_")
HELDOUT_SPLITS = (HELDOUT_SPLIT, PILOT_HELDOUT_SPLIT)
ALTERED_OBS = ("obs_to_call", "obs_street")

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class AuditResult:
    audit_id: str
    name: str
    passed: bool
    detail: str


def _result(audit_id: str, name: str, problems: list[str], checked: str) -> AuditResult:
    if problems:
        shown = "; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
        return AuditResult(audit_id, name, False, shown)
    return AuditResult(audit_id, name, True, checked)


def _same(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.shape == right.shape and np.array_equal(left.to_numpy(), right.to_numpy())


def time_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted by match, then the order decisions happened."""
    return frame.sort_values(["id_match", "id_hand_index", "id_decision_index"])


# --- L1: column allowlist --------------------------------------------------------------


def audit_feature_names(columns: list[str]) -> AuditResult:
    problems = []
    for column in columns:
        if not column.startswith(ALLOWED_FEATURE_PREFIXES):
            problems.append(f"{column}: not an a_/b_ feature")
        if column.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"{column}: forbidden prefix")
        hits = [term for term in FORBIDDEN_TERMS if term in column]
        if hits:
            problems.append(f"{column}: contains {hits}")
    if len(set(columns)) != len(columns):
        problems.append("duplicate feature names")
    return _result("L1", "Feature-name allowlist", problems, f"{len(columns)} feature names checked")


# --- L2: scrambling private columns ----------------------------------------------------


def scramble_private_columns(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle every hidden/label/outcome column and every non-key identifier."""
    rng = np.random.default_rng(seed)
    scrambled = frame.copy()
    for column in frame.columns:
        private = column.startswith(PRIVATE_PREFIXES)
        non_key_id = column.startswith("id_") and column not in HISTORY_KEYS
        if private or non_key_id:
            scrambled[column] = frame[column].to_numpy()[rng.permutation(len(frame))]
    return scrambled


def audit_scrambling(frame: pd.DataFrame, builders: dict[str, FeatureBuilder], seed: int = 0) -> AuditResult:
    scrambled = scramble_private_columns(frame, seed)
    problems = [name for name, build in builders.items() if not _same(build(frame), build(scrambled))]
    return _result("L2", "Private columns scrambled: features unchanged",
                   [f"{p} features changed" for p in problems], f"{len(frame)} rows, sets {sorted(builders)}")


# --- L3: future truncation -------------------------------------------------------------


def audit_future_truncation(
    frame: pd.DataFrame, builders: dict[str, FeatureBuilder], sample: int | None = None, seed: int = 0
) -> AuditResult:
    """A row's features are unchanged when every later decision of its match is deleted."""
    ordered = time_order(frame)
    full = {name: build(ordered) for name, build in builders.items()}
    positions = list(range(len(ordered)))
    if sample is not None and sample < len(positions):
        positions = sorted(random.Random(seed).sample(positions, sample))
    problems = []
    match_ids = ordered["id_match"].to_numpy()
    for position in positions:
        row_label = ordered.index[position]
        same_match = match_ids == match_ids[position]
        later = same_match & (np.arange(len(ordered)) > position)
        truncated = ordered[~later]
        for name, build in builders.items():
            if not np.array_equal(build(truncated).loc[row_label].to_numpy(), full[name].loc[row_label].to_numpy()):
                problems.append(f"{name} row {row_label}")
    return _result("L3", "Future truncation: features unchanged", problems,
                   f"{len(positions)} rows truncated, sets {sorted(builders)}")


# --- L4 / L5: history uses only earlier deals ------------------------------------------


def _alter_actions(frame: pd.DataFrame, mask: pd.Series, seed: int) -> pd.DataFrame:
    """Replace the actions (and the public context of those actions) on masked rows."""
    rng = random.Random(seed)
    altered = frame.copy()
    count = int(mask.sum())
    altered.loc[mask, "act_category"] = [rng.choice(CATEGORIES) for _ in range(count)]
    altered.loc[mask, "obs_to_call"] = [rng.choice((0, 2, 7)) for _ in range(count)]
    altered.loc[mask, "obs_street"] = [rng.choice(("preflop", "flop", "river")) for _ in range(count)]
    return altered


def audit_history_uses_earlier_deals(frame: pd.DataFrame, priors: HistoryPriors, seed: int = 0) -> AuditResult:
    """L4: B features of deal d ignore deal d and every later deal (altered or deleted)."""
    base = history_features(frame, priors)
    b_cols = list(base.columns)
    rng = random.Random(seed)
    problems = []
    checks = 0
    for match_id, rows in frame.groupby("id_match"):
        deals = sorted(rows["id_deal_index"].unique())
        for d in sorted(set(rng.sample(deals, min(4, len(deals)))) | {deals[0], deals[-1]}):
            target = (frame["id_match"] == match_id) & (frame["id_deal_index"] == d)
            later_or_same = (frame["id_match"] == match_id) & (frame["id_deal_index"] >= d)
            altered = history_features(_alter_actions(frame, later_or_same, rng.random()), priors)
            deleted_frame = frame[~((frame["id_match"] == match_id) & (frame["id_deal_index"] > d))]
            deleted = history_features(deleted_frame, priors)
            if not _same(altered.loc[target, b_cols], base.loc[target, b_cols]):
                problems.append(f"{match_id} deal {d}: changed when deals >= d altered")
            if not _same(deleted.loc[target[target].index, b_cols], base.loc[target, b_cols]):
                problems.append(f"{match_id} deal {d}: changed when later deals deleted")
            checks += 1
    return _result("L4", "History uses only earlier deals", problems, f"{checks} (match, deal) checks")


def audit_mirror_hand_excluded(frame: pd.DataFrame, priors: HistoryPriors, seed: int = 0) -> AuditResult:
    """L5: the first seating of deal d cannot change B features in its second seating."""
    base = history_features(frame, priors)
    b_cols = list(base.columns)
    rng = random.Random(seed)
    problems = []
    checks = 0
    for match_id, rows in frame.groupby("id_match"):
        deals = sorted(rows["id_deal_index"].unique())
        for d in rng.sample(deals, min(5, len(deals))):
            in_deal = (frame["id_match"] == match_id) & (frame["id_deal_index"] == d)
            first = in_deal & (frame["id_seat_swap"] == 0)
            second = in_deal & (frame["id_seat_swap"] == 1)
            if not second.any():
                continue
            altered = history_features(_alter_actions(frame, first, rng.random()), priors)
            if not _same(altered.loc[second, b_cols], base.loc[second, b_cols]):
                problems.append(f"{match_id} deal {d}")
            checks += 1
    return _result("L5", "Mirror hand of the current deal excluded", problems, f"{checks} deals checked")


# --- L6: public replay with fake hole cards ---------------------------------------------


def replay_public_row(row: pd.Series) -> dict[str, Any]:
    """Rebuild a row's ``obs_`` values from its public board and history,
    replayed in PokerKit with FAKE hole cards."""
    board = list(Card.parse(row["obs_board"])) if row["obs_board"] else []
    fake_cards = iter([card for card in Deck.STANDARD if card not in set(board)])
    board_cards = iter(board)
    state = create_heads_up_state()

    def deal_pending_cards() -> None:
        while True:
            if state.can_deal_hole():
                state.deal_hole(next(fake_cards))
            elif state.can_burn_card():
                state.burn_card(next(fake_cards))
            elif state.can_deal_board():
                state.deal_board(next(board_cards))
            else:
                return

    history = parse_history(row["obs_history"])
    for _street, _position, category, bet_to in history:
        deal_pending_cards()
        if category == "fold":
            state.fold()
        elif category == "check_call":
            state.check_or_call()
        else:
            state.complete_bet_or_raise_to(bet_to)
    deal_pending_cards()
    obs = observe(state, history=())
    this_street = [e for e in history if e[0] == obs.street]
    replayed = {
        "obs_street": obs.street,
        "obs_position": obs.position,
        "obs_board": row["obs_board"],
        "obs_history": row["obs_history"],
        "obs_pot_size": obs.pot_size,
        "obs_to_call": obs.to_call,
        "obs_pot_odds": pot_odds(obs.pot_size, obs.to_call),
        "obs_own_stack": obs.own_stack,
        "obs_opponent_stack": obs.opponent_stack,
        "obs_effective_stack": obs.effective_stack,
        "obs_own_bet": obs.own_bet,
        "obs_opponent_bet": obs.opponent_bet,
        "obs_actions_this_street": len(this_street),
        "obs_raises_this_street": sum(e[2] in RAISE_VALUES for e in this_street),
        "obs_raises_this_hand": sum(e[2] in RAISE_VALUES for e in history),
    }
    for category in ActionCategory:
        legal = obs.legal(category)
        replayed[f"obs_legal_{category.value}"] = legal is not None
        replayed[f"obs_bet_to_{category.value}"] = legal.bet_to if legal else pd.NA
    return replayed


def audit_public_replay(frame: pd.DataFrame) -> AuditResult:
    """L6: Feature Set A rebuilt from a fake-hole-card replay equals the logged one."""
    replayed = pd.DataFrame([replay_public_row(row) for _, row in frame.iterrows()], index=frame.index)
    logged = feature_set_a(frame)
    rebuilt = feature_set_a(replayed)
    mismatched = (~np.isclose(logged.to_numpy(), rebuilt.to_numpy(), rtol=0, atol=1e-12)).any(axis=1)
    problems = [f"row {i}" for i in frame.index[mismatched]]
    return _result("L6", "Public replay with fake hole cards", problems, f"{len(frame)} rows replayed, 0 mismatches")


# --- L7: player role --------------------------------------------------------------------


def audit_player_role(frame: pd.DataFrame) -> AuditResult:
    derived = player_role(frame)
    wrong = derived != frame["label_actor_role"]
    return _result("L7", "Player role from public seating equals true role",
                   [f"{int(wrong.sum())} rows differ"] if wrong.any() else [], f"{len(frame)} rows, 100% match")


# --- L8: split integrity ----------------------------------------------------------------


def audit_split_integrity(frame: pd.DataFrame, manifests: list[dict[str, Any]]) -> AuditResult:
    problems = []
    for manifest in manifests:
        configs = configs_by_name(manifest)
        for spec in manifest["matches"]:
            pools = {configs[spec["hero"]].pool, configs[spec["villain"]].pool}
            if spec["split"] in HELDOUT_SPLITS:
                scored_name = spec[spec["scored_role"]]
                if configs[scored_name].pool != "heldout":
                    problems.append(f"{manifest['name']} match {spec['number']}: scored player not held out")
            elif "heldout" in pools:
                problems.append(f"{manifest['name']} match {spec['number']}: held-out config in {spec['split']}")
    per_match = frame.groupby("id_match")[["id_split", "id_match_number"]].nunique()
    if (per_match > 1).any().any():
        problems.append("a match id has more than one split or number")
    if frame.groupby("id_match_number")["id_match"].nunique().max() > 1:
        problems.append("a match number has more than one match id")
    in_training_splits = ~frame["id_split"].isin(HELDOUT_SPLITS)
    if (frame.loc[in_training_splits, "label_actor_pool"] != "train").any():
        problems.append("held-out configuration rows in a training-pool split")
    heldout_rows = frame["id_split"].isin(HELDOUT_SPLITS)
    scored = heldout_rows & frame["id_scored"]
    if (frame.loc[scored, "label_actor_pool"] != "heldout").any():
        problems.append("scored held-out rows from a training-pool configuration")
    if (frame.loc[heldout_rows & ~frame["id_scored"], "label_actor_pool"] != "train").any():
        problems.append("unscored held-out rows from a held-out configuration")
    return _result("L8", "Split integrity", problems,
                   f"{len(manifests)} manifests, {frame['id_match'].nunique()} loaded matches")


# --- L9: seeds --------------------------------------------------------------------------


def audit_seed_uniqueness(manifests: list[dict[str, Any]], frame: pd.DataFrame | None = None) -> AuditResult:
    seeds = [m["seed"] for manifest in manifests for m in manifest["matches"]]
    problems = [] if len(seeds) == len(set(seeds)) else [f"{len(seeds) - len(set(seeds))} repeated match seeds"]
    if frame is not None:
        shared = frame.groupby("id_deal_seed")["id_match"].nunique()
        if (shared > 1).any():
            problems.append(f"{int((shared > 1).sum())} deal seeds shared between matches")
    return _result("L9", "Seed uniqueness", problems, f"{len(seeds)} match seeds distinct")


# --- L10: train-only priors -------------------------------------------------------------


def audit_train_only_priors(frame: pd.DataFrame, manifest: dict[str, Any], seed: int = 0) -> AuditResult:
    """Altering every row outside the prior-fitting splits leaves the priors unchanged."""
    priors = fit_priors(prior_rows(frame, manifest))
    outside = ~frame["id_split"].isin(manifest["prior_splits"])
    problems = []
    if not outside.any():
        problems.append("no rows outside the prior splits to perturb")
    altered = _alter_actions(frame, outside, seed)
    if fit_priors(prior_rows(altered, manifest)) != priors:
        problems.append("priors changed when non-training rows changed")
    if priors.fitted_rows != int((~outside).sum()):
        problems.append("priors fitted on the wrong number of rows")
    return _result("L10", "Priors fitted on training rows only", problems,
                   f"fitted on {priors.fitted_rows} rows; {int(outside.sum())} other rows perturbed")


# --- L11: reproducibility ---------------------------------------------------------------


def audit_manifest_reproducible(committed: dict[str, str]) -> AuditResult:
    """Rebuilding each manifest gives byte-identical JSON to the committed file text."""
    builders = {"phase3": build_manifest, "phase3_pilot": build_pilot_manifest}
    problems = []
    for name, text in committed.items():
        first, second = manifest_text(builders[name]()), manifest_text(builders[name]())
        if first != second:
            problems.append(f"{name}: two builds differ")
        if first != text:
            problems.append(f"{name}: differs from the committed file")
    return _result("L11a", "Manifests rebuild byte-identically", problems, f"{sorted(committed)} identical")


def audit_regenerated_bytes(original: bytes, regenerated: bytes, label: str) -> AuditResult:
    problems = [] if original == regenerated else [f"{label}: decisions.csv differs"]
    return _result("L11b", "Regenerated match is byte-identical", problems, f"{label}: {len(original)} bytes identical")
