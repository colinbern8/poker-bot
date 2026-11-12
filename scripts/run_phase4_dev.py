"""Stage 4C development matches: four heroes, shared deals, seat swaps (master seed 5000).

    python scripts/run_phase4_dev.py --deals 60 --workers 4           # play and report
    python scripts/run_phase4_dev.py --deals 60 --report-only         # re-analyse saved records

Every hero plays one seat-swapped match of ``--deals`` deals against each of
the 16 opponent configurations. Against opponent ``i`` all heroes use the same
match seed ``derive_seed(5000, "dev", i)``, so they receive identical cards.

Outputs:

* data/processed/phase4_dev/records_d<deals>.json.gz (git-ignored): every hand
  and hero decision;
* results/phase4/stage4c_dev_report.md and stage4c_dev_summary.json:
  bb/100, paired differences, all-in statistics, low-coverage reliance,
  latency, errors, reproducibility checks and the sample-size estimate.

Development only: these results may guide fixes before the freeze; they are
not the final evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from hybridpoker.agents.ev_agent import FROZEN_MODEL_SHA256
from hybridpoker.dataset_generation import warm_preflop_cache
from hybridpoker.hand_log import _git_commit
from hybridpoker.opponent_configs import FAMILY_ORDER, build_configs
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH, file_sha256
from hybridpoker.phase4_eval import (
    COMPARISONS,
    DEV_MASTER_SEED,
    HERO_LABELS,
    HEROES,
    Estimate,
    half_width,
    net_by_deal,
    pairs_needed,
    play_dev_match,
    to_bb100,
)
from hybridpoker.reporting import md_table

RAW_DIR = Path("data/processed/phase4_dev")
OUT = Path("results/phase4")
REPLAY_DEALS = 6
HALF_WIDTHS = (10.0, 15.0, 20.0, 30.0)


def _init_worker() -> None:
    warm_preflop_cache([500, 200])


def _dirty() -> bool | None:
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           check=True)
        return bool(r.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def play_all(deals: int, workers: int, heroes: tuple[str, ...]) -> dict:
    configs = build_configs()
    jobs = [(h, c, i, deals) for h in heroes for i, c in enumerate(configs)]
    replays = [(h, configs[i], i, REPLAY_DEALS) for h in heroes for i in (0, 10)]
    commit, dirty = _git_commit(), _dirty()
    start = time.perf_counter()
    results, replay_results = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(play_dev_match, *job): "main" for job in jobs}
        futures.update({pool.submit(play_dev_match, *job): "replay" for job in replays})
        for future in as_completed(futures):
            r = future.result()
            (results if futures[future] == "main" else replay_results).append(r)
            if futures[future] == "main":
                status = "ERROR" if r["error"] else f"{len(r['hands'])} hands"
                print(f"  {r['hero']:<13}{r['opponent']:<20}{status:<12}{r['seconds']:7.1f}s "
                      f"({len(results)}/{len(jobs)})", flush=True)
    return {"deals": deals, "master_seed": DEV_MASTER_SEED, "heroes": list(heroes),
            "opponents": [c.to_dict() for c in configs], "wall_seconds": time.perf_counter() - start,
            "workers": workers, "git_commit_at_start": commit, "working_tree_dirty_at_start": dirty,
            "model_sha256": file_sha256(PHASE4_MODEL_PATH), "python": platform.python_version(),
            "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
            "results": results, "replays": replay_results}


# --- Analysis -------------------------------------------------------------------------------------


def table(headers: list[str], rows: list[list]) -> str:
    return md_table(pd.DataFrame(rows, columns=headers))


def fmt(e: Estimate, digits: int = 1) -> str:
    return f"{e.mean:+.{digits}f} [{e.low:+.{digits}f}, {e.high:+.{digits}f}]"


def analyse(data: dict) -> tuple[dict, list[str]]:
    deals = data["deals"]
    opponents = data["opponents"]
    heroes = data["heroes"]
    by = {(r["hero"], r["opponent"]): r for r in data["results"]}
    errors = [{"hero": r["hero"], "opponent": r["opponent"], "error": r["error"]} for r in data["results"] if r["error"]]
    ok_opponents = [o for o in opponents if all(not by[(h, o["name"])]["error"] for h in heroes)]
    names = [o["name"] for o in ok_opponents]
    pair_bb = {(h, n): to_bb100(net_by_deal(by[(h, n)]["hands"], deals)) for h in heroes for n in names}
    summary: dict = {"deals": deals, "opponents": names, "errors": errors}
    lines: list[str] = []

    # 1. bb/100 per hero and opponent
    win = {(h, n): Estimate.from_units(pair_bb[(h, n)]) for h in heroes for n in names}
    groups = {"all 16": names,
              "training pool": [o["name"] for o in ok_opponents if o["pool"] == "train"],
              "held-out": [o["name"] for o in ok_opponents if o["pool"] == "heldout"],
              **{f: [o["name"] for o in ok_opponents if o["family"] == f] for f in FAMILY_ORDER}}
    summary["bb100"] = {h: {n: win[(h, n)].__dict__ for n in names} for h in heroes}
    summary["bb100_groups"] = {h: {g: Estimate.equal_weight([win[(h, n)] for n in members]).__dict__
                                   for g, members in groups.items() if members} for h in heroes}

    # 2. paired differences
    diff = {}
    for a, b in COMPARISONS:
        if a in heroes and b in heroes:
            diff[(a, b)] = {n: Estimate.from_units(pair_bb[(a, n)] - pair_bb[(b, n)]) for n in names}
    summary["paired"] = {f"{a} - {b}": {
        "per_opponent": {n: e.__dict__ for n, e in d.items()},
        "groups": {g: Estimate.equal_weight([d[n] for n in members]).__dict__ for g, members in groups.items()
                   if members}} for (a, b), d in diff.items()}

    # 3. all-in behaviour
    allin = {}
    for h in heroes:
        decisions = [d for n in names for d in by[(h, n)]["decisions"]]
        hands = [x for n in names for x in by[(h, n)]["hands"]]
        chose = [x for x in hands if x["chose_all_in"]]
        committed = [x for x in hands if x["hero_all_in"]]
        allin[h] = {
            "decisions": len(decisions), "hands": len(hands),
            "all_in_action_share": float(np.mean([d["category"] == "all_in" for d in decisions])),
            "all_in_any_share": float(np.mean([d["is_all_in"] for d in decisions])),
            "hands_chose_all_in": len(chose), "hands_hero_all_in": len(committed),
            "net_bb_chose_all_in": sum(x["hero_net"] for x in chose) / 2,
            "mean_bb_per_all_in_hand": (sum(x["hero_net"] for x in chose) / 2 / len(chose)) if chose else None,
            "bb100_from_chose_all_in": sum(x["hero_net"] for x in chose) / 2 / len(hands) * 100,
            "bb100_from_other_hands": sum(x["hero_net"] for x in hands if not x["chose_all_in"]) / 2 / len(hands) * 100,
            "all_in_share_by_street": {s: float(np.mean([d["category"] == "all_in" for d in decisions if d["street"] == s]))
                                       for s in ("preflop", "flop", "turn", "river")
                                       if any(d["street"] == s for d in decisions)},
            "by_family": {}}
        for f in FAMILY_ORDER:
            fam = [n for n in names if n in groups[f]]
            fdec = [d for n in fam for d in by[(h, n)]["decisions"]]
            fhands = [x for n in fam for x in by[(h, n)]["hands"]]
            fchose = [x for x in fhands if x["chose_all_in"]]
            allin[h]["by_family"][f] = {
                "all_in_action_share": float(np.mean([d["category"] == "all_in" for d in fdec])) if fdec else None,
                "hands_chose_all_in": len(fchose),
                "net_bb_chose_all_in": sum(x["hero_net"] for x in fchose) / 2,
                "bb100_from_chose_all_in": sum(x["hero_net"] for x in fchose) / 2 / len(fhands) * 100 if fhands else None,
                "win_rate_when_all_in": float(np.mean([x["hero_net"] > 0 for x in fchose])) if fchose else None}
    summary["all_in"] = allin

    # 4. low-coverage reliance (EV agents)
    coverage = {}
    for h in heroes:
        decisions = [d for n in names for d in by[(h, n)]["decisions"]]
        if not decisions or decisions[0]["any_low_coverage"] is None:
            continue
        bets = [d for d in decisions if d["category"] in ("raise_half_pot", "raise_pot", "all_in")]
        shoves = [d for d in decisions if d["category"] == "all_in"]
        coverage[h] = {"any_candidate_low": float(np.mean([d["any_low_coverage"] for d in decisions])),
                       "chosen_low_share_of_decisions": float(np.mean([d["chosen_low_coverage"] for d in decisions])),
                       "chosen_low_share_of_bets": float(np.mean([d["chosen_low_coverage"] for d in bets])) if bets else None,
                       "chosen_low_share_of_all_ins": float(np.mean([d["chosen_low_coverage"] for d in shoves])) if shoves else None,
                       "by_family": {f: float(np.mean([d["chosen_low_coverage"] for n in groups[f]
                                                       for d in by[(h, n)]["decisions"]])) for f in FAMILY_ORDER}}
        # net result of hands in which the hero chose a low-coverage bet
        low_hands = {(n, d["deal_index"], d["seat_swap"]) for n in names for d in by[(h, n)]["decisions"]
                     if d["chosen_low_coverage"]}
        nets = [x["hero_net"] for n in names for x in by[(h, n)]["hands"]
                if (n, x["deal_index"], x["seat_swap"]) in low_hands]
        coverage[h]["hands_with_low_coverage_choice"] = len(nets)
        coverage[h]["net_bb_in_those_hands"] = sum(nets) / 2
    summary["coverage"] = coverage

    # 5. latency
    latency = {}
    for h in heroes:
        decisions = [d for n in names for d in by[(h, n)]["decisions"]]
        lat = np.array([d["latency"] for d in decisions]) * 1000
        latency[h] = {"median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
                      "max_ms": float(lat.max()), "decisions": len(lat)}
        if decisions[0]["equity_seconds"] is not None:
            for part in ("equity", "model", "range"):
                values = np.array([d[f"{part}_seconds"] for d in decisions]) * 1000
                latency[h][f"{part}_median_ms"] = float(np.median(values))
                latency[h][f"{part}_p95_ms"] = float(np.percentile(values, 95))
        seconds = sum(by[(h, n)]["seconds"] for n in names)
        latency[h]["seconds_per_deal_pair"] = seconds / (deals * len(names))
    summary["latency"] = latency

    # 6. reproducibility
    replay_ok, replay_rows = True, []
    for r in data["replays"]:
        main = by[(r["hero"], r["opponent"])]
        n = min(r["deals"], deals)  # compare the common prefix of deals

        def hands_of(x):
            return [(h["deal_index"], h["seat_swap"], h["hero_net"], h["cards_hash"]) for h in x["hands"]
                    if h["deal_index"] < n]

        def actions_of(x):
            return [(d["deal_index"], d["seat_swap"], d["category"]) for d in x["decisions"] if d["deal_index"] < n]

        same = not r["error"] and hands_of(main) == hands_of(r) and actions_of(main) == actions_of(r)
        replay_ok &= same
        replay_rows.append([HERO_LABELS[r["hero"]], r["opponent"], n, "yes" if same else "NO"])
    shared = all(len({tuple(x["cards_hash"] for x in by[(h, n)]["hands"]) for h in heroes}) == 1 for n in names)
    summary["reproducibility"] = {"replays_identical": replay_ok, "deals_shared_across_heroes": shared}

    # 7. sample size
    sample = {}
    for (a, b), d in diff.items():
        sds = [d[n].sd for n in names]
        sample[f"{a} - {b}"] = {"per_opponent_sd": dict(zip(names, sds)),
                                "pairs_for_half_width": {str(w): pairs_needed(sds, w) for w in HALF_WIDTHS},
                                "half_width_at": {str(p): half_width(sds, p) for p in (100, 200, 300, 400, 600)}}
    summary["sample_size"] = sample

    # --- Markdown ---
    ev_heroes = [h for h in heroes if h != "probability"]
    lines += [
        "# Stage 4C development evaluation (master seed 5000)",
        "",
        "Generated by `scripts/run_phase4_dev.py`. **Development results only**: master seed 5000, used to find "
        "problems and size the final evaluation. They are not the final evaluation (master seed 4000, after the "
        "freeze).",
        "",
        f"- Deals per match: {deals} (each played twice with seats swapped: {2 * deals} hands per hero and opponent).",
        f"- Opponents: {len(names)} of {len(opponents)} configurations (12 training-pool, 4 held-out).",
        f"- Heroes: {', '.join(HERO_LABELS[h] for h in heroes)}. All heroes face identical deals "
        f"(`derive_seed(5000, \"dev\", i)` per opponent); each starts with no memory.",
        f"- Code commit at start: `{data['git_commit_at_start']}` (uncommitted changes: "
        f"{data['working_tree_dirty_at_start']}). Model SHA-256: `{data['model_sha256']}` "
        f"({'matches' if data['model_sha256'] == FROZEN_MODEL_SHA256 else 'DOES NOT MATCH'} the frozen hash).",
        f"- Runtime: {data['wall_seconds'] / 60:.1f} min wall time, {data['workers']} worker processes, "
        f"Python {data['python']}.",
        "- Uncertainty: 95% normal intervals with the deal pair as the unit. Group rows weight every opponent "
        "equally.",
        "",
        "## 1. Win rate (bb/100)",
        "",
    ]
    rows = [[o["name"], o["family"], o["pool"], *[fmt(win[(h, o['name'])]) for h in heroes]] for o in ok_opponents]
    lines += [table(["opponent", "style", "pool", *[HERO_LABELS[h] for h in heroes]], rows), ""]
    rows = [[g, *[fmt(Estimate(**summary["bb100_groups"][h][g])) for h in heroes]] for g in groups if groups[g]]
    lines += ["Equal-weight averages:", "", table(["group", *[HERO_LABELS[h] for h in heroes]], rows), ""]

    lines += ["## 2. Paired differences (bb/100, same deals)", ""]
    rows = []
    for (a, b), d in diff.items():
        s = summary["paired"][f"{a} - {b}"]["groups"]
        rows.append([f"{HERO_LABELS[a]} − {HERO_LABELS[b]}", *[fmt(Estimate(**s[g])) for g in groups if groups[g]]])
    lines += [table(["comparison", *[g for g in groups if groups[g]]], rows), ""]
    for a, b in (("ev_ml", "ev_situation"), ("ev_ml", "ev_global"), ("ev_ml", "probability")):
        if (a, b) in diff:
            lines += [f"Per opponent, {HERO_LABELS[a]} − {HERO_LABELS[b]}:", "",
                      table(["opponent", "style", "pool", "difference", "SD per pair"],
                               [[o["name"], o["family"], o["pool"], fmt(diff[(a, b)][o["name"]]),
                                 f"{diff[(a, b)][o['name']].sd:.0f}"] for o in ok_opponents]), ""]

    lines += ["## 3. All-in decisions", "",
              "\"All-in action\" is the hero choosing the all_in category; \"any all-in\" also counts calls that put "
              "the hero all-in. Profit of all-in hands is the hero's net result in every hand in which it chose "
              "all_in (the result of the whole hand, not only of the shove).", ""]
    rows = [[HERO_LABELS[h], f"{a['all_in_action_share']:.1%}", f"{a['all_in_any_share']:.1%}",
             f"{a['hands_chose_all_in']} / {a['hands']}", f"{a['net_bb_chose_all_in']:+.0f}",
             f"{a['mean_bb_per_all_in_hand']:+.1f}" if a["mean_bb_per_all_in_hand"] is not None else "–",
             f"{a['bb100_from_chose_all_in']:+.1f}", f"{a['bb100_from_other_hands']:+.1f}"]
            for h, a in allin.items()]
    lines += [table(["hero", "all-in action (of decisions)", "any all-in", "hands with a shove",
                        "net bb in those hands", "bb per shove hand", "bb/100 from shove hands",
                        "bb/100 from other hands"], rows), ""]
    rows = [[HERO_LABELS[h], *[f"{allin[h]['all_in_share_by_street'].get(s, 0):.1%}"
                               for s in ("preflop", "flop", "turn", "river")]] for h in heroes]
    lines += ["All-in action share by street:", "", table(["hero", "preflop", "flop", "turn", "river"], rows), ""]
    rows = []
    for h in heroes:
        for f in FAMILY_ORDER:
            x = allin[h]["by_family"][f]
            rows.append([HERO_LABELS[h], f, f"{x['all_in_action_share']:.1%}", x["hands_chose_all_in"],
                         f"{x['net_bb_chose_all_in']:+.0f}",
                         f"{x['bb100_from_chose_all_in']:+.1f}",
                         f"{x['win_rate_when_all_in']:.0%}" if x["win_rate_when_all_in"] is not None else "–"])
    lines += ["By opponent style:", "", table(["hero", "style", "all-in action", "shove hands", "net bb",
                                                  "bb/100 from shove hands", "shove hands won"], rows), ""]

    lines += ["## 4. Reliance on poorly covered hypothetical states", "",
              "A candidate bet is *low-coverage* when the opponent state it creates has fewer than 30 training rows "
              "(street | facing all-in | raises; `state_coverage.json`). The flag is diagnostic only.", ""]
    rows = [[HERO_LABELS[h], f"{c['any_candidate_low']:.1%}", f"{c['chosen_low_share_of_decisions']:.1%}",
             f"{c['chosen_low_share_of_bets']:.1%}" if c["chosen_low_share_of_bets"] is not None else "–",
             f"{c['chosen_low_share_of_all_ins']:.1%}" if c["chosen_low_share_of_all_ins"] is not None else "–",
             c["hands_with_low_coverage_choice"], f"{c['net_bb_in_those_hands']:+.0f}"] for h, c in coverage.items()]
    lines += [table(["hero", "decisions with a low-coverage candidate", "chosen action low-coverage",
                        "of its bets", "of its all-ins", "hands", "net bb in those hands"], rows), ""]
    rows = [[HERO_LABELS[h], *[f"{c['by_family'][f]:.1%}" for f in FAMILY_ORDER]] for h, c in coverage.items()]
    lines += ["Chosen action low-coverage, by opponent style:", "", table(["hero", *FAMILY_ORDER], rows), ""]

    lines += ["## 5. Decision latency (warm caches)", ""]
    rows = []
    for h in heroes:
        x = latency[h]
        parts = (f"{x['equity_median_ms']:.0f} / {x['model_median_ms']:.0f} / {x['range_median_ms']:.0f}"
                 if "equity_median_ms" in x else "–")
        rows.append([HERO_LABELS[h], x["decisions"], f"{x['median_ms']:.0f}", f"{x['p95_ms']:.0f}",
                     f"{x['max_ms']:.0f}", parts, f"{x['seconds_per_deal_pair']:.2f}"])
    lines += [table(["hero", "decisions", "median ms", "95th pct ms", "max ms",
                        "median equity / model / range ms", "s per deal pair (incl. opponent)"], rows), ""]

    lines += ["## 6. Errors, illegal actions and reproducibility", "",
              f"- Matches with errors (including illegal actions): {len(errors)} of {len(data['results'])}.",
              f"- Every hero received identical cards against each opponent: "
              f"{'yes' if shared else 'NO'}.",
              f"- Replays of the first {REPLAY_DEALS} deals (fresh processes) reproduce the main matches exactly "
              f"(net chips, cards and hero actions): {'yes' if replay_ok else 'NO'}.", "",
              table(["hero", "opponent", "deals replayed", "identical"], replay_rows), ""]
    for e in errors:
        lines += [f"Error, {e['hero']} vs {e['opponent']}:", "", "```", e["error"].strip()[-1500:], "```", ""]

    lines += ["## 7. Sample size for the final evaluation", "",
              "Deal pairs per opponent needed so that the equal-weight average over 16 opponents has the given 95% "
              "interval half-width (from the per-opponent standard deviation of paired differences measured here):",
              ""]
    rows = []
    for key, s in sample.items():
        a, b = key.split(" - ")
        rows.append([f"{HERO_LABELS[a]} − {HERO_LABELS[b]}",
                     f"{np.sqrt(np.mean(np.square(list(s['per_opponent_sd'].values())))):.0f}",
                     *[s["pairs_for_half_width"][str(w)] for w in HALF_WIDTHS],
                     *[f"±{s['half_width_at'][str(p)]:.1f}" for p in (200, 400, 600)]])
    lines += [table(["comparison", "RMS SD per pair", *[f"pairs for ±{w:.0f}" for w in HALF_WIDTHS],
                        "half-width at 200", "at 400", "at 600"], rows), ""]
    per_pair = sum(latency[h]["seconds_per_deal_pair"] for h in heroes)
    lines += [f"Measured compute: {per_pair:.1f} CPU-seconds per deal pair for all {len(heroes)} heroes together "
              f"(per opponent). For 16 opponents on 4 processes: about {per_pair * 16 / 4 / 3600 * 100:.2f} h "
              "per 100 deal pairs.", ""]
    return summary, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deals", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--heroes", nargs="*", default=list(HEROES))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    raw = RAW_DIR / f"records_d{args.deals}.json.gz"
    if args.report_only:
        data = json.loads(gzip.decompress(raw.read_bytes()))
    else:
        if file_sha256(PHASE4_MODEL_PATH) != FROZEN_MODEL_SHA256:
            print("The model file does not match the frozen SHA-256.", file=sys.stderr)
            return 1
        data = play_all(args.deals, args.workers, tuple(args.heroes))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(gzip.compress(json.dumps(data).encode()))
        print(f"wrote {raw}")
    summary, lines = analyse(data)
    summary.update({k: data[k] for k in ("master_seed", "git_commit_at_start", "working_tree_dirty_at_start",
                                         "model_sha256", "wall_seconds", "workers", "python", "machine")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage4c_dev_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (OUT / "stage4c_dev_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'stage4c_dev_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
