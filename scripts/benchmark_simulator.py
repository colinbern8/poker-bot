"""Measure simulator speed for four matchups (scripted-only and probability).

Usage:
    python scripts/benchmark_simulator.py --deals 100
    python scripts/benchmark_simulator.py --deals 30 --profile

Matchups (default equity samples: probability 500, style bots 200):
    random vs random            no equity calculations at all
    solid vs solid, maniac vs nit   scripted StyleAgent vs StyleAgent
    probability vs solid        ProbabilityAgent vs StyleAgent

The preflop equity cache is filled first and timed separately, so the
match timings show steady-state speed. With --profile, the share of time
spent in Monte Carlo equity is reported for each matchup. Timing depends on
the machine: run it on an otherwise idle computer.
"""

from __future__ import annotations

import argparse
import cProfile
import platform
import pstats
import time

from hybridpoker.agents import make_agent
from hybridpoker.agents.simple import STYLE_PRESETS
from hybridpoker.agents.probability import ProbabilityParameters
from hybridpoker.agents.strength import RANK_ORDER, preflop_equity
from hybridpoker.match import run_match

MATCHUPS = [
    ("random", "random"),  # no equity calculations
    ("solid", "solid"),  # scripted StyleAgent vs StyleAgent
    ("maniac", "nit"),  # scripted StyleAgent vs StyleAgent
    ("probability", "solid"),  # ProbabilityAgent vs StyleAgent
]


def all_hand_classes() -> list[str]:
    classes = []
    for i, high in enumerate(reversed(RANK_ORDER)):
        for low in reversed(RANK_ORDER[: len(RANK_ORDER) - i]):
            if high == low:
                classes.append(high * 2)
            else:
                classes += [f"{high}{low}s", f"{high}{low}o"]
    return classes


def warm_preflop_cache() -> float:
    start = time.perf_counter()
    sample_counts = {ProbabilityParameters().equity_samples, STYLE_PRESETS["solid"].equity_samples}
    for samples in sample_counts:
        for hand_class in all_hand_classes():
            preflop_equity(hand_class, samples)
    return time.perf_counter() - start


def equity_share(profiler: cProfile.Profile) -> float:
    stats = pstats.Stats(profiler)
    equity_time = sum(
        v[3] for k, v in stats.stats.items() if k[0].endswith("equity.py") and k[2] == "estimate_equity"
    )
    return equity_time / stats.total_tt


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simulator speed.")
    parser.add_argument("--deals", type=int, default=100, help="seat-swapped deals per matchup")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--profile", action="store_true", help="also report Monte Carlo time share")
    args = parser.parse_args()

    assert len(all_hand_classes()) == 169
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"Preflop cache warm-up (169 classes x 2 sample counts): {warm_preflop_cache():.1f} s\n")

    header = f"{'matchup':<24}{'hands':>7}{'hands/s':>10}{'decisions/s':>13}{'ms/decision':>13}{'dec/hand':>10}"
    if args.profile:
        header += f"{'MC share':>10}"
    print(header)
    print("-" * len(header))

    for hero_name, villain_name in MATCHUPS:
        hero, villain = make_agent(hero_name), make_agent(villain_name)
        profiler = cProfile.Profile() if args.profile else None
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        match = run_match(hero, villain, args.deals, args.seed)
        if profiler:
            profiler.disable()
        elapsed = time.perf_counter() - start

        hands = len(match.hands)
        decisions = sum(len(h.result.decisions) for h in match.hands)
        line = (
            f"{hero_name + ' vs ' + villain_name:<24}{hands:>7}{hands / elapsed:>10.1f}"
            f"{decisions / elapsed:>13.1f}{1000 * elapsed / decisions:>13.2f}{decisions / hands:>10.2f}"
        )
        if profiler:
            line += f"{equity_share(profiler):>10.1%}"
        print(line)

    if args.profile:
        print("\nNote: profiling adds overhead, so use the run without --profile for speed.")


if __name__ == "__main__":
    main()
