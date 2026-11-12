"""Run a seeded, seat-swapped match and write hand logs.

Examples:
    python scripts/run_match.py --villain calling_station --deals 500 --seed 1
    python scripts/run_match.py --villain all --deals 300 --seed 1

Logs are written to data/raw/simulations/<run-name>/ (ignored by git).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hybridpoker.agents import AGENT_NAMES, make_agent
from hybridpoker.hand_log import write_run
from hybridpoker.match import run_match

DEFAULT_OUT = Path("data/raw/simulations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hero", default="probability", choices=AGENT_NAMES)
    parser.add_argument("--villain", default="all", choices=(*AGENT_NAMES, "all"))
    parser.add_argument("--deals", type=int, default=200, help="deals per match")
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    parser.add_argument("--hero-samples", type=int, default=None, help="hero equity samples")
    parser.add_argument("--villain-samples", type=int, default=None, help="villain equity samples")
    parser.add_argument("--no-seat-swap", action="store_true", help="play each deal once")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="log directory")
    parser.add_argument("--no-log", action="store_true", help="do not write CSV logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    villains = [n for n in AGENT_NAMES if n != "probability"] if args.villain == "all" else [args.villain]

    print(f"hero={args.hero} deals={args.deals} seed={args.seed} seat_swap={not args.no_seat_swap}")
    header = f"{'villain':<16}{'bb/100':>9}{'95% CI':>20}{'SE paired':>11}{'SE naive':>10}{'hands/s':>9}{'dec/s':>8}"
    print(header)
    print("-" * len(header))

    for villain_name in villains:
        hero = make_agent(args.hero, args.hero_samples)
        villain = make_agent(villain_name, args.villain_samples)
        start = time.perf_counter()
        match = run_match(hero, villain, args.deals, args.seed, seat_swap=not args.no_seat_swap)
        elapsed = time.perf_counter() - start

        decisions = sum(len(h.result.decisions) for h in match.hands)
        paired = match.win_rate()
        naive = match.naive_win_rate()
        print(
            f"{villain_name:<16}{paired.bb_per_100:>+9.1f}"
            f"{f'[{paired.ci_low:+.1f}, {paired.ci_high:+.1f}]':>20}"
            f"{paired.standard_error:>11.1f}{naive.standard_error:>10.1f}"
            f"{len(match.hands) / elapsed:>9.1f}{decisions / elapsed:>8.1f}"
        )

        if not args.no_log:
            run_name = f"{args.hero}_vs_{villain_name}_seed{args.seed}_deals{args.deals}"
            timing = {"seconds": elapsed, "hands_per_second": len(match.hands) / elapsed}
            write_run(match, hero, villain, args.out / run_name, extra={"timing": timing})

    if not args.no_log:
        print(f"\nLogs written under {args.out}/")


if __name__ == "__main__":
    main()
