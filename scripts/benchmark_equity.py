"""Measure how fast the Monte Carlo equity engine runs on this machine.

Usage:
    python scripts/benchmark_equity.py

Prints samples per second for typical spots. Timing depends on the
computer, so record the machine when reporting results.
"""

import platform
import time

from hybridpoker.equity import estimate_equity, exact_river_equity

SAMPLE_COUNT = 5000
SPOTS = [
    ("Preflop, 1 opponent", "AsKh", "", 1),
    ("Flop, 1 opponent", "AsKh", "Td9d2c", 1),
    ("River, 1 opponent", "AsKh", "Td9d2c4s7h", 1),
    ("Preflop, 5 opponents", "AsKh", "", 5),
]


def main() -> None:
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"{SAMPLE_COUNT} samples per spot\n")
    for label, hole, board, opponents in SPOTS:
        start = time.perf_counter()
        result = estimate_equity(
            hole, board, opponent_count=opponents, sample_count=SAMPLE_COUNT, seed=0
        )
        elapsed = time.perf_counter() - start
        print(
            f"{label:<22} {SAMPLE_COUNT / elapsed:>7.0f} samples/s   "
            f"equity={result.equity:.3f} +/- {result.standard_error:.3f}"
        )

    start = time.perf_counter()
    exact_river_equity("AsKh", "Td9d2c4s7h")
    print(f"\nExact river equity (990 hands): {time.perf_counter() - start:.3f} s")


if __name__ == "__main__":
    main()
