"""The latency summary printed by the opt-in acceptance tests (median, p95, max, targets)."""

from __future__ import annotations

import pytest

from demo_support import P95_TARGET_SECONDS, STARTUP_TARGET_SECONDS, latency_report, p95, startup_report


def test_p95_uses_the_nearest_rank_method() -> None:
    assert p95([0.1] * 19 + [5.0]) == 0.1            # 19 of 20 values (95%) are 0.1
    assert p95([i / 100 for i in range(1, 101)]) == 0.95
    assert p95([0.3, 0.1, 0.2]) == 0.3                # small samples: p95 is the maximum
    assert p95([0.7]) == 0.7
    with pytest.raises(ValueError):
        p95([])


def test_latency_report_gives_median_p95_max_and_the_target_status() -> None:
    fast = [0.5] * 56 + [1.5] * 4   # 4 of 60 (6.7%) are slow, so the 57th value (p95) is slow
    line = latency_report(fast, "seed 1")
    assert line.startswith("seed 1: 60 AI decisions; median 500 ms, p95 1500 ms, max 1500 ms")
    assert "p95 500 ms" in latency_report([0.5] * 60 + [1.5] * 3, "x")  # 3 of 63 (4.8%) do not reach p95
    assert f"target p95 < {1000 * P95_TARGET_SECONDS:.0f} ms: met" in line and "close to the max" not in line

    slow = latency_report([2.5] * 30, "app")
    assert "p95 2500 ms" in slow and "NOT met (a target, not a test failure)" in slow

    few = latency_report([0.2, 0.4, 0.9], "few")
    assert "only 3 decisions, so p95 is close to the max" in few


def test_startup_report_compares_with_the_target_without_failing() -> None:
    assert startup_report(42.0, "startup").endswith(f"target < {STARTUP_TARGET_SECONDS:.0f} s: met")
    assert "NOT met (a target, not a test failure)" in startup_report(120.0, "startup")
