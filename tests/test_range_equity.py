"""Tests for hybridpoker.range_equity (equity against an estimated continuing range)."""

import numpy as np
import pytest
from pokerkit import Card

from hybridpoker.agents.strength import hand_class
from hybridpoker.equity import estimate_equity, exact_river_equity
from hybridpoker.range_equity import equity_vs_top_fraction, opponent_ranking, top_count


def cards(text: str) -> tuple:
    return tuple(Card.parse(text))


def test_ranking_excludes_blockers_and_lists_every_legal_hand() -> None:
    hero, board = cards("QsQd"), cards("Kc7h2d")
    ranking = opponent_ranking(hero, board)
    assert ranking.size == 47 * 46 // 2  # 1081 two-card hands from the 47 unseen cards
    seen = set(hero) | set(board)
    assert all(not (set(c) & seen) for c in ranking.combos)
    assert len(set(ranking.combos)) == ranking.size
    assert opponent_ranking(hero, ()).size == 50 * 49 // 2  # preflop: 1225


def test_preflop_ranking_puts_premium_pairs_first_and_trash_last() -> None:
    ranking = opponent_ranking(cards("2c3d"), ())
    assert hand_class(ranking.combos[0]) == "AA"
    assert hand_class(ranking.combos[-1]) in {"32o", "42o", "52o", "43o", "62o", "72o"}


def test_river_ranking_is_strongest_first_and_exact_against_all_hands() -> None:
    hero, board = cards("QsQd"), cards("Kc7h2d4s9c")
    ranking = opponent_ranking(hero, board)
    values = ranking.river_values
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    # Against every hand, the range equity equals Phase 1's exact river enumeration.
    assert equity_vs_top_fraction(hero, board, 1.0, samples=1, seed=0) == pytest.approx(
        exact_river_equity(hero, board).equity, abs=1e-12)


def test_equity_falls_as_the_range_gets_more_selective() -> None:
    hero, board = cards("QsQd"), cards("Kc7h2d4s9c")  # an underpair to the king
    e10, e50, e100 = (equity_vs_top_fraction(hero, board, f, samples=1, seed=0) for f in (0.1, 0.5, 1.0))
    assert e10 < e50 < e100
    assert e10 == 0.0  # the strongest 10% (sets, two pairs, kings) all beat queens


def test_the_nuts_win_against_any_range() -> None:
    hero, board = cards("AhKh"), cards("QhJhTh2c3d")  # royal flush
    for fraction in (0.01, 0.3, 1.0):
        assert equity_vs_top_fraction(hero, board, fraction, samples=1, seed=0) == 1.0


def test_preflop_full_range_matches_random_hand_equity_and_selective_ranges_are_tougher() -> None:
    aces, trash = cards("AsAd"), cards("7c2d")
    full = equity_vs_top_fraction(aces, (), 1.0, samples=2000, seed=1)
    random_hand = estimate_equity(aces, (), sample_count=2000, seed=2)
    assert full == pytest.approx(random_hand.equity, abs=4 * random_hand.standard_error)
    assert equity_vs_top_fraction(aces, (), 0.05, samples=1000, seed=3) < full
    assert equity_vs_top_fraction(trash, (), 0.3, samples=1000, seed=4) < \
        equity_vs_top_fraction(trash, (), 1.0, samples=1000, seed=4)


def test_flop_monte_carlo_is_seeded() -> None:
    hero, board = cards("QsQd"), cards("Kc7h2d")
    a = equity_vs_top_fraction(hero, board, 0.4, samples=150, seed=9)
    assert a == equity_vs_top_fraction(hero, board, 0.4, samples=150, seed=9)
    assert a != equity_vs_top_fraction(hero, board, 0.4, samples=150, seed=10)


def test_fraction_validation_and_top_count() -> None:
    ranking = opponent_ranking(cards("QsQd"), cards("Kc7h2d"))
    assert top_count(ranking, 1.0) == ranking.size
    assert top_count(ranking, 0.0) == 1  # at least one hand
    assert top_count(ranking, 0.5) == round(0.5 * ranking.size)
    with pytest.raises(ValueError):
        top_count(ranking, 1.5)


# --- Low-noise correction used by the agents (common random numbers) ---------------------

from hybridpoker.range_equity import range_adjustments  # noqa: E402


def test_river_adjustment_is_the_exact_difference() -> None:
    hero, board = cards("QsQd"), cards("Kc7h2d4s9c")
    full = equity_vs_top_fraction(hero, board, 1.0, samples=1, seed=0)
    adj = range_adjustments(hero, board, [0.2, 0.6, 1.0], samples=1, seed=0)
    for q in (0.2, 0.6):
        assert adj[q] == pytest.approx(equity_vs_top_fraction(hero, board, q, samples=1, seed=0) - full, abs=1e-12)
    assert adj[1.0] == pytest.approx(0.0, abs=1e-12)


def test_full_range_needs_no_correction_and_draws_no_samples() -> None:
    assert range_adjustments(cards("QsQd"), cards("Kc7h2d"), [1.0], samples=400, seed=1) == {1.0: 0.0}


def test_wide_ranges_get_small_low_noise_corrections() -> None:
    # The problem found in Stage 4B: independent 200-sample estimates had about +-0.03 noise,
    # larger than the true correction for a 94% range, and could even exceed the random-hand equity.
    hero, board = cards("QsQd"), cards("Kc7h2d")
    values = [range_adjustments(hero, board, [0.94], samples=400, seed=s)[0.94] for s in range(6)]
    assert max(values) <= 0.005  # removing the weakest 6% never helps the hero here
    assert float(np.std(values)) < 0.01


def test_narrow_ranges_are_topped_up_and_seeded() -> None:
    hero, board = cards("QsQd"), cards("Kc7h2d")
    a = range_adjustments(hero, board, [0.05, 0.5], samples=200, seed=7)
    assert a == range_adjustments(hero, board, [0.05, 0.5], samples=200, seed=7)
    assert a[0.05] < a[0.5] < 0  # tighter ranges are tougher for an underpair to the king
