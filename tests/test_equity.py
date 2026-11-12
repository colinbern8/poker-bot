"""Tests for hybridpoker.equity.

Statistical tests compare Monte Carlo estimates to exact answers computed
by enumeration, never to equity numbers quoted from memory. All random
tests use fixed seeds, so they are deterministic.
"""

import math

import pytest

from hybridpoker.equity import (
    EquityResult,
    estimate_equity,
    exact_river_equity,
    parse_cards,
    summarize_pot_shares,
)

# Tolerance for "Monte Carlo agrees with the exact answer": 4 standard
# errors. With fixed seeds the tests are deterministic; 4 SE also means a
# correct implementation would fail by chance less than 0.01% of the time
# if the seeds were changed.
Z_TOLERANCE = 4.0

ROYAL_FLUSH_BOARD = "AhKhQhJhTh"


def assert_rates_consistent(result: EquityResult) -> None:
    assert result.win_rate + result.tie_rate + result.loss_rate == pytest.approx(1.0)
    assert 0.0 <= result.equity <= 1.0
    # Equity lies between "count wins only" and "count ties as wins".
    assert result.win_rate <= result.equity + 1e-12
    assert result.equity <= result.win_rate + result.tie_rate + 1e-12


# --- Card parsing and validation ------------------------------------------


def test_parse_cards_accepts_text_with_or_without_spaces() -> None:
    assert parse_cards("AsKh") == parse_cards("As Kh")
    assert len(parse_cards("Td9d2c")) == 3
    assert parse_cards("") == ()


@pytest.mark.parametrize("bad_text", ["as", "Xx", "A", "AsK", "??"])
def test_parse_cards_rejects_invalid_text(bad_text: str) -> None:
    with pytest.raises(ValueError):
        parse_cards(bad_text)


@pytest.mark.parametrize(
    ("hole", "board", "message"),
    [
        ("As", "", "2 hole cards"),
        ("AsKhQd", "", "2 hole cards"),
        ("AsKh", "2c3d", "Board must have"),
        ("AsKh", "2c3d4h5s6c7d", "Board must have"),
        ("AsAs", "", "Duplicate"),
        ("AsKh", "As2c3d", "Duplicate"),
    ],
)
def test_invalid_hands_are_rejected(hole: str, board: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_equity(hole, board, seed=0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"opponent_count": 0}, ValueError),
        ({"opponent_count": 10}, ValueError),
        ({"opponent_count": 1.5}, TypeError),
        ({"opponent_count": True}, TypeError),
        ({"sample_count": 0}, ValueError),
        ({"sample_count": 100.0}, TypeError),
        ({"seed": None}, TypeError),
        ({"seed": "42"}, TypeError),
    ],
)
def test_invalid_simulation_settings_are_rejected(kwargs: dict, error: type) -> None:
    settings = {"seed": 0, "sample_count": 10} | kwargs
    with pytest.raises(error):
        estimate_equity("AsKh", **settings)


def test_seed_is_required() -> None:
    with pytest.raises(TypeError):
        estimate_equity("AsKh")  # type: ignore[call-arg]


# --- Split pots and the standard error formula -----------------------------


@pytest.mark.parametrize("opponent_count", [1, 2, 3, 5, 9])
def test_royal_flush_board_splits_pot_between_all_players(opponent_count: int) -> None:
    # Everyone plays the board, so every outcome is an (opponents + 1)-way tie.
    result = estimate_equity(
        "2c3d",
        ROYAL_FLUSH_BOARD,
        opponent_count=opponent_count,
        sample_count=50,
        seed=0,
    )
    assert result.equity == pytest.approx(1 / (opponent_count + 1))
    assert result.tie_rate == 1.0
    assert result.win_rate == 0.0
    assert result.loss_rate == 0.0
    # Every outcome has the same pot share, so there is no uncertainty.
    # (A binary win/loss variable would get this wrong.)
    assert result.standard_error == pytest.approx(0.0, abs=1e-12)


def test_royal_flush_board_exact_equity_is_half() -> None:
    result = exact_river_equity("2c3d", ROYAL_FLUSH_BOARD)
    assert result.equity == pytest.approx(0.5)
    assert result.tie_rate == 1.0


# --- Hand-counted river examples ---------------------------------------------
#
# Each expected value below was worked out by counting cards on paper, not by
# running the engine. On the river, the hero sees 7 cards, so 45 are unseen
# and the opponent holds one of C(45, 2) = 990 equally likely hands.
# Each case lists (wins, ties, losses) out of 990.

HAND_COUNTED_RIVER_CASES = [
    pytest.param(
        # Quad nines on board; the hero's ace is the best possible kicker.
        # Opponent ties only by also holding an ace (3 unseen aces):
        #   ties   = 990 - C(42, 2) = 990 - 861 = 129
        #   wins   = C(42, 2)       = 861
        "AsKd", "9c9d9h9s2c", (861, 129, 0),
        id="quads-on-board-ace-kicker",
    ),
    pytest.param(
        # Quad nines on board; the hero plays the board's king kicker.
        # Any opponent ace (4 unseen) wins; everyone else ties:
        #   losses = 990 - C(41, 2) = 990 - 820 = 170
        #   ties   = C(41, 2)       = 820
        "2s3s", "9c9d9h9sKc", (0, 820, 170),
        id="quads-on-board-king-kicker",
    ),
    pytest.param(
        # Board full house AAAKK, the best possible full house. The hero plays
        # the board. Only quads beat it: the last ace (Ac with any of 44 other
        # cards = 44 hands) or the last two kings (KhKs = 1 hand).
        #   losses = 44 + 1 = 45, ties = 945
        "5c5d", "AhAdAsKcKd", (0, 945, 45),
        id="board-full-house-aces-full",
    ),
    pytest.param(
        # Board is an ace-high straight with only two hearts and no pair:
        # no flush or full house is possible and nothing beats a broadway
        # straight, so all 990 opponent hands tie.
        "2c3d", "AhKdQcJsTh", (0, 990, 0),
        id="board-broadway-straight",
    ),
]


@pytest.mark.parametrize(("hole", "board", "counts"), HAND_COUNTED_RIVER_CASES)
def test_exact_river_equity_matches_hand_counts(
    hole: str, board: str, counts: tuple[int, int, int]
) -> None:
    wins, ties, losses = counts
    assert wins + ties + losses == 990
    result = exact_river_equity(hole, board)

    assert result.equity == pytest.approx((wins + ties / 2) / 990)
    assert result.win_rate == pytest.approx(wins / 990)
    assert result.tie_rate == pytest.approx(ties / 990)
    assert result.loss_rate == pytest.approx(losses / 990)


@pytest.mark.parametrize(("hole", "board", "counts"), HAND_COUNTED_RIVER_CASES)
def test_monte_carlo_matches_hand_counts(
    hole: str, board: str, counts: tuple[int, int, int]
) -> None:
    wins, ties, _ = counts
    expected = (wins + ties / 2) / 990
    estimate = estimate_equity(hole, board, sample_count=2000, seed=11)

    if estimate.standard_error == 0:
        # Every outcome has the same share (e.g. all ties).
        assert estimate.equity == pytest.approx(expected)
    else:
        z_score = (estimate.equity - expected) / estimate.standard_error
        assert abs(z_score) < Z_TOLERANCE


def test_standard_error_uses_fractional_pot_shares() -> None:
    # Shares: win, 2-way tie, loss, 2-way tie.
    shares = [1.0, 0.5, 0.0, 0.5]
    result = summarize_pot_shares(shares, opponent_count=1, exact=False)

    mean = 0.5
    sample_variance = sum((s - mean) ** 2 for s in shares) / (len(shares) - 1)
    assert result.equity == pytest.approx(mean)
    assert result.standard_error == pytest.approx(math.sqrt(sample_variance / 4))
    assert (result.win_rate, result.tie_rate, result.loss_rate) == (0.25, 0.5, 0.25)


def test_three_way_tie_share_counts_as_one_third() -> None:
    result = summarize_pot_shares([1 / 3, 1 / 3, 1.0], opponent_count=2, exact=False)
    assert result.equity == pytest.approx((1 / 3 + 1 / 3 + 1) / 3)
    assert result.tie_rate == pytest.approx(2 / 3)


def test_single_sample_has_undefined_standard_error() -> None:
    result = estimate_equity("AsKh", sample_count=1, seed=0)
    assert math.isnan(result.standard_error)


# --- Exact river equity ----------------------------------------------------


def test_exact_river_equity_enumerates_all_990_opponent_hands() -> None:
    result = exact_river_equity("AsKd", "Ah7c2d9sJh")
    assert result.sample_count == 990  # C(45, 2)
    assert result.exact is True
    assert result.standard_error == 0.0
    assert_rates_consistent(result)


def test_hero_with_the_nuts_has_full_equity() -> None:
    # Hero holds a royal flush; no opponent hand can tie or beat it.
    exact = exact_river_equity("AhKh", "QhJhTh2c3d")
    assert exact.equity == 1.0
    assert exact.win_rate == 1.0

    simulated = estimate_equity("AhKh", "QhJhTh2c3d", opponent_count=3, sample_count=200, seed=1)
    assert simulated.equity == 1.0


def test_exact_river_equity_requires_full_board() -> None:
    with pytest.raises(ValueError, match="full 5-card board"):
        exact_river_equity("AsKd", "Ah7c2d")


@pytest.mark.parametrize(
    ("hole", "board"),
    [
        ("AsKd", "Ah7c2d9sJh"),  # strong one-pair hand
        ("Td9d", "8d7c2s3hKh"),  # missed draw
        ("2c3d", "AhKhQhJh5s"),  # very weak, a few ties
        ("5c5d", "AhAdAsKcKd"),  # board plays: mostly split pots
    ],
)
def test_monte_carlo_matches_exact_river_equity(hole: str, board: str) -> None:
    exact = exact_river_equity(hole, board)
    estimate = estimate_equity(hole, board, sample_count=3000, seed=2026)

    assert_rates_consistent(estimate)
    assert estimate.standard_error > 0
    z_score = (estimate.equity - exact.equity) / estimate.standard_error
    assert abs(z_score) < Z_TOLERANCE, (
        f"Monte Carlo {estimate.equity:.4f} vs exact {exact.equity:.4f} "
        f"(z = {z_score:.2f})"
    )


# --- Reproducibility --------------------------------------------------------


def test_same_seed_gives_identical_results() -> None:
    first = estimate_equity("AsKh", "Td9d2c", opponent_count=2, sample_count=300, seed=7)
    second = estimate_equity("AsKh", "Td9d2c", opponent_count=2, sample_count=300, seed=7)
    assert first == second


def test_different_seeds_give_different_samples() -> None:
    first = estimate_equity("AsKh", sample_count=300, seed=1)
    second = estimate_equity("AsKh", sample_count=300, seed=2)
    assert first != second


def test_seeded_results_are_identical_across_platforms() -> None:
    # Regression values recorded on Linux with Python 3.12. GitHub Actions
    # checks the same numbers on Windows. If the sampling code changes on
    # purpose, these values must be re-recorded.
    assert estimate_equity("AsAh", sample_count=500, seed=2026).equity == 0.833
    assert (
        estimate_equity("Td9d", "8d7c2s", opponent_count=2, sample_count=300, seed=2026).equity
        == 0.46
    )


# --- Preflop sanity checks ---------------------------------------------------


def test_preflop_hand_strength_ordering() -> None:
    settings = {"sample_count": 2000, "seed": 3}
    aces = estimate_equity("AsAh", **settings)
    kings = estimate_equity("KsKh", **settings)
    seven_deuce = estimate_equity("7c2d", **settings)

    for result in (aces, kings, seven_deuce):
        assert_rates_consistent(result)
    assert aces.equity > kings.equity > seven_deuce.equity


def test_more_opponents_lowers_equity() -> None:
    equities = [
        estimate_equity("AsAh", opponent_count=count, sample_count=1500, seed=3).equity
        for count in (1, 2, 4)
    ]
    assert equities[0] > equities[1] > equities[2]


def test_card_objects_and_text_give_same_result() -> None:
    from_text = estimate_equity("AsKh", "Td9d2c", sample_count=100, seed=5)
    from_cards = estimate_equity(
        parse_cards("AsKh"), parse_cards("Td9d2c"), sample_count=100, seed=5
    )
    assert from_text == from_cards
