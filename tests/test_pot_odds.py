"""Tests for hybridpoker.pot_odds.

Convention: pot_size includes the opponent's bet but excludes the hero's
pending call.
"""

import math

import pytest

from hybridpoker.pot_odds import call_expected_value, pot_odds

# --- pot_odds ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("pot_size", "call_amount", "expected"),
    [
        (30, 10, 0.25),  # pot 20 + bet 10 = 30; call 10 into final pot 40
        (20, 10, 1 / 3),  # pot 10 + pot-sized bet 10; call 10 into 30
        (15, 5, 0.25),  # pot 10 + half-pot bet 5; call 5 into 20
        (3, 1, 0.25),  # heads-up preflop: small blind completes 1 into 3 + 1
        (100, 0, 0.0),  # nothing to call: checking is free
        (2.5, 0.5, 0.5 / 3.0),  # fractional chip amounts are allowed
    ],
)
def test_pot_odds_examples(pot_size: float, call_amount: float, expected: float) -> None:
    assert pot_odds(pot_size, call_amount) == pytest.approx(expected)


def test_pot_odds_is_at_most_one_half() -> None:
    # The pot already contains the opponent's bet, so the call can never
    # exceed the pot and the price can never exceed half of the final pot.
    assert pot_odds(pot_size=10, call_amount=10) == pytest.approx(0.5)


def test_call_larger_than_pot_is_rejected() -> None:
    # A call bigger than the pot means the opponent's bet was left out of
    # pot_size, which breaks the project-wide convention.
    with pytest.raises(ValueError, match="must already include the bet"):
        pot_odds(pot_size=20, call_amount=30)
    with pytest.raises(ValueError, match="must already include the bet"):
        call_expected_value(0.5, pot_size=20, call_amount=30)


# --- call_expected_value -----------------------------------------------------


def test_call_ev_worked_example() -> None:
    # 40% equity, pot 30, call 10: 0.4 * 40 - 10 = +6 chips.
    assert call_expected_value(0.4, 30, 10) == pytest.approx(6.0)
    # Same spot, 20% equity: 0.2 * 40 - 10 = -2 chips.
    assert call_expected_value(0.2, 30, 10) == pytest.approx(-2.0)


@pytest.mark.parametrize(("pot_size", "call_amount"), [(30, 10), (20, 10), (3, 1), (7.5, 2.5)])
def test_call_ev_is_zero_exactly_at_pot_odds(pot_size: float, call_amount: float) -> None:
    break_even = pot_odds(pot_size, call_amount)
    assert call_expected_value(break_even, pot_size, call_amount) == pytest.approx(0.0, abs=1e-12)
    assert call_expected_value(break_even + 0.01, pot_size, call_amount) > 0
    assert call_expected_value(break_even - 0.01, pot_size, call_amount) < 0


def test_call_ev_matches_win_minus_lose_form() -> None:
    equity, pot, call = 0.37, 45, 15
    assert call_expected_value(equity, pot, call) == pytest.approx(
        equity * pot - (1 - equity) * call
    )


def test_call_ev_extremes() -> None:
    assert call_expected_value(1.0, 30, 10) == pytest.approx(30.0)  # win the whole pot
    assert call_expected_value(0.0, 30, 10) == pytest.approx(-10.0)  # lose the call
    assert call_expected_value(0.5, 30, 0) == pytest.approx(15.0)  # free check, half the pot


# --- Validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("pot_size", "call_amount", "error"),
    [
        (0, 10, ValueError),
        (-5, 10, ValueError),
        (30, -1, ValueError),
        (math.inf, 10, ValueError),
        (30, math.nan, ValueError),
        ("30", 10, TypeError),
        (30, True, TypeError),
        (None, 10, TypeError),
    ],
)
def test_invalid_amounts_are_rejected(pot_size: object, call_amount: object, error: type) -> None:
    with pytest.raises(error):
        pot_odds(pot_size, call_amount)  # type: ignore[arg-type]
    with pytest.raises(error):
        call_expected_value(0.5, pot_size, call_amount)  # type: ignore[arg-type]


@pytest.mark.parametrize("equity", [-0.01, 1.01, math.nan])
def test_invalid_equity_is_rejected(equity: float) -> None:
    with pytest.raises(ValueError):
        call_expected_value(equity, 30, 10)
