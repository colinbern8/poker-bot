"""Tests for hybridpoker.metrics (bb/100 and confidence intervals)."""

import math

import pytest

from hybridpoker.metrics import win_rate


def test_hand_computed_example() -> None:
    # Four deal pairs (2 hands each), big blind 2.
    # bb/100 per pair = net / 2 hands / 2 chips * 100 = [250, -150, 100, 0].
    result = win_rate([10, -6, 4, 0], hands_per_unit=2, big_blind=2)

    assert result.bb_per_100 == pytest.approx(50.0)
    # Deviations 200, -200, 50, -50 -> sample variance 85000 / 3.
    expected_se = math.sqrt(85000 / 3 / 4)
    assert result.standard_error == pytest.approx(expected_se)
    assert result.ci_low == pytest.approx(50 - 1.959964 * expected_se, rel=1e-6)
    assert result.ci_high == pytest.approx(50 + 1.959964 * expected_se, rel=1e-6)
    assert result.unit_count == 4
    assert result.hands == 8


def test_single_hands_as_units() -> None:
    result = win_rate([2, -2, 2, -2], hands_per_unit=1, big_blind=2)
    assert result.bb_per_100 == 0.0
    assert result.hands == 4


def test_identical_results_have_zero_uncertainty() -> None:
    result = win_rate([4, 4, 4], hands_per_unit=2, big_blind=2)
    assert result.bb_per_100 == pytest.approx(100.0)
    assert result.standard_error == 0.0
    assert result.ci_low == result.ci_high == pytest.approx(100.0)


def test_other_confidence_levels() -> None:
    narrow = win_rate([10, -6, 4, 0], 2, 2, confidence=0.80)
    wide = win_rate([10, -6, 4, 0], 2, 2, confidence=0.99)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unit_net_chips": [5]},
        {"unit_net_chips": [1, 2], "hands_per_unit": 0},
        {"unit_net_chips": [1, 2], "big_blind": 0},
        {"unit_net_chips": [1, 2], "confidence": 1.0},
    ],
)
def test_invalid_inputs_are_rejected(kwargs: dict) -> None:
    arguments = {"hands_per_unit": 2, "big_blind": 2} | kwargs
    with pytest.raises(ValueError):
        win_rate(**arguments)


def test_string_form() -> None:
    text = str(win_rate([10, -6, 4, 0], 2, 2))
    assert text.startswith("+50.0 bb/100 [")
    assert "95% CI, 4 units x 2 hands" in text
