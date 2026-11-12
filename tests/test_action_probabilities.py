"""Tests for hybridpoker.action_probabilities (class order, masking, clipping)."""

import numpy as np
import pandas as pd
import pytest

from hybridpoker.action_probabilities import (
    CLASS_ORDER,
    PROBABILITY_FLOOR,
    legal_mask,
    mask_and_renormalize,
)

ALL_LEGAL = [True] * 5
FACING_BET_NO_RAISE = [True, True, False, False, False]  # fold or call only
FREE_CHECK = [False, True, True, True, True]  # fold is never offered when checking is free


def test_class_order_is_fixed() -> None:
    assert CLASS_ORDER == ("fold", "check_call", "raise_half_pot", "raise_pot", "all_in")


def test_illegal_actions_get_zero_and_rows_sum_to_one() -> None:
    raw = np.array([[0.1, 0.2, 0.3, 0.3, 0.1], [0.5, 0.1, 0.2, 0.1, 0.1]])
    result = mask_and_renormalize(raw, np.array([FACING_BET_NO_RAISE, FREE_CHECK]))
    assert result.masked[0, 2:].sum() == 0 and result.masked[1, 0] == 0
    np.testing.assert_allclose(result.masked.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    # Legal probabilities keep their relative sizes: 0.1 : 0.2 -> 1/3 : 2/3.
    np.testing.assert_allclose(result.masked[0, :2], [1 / 3, 2 / 3], atol=1e-6)


def test_raw_probabilities_are_preserved_unchanged() -> None:
    raw = np.array([[0.0, 0.0, 0.6, 0.4, 0.0]])
    copy = raw.copy()
    result = mask_and_renormalize(raw, np.array([FACING_BET_NO_RAISE]))
    np.testing.assert_array_equal(raw, copy)  # caller's array untouched
    np.testing.assert_array_equal(result.raw, copy)  # kept for diagnostics
    assert not result.raw.flags.writeable
    assert result.illegal_mass[0] == pytest.approx(1.0)  # all raw mass was on illegal actions


def test_zero_legal_mass_becomes_uniform_over_legal_actions() -> None:
    result = mask_and_renormalize(np.array([[0.0, 0.0, 0.5, 0.5, 0.0]]), np.array([FACING_BET_NO_RAISE]))
    np.testing.assert_allclose(result.masked[0], [0.5, 0.5, 0, 0, 0])


def test_floor_keeps_legal_probabilities_positive_and_renormalizes() -> None:
    raw = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    result = mask_and_renormalize(raw, np.array([ALL_LEGAL]))
    assert (result.masked[0] > 0).all()
    assert result.masked.sum() == pytest.approx(1.0, abs=1e-12)
    expected_small = PROBABILITY_FLOOR / (1 + 4 * PROBABILITY_FLOOR)
    np.testing.assert_allclose(result.masked[0, [0, 2, 3, 4]], expected_small, rtol=1e-9)
    assert np.isfinite(np.log(result.masked)).all()  # log-loss stays finite


def test_floor_zero_keeps_exact_zeros() -> None:
    raw = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    result = mask_and_renormalize(raw, np.array([ALL_LEGAL]), floor=0.0)
    np.testing.assert_array_equal(result.masked, raw)


def test_predicted_is_most_probable_legal_action() -> None:
    raw = np.array([[0.0, 0.2, 0.0, 0.0, 0.8], [0.4, 0.1, 0.5, 0.0, 0.0]])
    result = mask_and_renormalize(raw, np.array([FACING_BET_NO_RAISE, FREE_CHECK]))
    assert list(result.predicted) == ["check_call", "raise_half_pot"]


@pytest.mark.parametrize(
    "raw, legal",
    [
        (np.ones((1, 4)), np.ones((1, 4), dtype=bool)),  # wrong number of classes
        (np.ones((1, 5)), np.ones((2, 5), dtype=bool)),  # shape mismatch
        (np.array([[-0.1, 0.5, 0.6, 0, 0]]), np.ones((1, 5), dtype=bool)),  # negative
        (np.array([[np.nan, 1, 0, 0, 0]]), np.ones((1, 5), dtype=bool)),  # not finite
        (np.ones((1, 5)), np.zeros((1, 5), dtype=bool)),  # no legal action
    ],
)
def test_invalid_input_is_rejected(raw, legal) -> None:
    with pytest.raises(ValueError):
        mask_and_renormalize(raw, legal)


def test_legal_mask_reads_public_flags_in_class_order(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    mask = legal_mask(frame)
    assert mask.shape == (len(frame), 5) and mask.dtype == bool
    taken = frame["act_category"].map(CLASS_ORDER.index).to_numpy()
    assert mask[np.arange(len(frame)), taken].all()  # the action actually taken was always legal
    single = legal_mask(pd.DataFrame([{f"obs_legal_{c}": c != "fold" for c in CLASS_ORDER}]))
    assert single.tolist() == [FREE_CHECK]
