"""Tests for hybridpoker.features (Feature Set A)."""

import numpy as np
import pandas as pd
import pytest

from hybridpoker.features import (
    A_FEATURES,
    A_INPUT_COLUMNS,
    FEATURE_GROUPS_A,
    board_texture,
    feature_set_a,
    history_summary,
    parse_board,
    parse_history,
)


# --- Board texture (hand-worked examples) ---------------------------------------------------


def test_empty_board_is_all_zero() -> None:
    assert set(board_texture("").values()) == {0}


def test_board_texture_flop_example() -> None:
    # K-J-5 rainbow: no pair, 1 card per suit, K and J broadway.
    t = board_texture("KhJc5s")
    assert t["a_board_cards"] == 3
    assert t["a_board_high_rank"] == 13
    assert t["a_board_broadway"] == 2
    assert t["a_board_paired"] == 0 and t["a_board_trips"] == 0
    assert t["a_board_max_suit"] == 1
    assert t["a_board_straightness"] == 2  # J and K fit in the window 9..K (or T..A)
    assert t["a_board_rank_spread"] == 13 - 5


def test_board_texture_paired_monotone_and_trips() -> None:
    t = board_texture("8h8d8s2h")
    assert t["a_board_paired"] == 1 and t["a_board_trips"] == 1
    assert t["a_board_max_suit"] == 2
    t = board_texture("2h7h9hKh")
    assert t["a_board_max_suit"] == 4 and t["a_board_paired"] == 0


def test_straightness_counts_ace_low_and_high() -> None:
    assert board_texture("As2d3c4h")["a_board_straightness"] == 4  # wheel draw A-2-3-4
    assert board_texture("AsKdQcJh")["a_board_straightness"] == 4  # broadway draw
    assert board_texture("9s8d7c6h5d")["a_board_straightness"] == 5


def test_parse_board_rejects_malformed_text() -> None:
    assert parse_board("Td") == [(10, "d")]
    with pytest.raises(ValueError):
        parse_board("Tdx")


# --- Betting history -----------------------------------------------------------------------


def test_parse_history_and_relative_features() -> None:
    text = "preflop:button:raise_pot:6;preflop:big_blind:raise_half_pot:14;preflop:button:check_call:14"
    assert parse_history(text)[0] == ("preflop", "button", "raise_pot", 6)
    as_bb = history_summary(text, "big_blind")
    assert as_bb["a_actor_raises_hand"] == 1 and as_bb["a_opp_raises_hand"] == 1
    assert as_bb["a_preflop_aggressor_actor"] == 1  # the big blind made the last preflop raise
    assert as_bb["a_preflop_aggressor_opp"] == 0
    assert as_bb["a_prev_check_call"] == 1 and as_bb["a_prev_none"] == 0
    assert as_bb["a_streets_with_raise"] == 1
    as_button = history_summary(text, "button")
    assert as_button["a_preflop_aggressor_opp"] == 1


def test_empty_history() -> None:
    summary = history_summary("", "button")
    assert summary["a_prev_none"] == 1
    assert summary["a_actor_raises_hand"] == summary["a_opp_raises_hand"] == 0
    assert summary["a_preflop_aggressor_actor"] == summary["a_preflop_aggressor_opp"] == 0


def test_history_rejects_unknown_category() -> None:
    with pytest.raises(ValueError):
        parse_history("preflop:button:shove:200")


# --- Feature Set A on real decision rows -----------------------------------------------------


def test_feature_set_a_shape_order_and_no_missing_values(tiny_dataset) -> None:
    features = feature_set_a(tiny_dataset.frame)
    assert list(features.columns) == list(A_FEATURES)
    assert features.index.equals(tiny_dataset.frame.index)
    assert not features.isna().any().any()
    assert all(dtype == np.float64 for dtype in features.dtypes)
    assert sorted(f for group in FEATURE_GROUPS_A.values() for f in group) == sorted(A_FEATURES)


def test_first_decision_is_button_facing_the_big_blind(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    first = frame[frame["id_decision_index"] == 0]
    features = feature_set_a(first)
    assert (features["a_street_preflop"] == 1).all()
    assert (features["a_position_button"] == 1).all()
    assert (features["a_to_call_bb"] == 0.5).all()  # 1 chip = half a big blind
    assert (features["a_pot_bb"] == 1.5).all()
    assert (features["a_facing_bet"] == 1).all()
    assert (features["a_prev_none"] == 1).all()
    assert (features["a_board_cards"] == 0).all()


def test_legal_features_match_observed_mask(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    features = feature_set_a(frame)
    for category in ("fold", "check_call", "raise_half_pot", "raise_pot", "all_in"):
        legal = frame[f"obs_legal_{category}"].astype(bool)
        assert (features[f"a_legal_{category}"] == legal.astype(int)).all()
        assert (features.loc[~legal, f"a_bet_to_{category}_bb"] == 0).all()
    # Folding is never offered when checking is free.
    assert (features.loc[features["a_facing_bet"] == 0, "a_legal_fold"] == 0).all()


def test_feature_set_a_reads_only_listed_obs_columns(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    only_inputs = frame.loc[:, list(A_INPUT_COLUMNS)]
    pd.testing.assert_frame_equal(feature_set_a(frame), feature_set_a(only_inputs))
    assert all(c.startswith("obs_") for c in A_INPUT_COLUMNS)
    with pytest.raises(KeyError):
        feature_set_a(only_inputs.drop(columns=["obs_board"]))
