"""Tests for hybridpoker.reporting (descriptive dataset tables)."""

import pandas as pd
import pytest

from hybridpoker.action_probabilities import CLASS_ORDER
from hybridpoker.reporting import (
    class_counts,
    class_shares,
    evaluation_set,
    format_counts_with_shares,
    legal_conditioned,
    md_table,
    per_match_share_range,
    share_differences,
)


def rows(spec):
    """spec: (match, split, scored, action, fold legal, all-in legal)."""
    return pd.DataFrame([
        {"id_match": m, "id_split": s, "id_scored": sc, "act_category": a,
         "obs_legal_fold": fl, "obs_legal_all_in": al, "family": "x" if m != "m3" else "y"}
        for m, s, sc, a, fl, al in spec
    ])


FRAME = rows([
    ("m1", "train", True, "check_call", False, True),
    ("m1", "train", True, "fold", True, True),
    ("m1", "train", True, "check_call", True, False),
    ("m1", "train", True, "all_in", True, True),
    ("m2", "test", True, "check_call", True, True),
    ("m2", "test", True, "fold", True, False),
    ("m3", "heldout", True, "raise_pot", False, True),
    ("m3", "heldout", False, "check_call", True, True),
])


def test_class_counts_and_shares() -> None:
    counts = class_counts(FRAME, "id_split").set_index("group")
    assert counts.loc["train", "rows"] == 4
    assert counts.loc["train", ["fold", "check_call", "all_in"]].tolist() == [1, 2, 1]
    shares = class_shares(FRAME).iloc[0]
    assert shares["check_call"] == pytest.approx(4 / 8)
    assert sum(shares[c] for c in CLASS_ORDER) == pytest.approx(1.0)
    assert format_counts_with_shares(FRAME).iloc[0]["fold"] == "2 (25.0%)"


def test_evaluation_set_keeps_only_scored_heldout_rows() -> None:
    sets = evaluation_set(FRAME)
    assert sets.tolist()[:6] == ["train"] * 4 + ["test"] * 2
    assert sets.iloc[6] == "heldout"
    assert pd.isna(sets.iloc[7])  # the unscored counterpart is not used


def test_legal_conditioned_rates() -> None:
    fold = legal_conditioned(FRAME, "fold").iloc[0]
    assert fold["legal rows"] == 6 and fold["chosen"] == 2
    assert fold["legal share"] == pytest.approx(6 / 8)
    assert fold["chosen | legal"] == pytest.approx(2 / 6)
    by_split = legal_conditioned(FRAME, "all_in", "id_split").set_index("group")
    assert by_split.loc["train", "chosen | legal"] == pytest.approx(1 / 3)
    assert by_split.loc["test", "chosen"] == 0


def test_legal_conditioned_rejects_illegal_choice_and_unknown_action() -> None:
    bad = FRAME.copy()
    bad.loc[0, "act_category"] = "fold"  # fold was illegal on row 0
    with pytest.raises(ValueError):
        legal_conditioned(bad, "fold")
    with pytest.raises(ValueError):
        legal_conditioned(FRAME, "shove")


def test_share_differences_against_train() -> None:
    diffs = share_differences(class_shares(FRAME, "id_split")).set_index("group")
    assert diffs.loc["train", "max |diff| (pp)"] == 0
    # test: fold 50% vs 25% in train -> +25 pp.
    assert diffs.loc["test", "fold"] == pytest.approx(25.0)
    with pytest.raises(ValueError):
        share_differences(class_shares(FRAME, "id_split"), reference="missing")


def test_per_match_share_range_and_markdown() -> None:
    ranges = per_match_share_range(FRAME, "family").set_index("group")
    assert ranges.loc["x", "matches"] == 2
    assert ranges.loc["x", "fold"] == "25.0%-50.0%"
    table = md_table(pd.DataFrame({"a": [1], "b": [0.5]}))
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | 0.500 |"]
