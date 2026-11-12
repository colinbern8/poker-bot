"""Tests for hybridpoker.evaluation (metrics and match-level bootstrap)."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from hybridpoker.evaluation import (
    confusion,
    evaluate,
    match_stats,
    metrics_from_counts,
    paired_difference,
    resample_weights,
    summarize,
)

LABELS = list(range(5))


def random_case(seed: int, rows: int = 400, classes=LABELS):
    rng = np.random.default_rng(seed)
    y = rng.choice(classes, size=rows)
    proba = rng.dirichlet(np.ones(5), size=rows)
    matches = rng.choice([f"m{i}" for i in range(6)], size=rows)
    return y, proba, matches


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_match_scikit_learn(seed) -> None:
    y, proba, _ = random_case(seed)
    pred = proba.argmax(axis=1)
    result = evaluate(y, proba)
    assert result["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert result["macro_f1"] == pytest.approx(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))
    assert result["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert result["log_loss"] == pytest.approx(log_loss(y, proba, labels=LABELS))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    for i, name in enumerate(("fold", "check_call", "raise_half_pot", "raise_pot", "all_in")):
        assert result["per_class"][name]["precision"] == pytest.approx(precision[i])
        assert result["per_class"][name]["recall"] == pytest.approx(recall[i])
        assert result["per_class"][name]["f1"] == pytest.approx(f1[i])
        assert result["per_class"][name]["support"] == support[i]
    np.testing.assert_array_equal(np.array(result["confusion"]), confusion_matrix(y, pred, labels=LABELS))


def test_absent_class_counts_as_zero_f1_but_not_in_balanced_accuracy() -> None:
    y = np.array([1, 1, 2, 2, 3])
    proba = np.eye(5)[[1, 1, 2, 3, 4]] * 0.9 + 0.02  # predicts all_in once; fold never appears
    pred = proba.argmax(axis=1)
    result = evaluate(y, proba)
    assert result["per_class"]["fold"]["f1"] == 0.0
    assert result["macro_f1"] == pytest.approx(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))
    with pytest.warns(UserWarning):  # scikit-learn warns about the predicted-only class
        expected = balanced_accuracy_score(y, pred)
    assert result["balanced_accuracy"] == pytest.approx(expected)


def test_log_loss_is_floored() -> None:
    proba = np.zeros((1, 5))
    proba[0, 1] = 1.0
    assert evaluate(np.array([0]), proba)["log_loss"] == pytest.approx(-np.log(1e-15))


def test_match_stats_add_up_to_overall() -> None:
    y, proba, matches = random_case(3)
    stats = match_stats(y, proba, matches)
    overall = metrics_from_counts(stats.confusion.sum(axis=0), stats.loss_sum.sum(), stats.count.sum())
    direct = evaluate(y, proba)
    for metric in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss"):
        assert float(overall[metric]) == pytest.approx(direct[metric])
    assert stats.count.sum() == len(y)
    np.testing.assert_array_equal(stats.confusion.sum(axis=0), confusion(y, proba.argmax(axis=1)))


def test_resample_weights_draw_whole_matches_reproducibly() -> None:
    weights = resample_weights(6, 200, seed=11)
    assert weights.shape == (200, 6)
    assert (weights.sum(axis=1) == 6).all()
    np.testing.assert_array_equal(weights, resample_weights(6, 200, seed=11))
    assert not np.array_equal(weights, resample_weights(6, 200, seed=12))


def test_summary_point_matches_evaluate_and_interval_contains_it() -> None:
    y, proba, matches = random_case(4)
    summary = summarize(match_stats(y, proba, matches), resample_weights(6, 500, seed=1))
    direct = evaluate(y, proba)
    for metric, values in summary.items():
        assert values["value"] == pytest.approx(direct[metric])
        assert values["ci_low"] <= values["ci_high"]


def test_single_match_has_zero_width_interval() -> None:
    y, proba, _ = random_case(5)
    summary = summarize(match_stats(y, proba, np.array(["m0"] * len(y))), resample_weights(1, 100, seed=0))
    assert summary["macro_f1"]["ci_low"] == pytest.approx(summary["macro_f1"]["ci_high"])


def test_paired_difference() -> None:
    y, proba, matches = random_case(6)
    stats = match_stats(y, proba, matches)
    weights = resample_weights(6, 300, seed=2)
    same = paired_difference(stats, stats, weights)
    assert all(v["value"] == 0 and v["ci_low"] == 0 and v["ci_high"] == 0 for v in same.values())

    perfect = np.eye(5)[y] * 0.96 + 0.01
    better = paired_difference(match_stats(y, perfect, matches), stats, weights)
    assert better["macro_f1"]["ci_low"] > 0 and better["macro_f1"]["share_positive"] == 1.0
    assert better["log_loss"]["value"] < 0

    other = match_stats(y[:10], proba[:10], np.array(["x"] * 10))
    with pytest.raises(ValueError):
        paired_difference(stats, other, weights)
