"""Tests for hybridpoker.calibration (probability-quality diagnostics)."""

import numpy as np
import pytest
from sklearn.calibration import calibration_curve

from hybridpoker.calibration import (
    brier_score,
    calibration_in_the_large,
    class_reliability,
    expected_calibration_error,
    one_vs_rest_ranking,
    reliability_table,
    top_label_reliability,
)


def test_brier_score_examples() -> None:
    perfect = np.eye(5)[[0, 1, 2]]
    assert brier_score(np.array([0, 1, 2]), perfect) == 0.0
    wrong = np.eye(5)[[1]]
    assert brier_score(np.array([0]), wrong) == pytest.approx(2.0)
    uniform = np.full((1, 5), 0.2)
    assert brier_score(np.array([3]), uniform) == pytest.approx(0.8 ** 2 + 4 * 0.2 ** 2)


def test_reliability_table_matches_scikit_learn_calibration_curve() -> None:
    rng = np.random.default_rng(0)
    confidence = rng.random(2000)
    outcome = rng.random(2000) < confidence ** 2
    table = reliability_table(confidence, outcome)
    observed, predicted = calibration_curve(outcome, confidence, n_bins=10, strategy="uniform")
    np.testing.assert_allclose([r["observed"] for r in table], observed)
    np.testing.assert_allclose([r["mean_predicted"] for r in table], predicted)
    assert sum(r["rows"] for r in table) == 2000


def test_perfectly_calibrated_bins_have_zero_error() -> None:
    table = [{"rows": 10, "mean_predicted": 0.3, "observed": 0.3}, {"rows": 30, "mean_predicted": 0.9, "observed": 0.9}]
    assert expected_calibration_error(table) == 0.0
    table[1]["observed"] = 0.7  # 30 rows off by 0.2 -> ECE = 30 * 0.2 / 40
    assert expected_calibration_error(table) == pytest.approx(0.15)


def test_overconfident_model_has_positive_top_label_error() -> None:
    y = np.array([0, 1, 0, 1])
    proba = np.tile([0.95, 0.05, 0, 0, 0], (4, 1))  # always 95% sure of class 0; right half the time
    result = top_label_reliability(y, proba)
    assert result["ece"] == pytest.approx(0.45)
    assert result["bins"][0]["observed"] == 0.5


def test_class_reliability_and_calibration_in_the_large() -> None:
    y = np.array([4, 1, 1, 1])
    proba = np.tile([0.0, 0.75, 0.0, 0.0, 0.25], (4, 1))
    per_class = class_reliability(y, proba)
    assert per_class["all_in"]["positives"] == 1
    assert per_class["all_in"]["ece"] == pytest.approx(0.0)  # predicted 0.25, observed 1 of 4
    large = calibration_in_the_large(y, proba)
    assert large["check_call"] == {"mean_predicted": 0.75, "observed": 0.75}


def test_one_vs_rest_ranking_skips_rare_classes() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, size=200)  # all_in never occurs
    proba = rng.dirichlet(np.ones(5), size=200)
    proba[np.arange(200), y] += 1.0
    proba /= proba.sum(axis=1, keepdims=True)
    ranking = one_vs_rest_ranking(y, proba)
    assert ranking["all_in"]["auc"] is None and ranking["all_in"]["positives"] == 0
    assert ranking["fold"]["auc"] > 0.8
    assert 0 < ranking["fold"]["base_rate"] < 1
