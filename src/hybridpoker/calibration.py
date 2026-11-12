"""Probability-quality diagnostics: Brier score, reliability and calibration error.

All functions take probabilities of shape (rows, 5) in ``CLASS_ORDER`` and
integer true classes 0-4.

* **Brier score** (multiclass): mean over rows of the squared distance
  between the probability vector and the one-hot true class (0 = perfect,
  lower is better).
* **Top-label reliability**: rows are grouped by the model's confidence in
  its predicted class; in a well-calibrated model the accuracy in each
  group equals the mean confidence. **ECE** (expected calibration error)
  is the row-weighted mean absolute gap.
* **Per-class reliability**: for one class, rows are grouped by the
  predicted probability of that class and compared with how often the
  class actually occurred.
* **Calibration-in-the-large**: mean predicted probability of each class
  versus its observed frequency.
* **One-vs-rest AUC**: how well a class's probability ranks rows where the
  class occurred above rows where it did not (0.5 = no ranking ability),
  even if the class is never the most probable.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from hybridpoker.action_probabilities import CLASS_ORDER

N_CLASSES = len(CLASS_ORDER)
DEFAULT_BINS = np.linspace(0.0, 1.0, 11)


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(N_CLASSES)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability_table(confidence: np.ndarray, outcome: np.ndarray, edges: np.ndarray = DEFAULT_BINS) -> list[dict]:
    """Group rows by ``confidence`` into bins; mean confidence vs observed rate per bin.

    ``outcome`` is 1 where the event happened (prediction correct, or the
    class occurred) and 0 otherwise. Empty bins are left out.
    """
    confidence = np.asarray(confidence, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        members = index == b
        if members.any():
            rows.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "rows": int(members.sum()),
                         "mean_predicted": float(confidence[members].mean()),
                         "observed": float(outcome[members].mean())})
    return rows


def expected_calibration_error(table: list[dict]) -> float:
    total = sum(r["rows"] for r in table)
    return float(sum(r["rows"] * abs(r["mean_predicted"] - r["observed"]) for r in table) / total)


def top_label_reliability(y_true: np.ndarray, proba: np.ndarray, edges: np.ndarray = DEFAULT_BINS) -> dict:
    predicted = proba.argmax(axis=1)
    table = reliability_table(proba.max(axis=1), predicted == np.asarray(y_true), edges)
    return {"bins": table, "ece": expected_calibration_error(table)}


def class_reliability(y_true: np.ndarray, proba: np.ndarray, edges: np.ndarray = DEFAULT_BINS) -> dict:
    """Per-class reliability tables and one-vs-rest calibration error."""
    y_true = np.asarray(y_true, dtype=int)
    out = {}
    for c, name in enumerate(CLASS_ORDER):
        table = reliability_table(proba[:, c], y_true == c, edges)
        out[name] = {"bins": table, "ece": expected_calibration_error(table),
                     "positives": int((y_true == c).sum())}
    return out


def calibration_in_the_large(y_true: np.ndarray, proba: np.ndarray) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int)
    return {name: {"mean_predicted": float(proba[:, c].mean()), "observed": float(np.mean(y_true == c))}
            for c, name in enumerate(CLASS_ORDER)}


def one_vs_rest_ranking(y_true: np.ndarray, proba: np.ndarray, min_positives: int = 10) -> dict[str, dict]:
    """ROC AUC and average precision per class; skipped (None) with too few positives."""
    y_true = np.asarray(y_true, dtype=int)
    out = {}
    for c, name in enumerate(CLASS_ORDER):
        positives = int((y_true == c).sum())
        if positives < min_positives or positives == len(y_true):
            out[name] = {"positives": positives, "auc": None, "average_precision": None, "base_rate": positives / len(y_true)}
            continue
        out[name] = {"positives": positives, "auc": float(roc_auc_score(y_true == c, proba[:, c])),
                     "average_precision": float(average_precision_score(y_true == c, proba[:, c])),
                     "base_rate": positives / len(y_true)}
    return out
