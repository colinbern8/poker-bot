"""Evaluation metrics and match-level bootstrap confidence intervals.

All metrics come from a 5x5 confusion matrix plus a summed log-loss, in the
fixed ``CLASS_ORDER``:

* **accuracy**: share of correct predictions;
* **macro-F1**: unweighted mean of the five per-class F1 scores. A class
  with no correct predictions (including one never predicted) scores 0
  (scikit-learn's ``zero_division=0``), so rare classes are never ignored;
* **balanced accuracy**: mean recall over the classes present in the true
  labels (scikit-learn's definition);
* **log-loss**: mean of -log(probability given to the true class), with
  probabilities floored at 1e-15.

**Match-level bootstrap.** Matches, not rows, are the independent units
(rows in a match share opponents, cards and history). Each resample draws
whole matches with replacement and recomputes the metric from the summed
per-match confusion matrices. Paired comparisons (B vs A, model vs Dummy)
use the same resampled matches for both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybridpoker.action_probabilities import CLASS_ORDER

N_CLASSES = len(CLASS_ORDER)
LOG_FLOOR = 1e-15
METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "log_loss")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """5x5 counts; rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)


def log_loss_terms(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """-log(probability of the true class), floored at ``LOG_FLOOR``."""
    p_true = proba[np.arange(len(y_true)), np.asarray(y_true, dtype=int)]
    return -np.log(np.maximum(p_true, LOG_FLOOR))


def per_class_scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support for each class (works on stacks of matrices)."""
    cm = np.asarray(cm, dtype=float)
    true_positive = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, true_positive / predicted, 0.0)
        recall = np.where(support > 0, true_positive / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def metrics_from_counts(cm: np.ndarray, loss_sum, count) -> dict[str, np.ndarray]:
    """The four headline metrics from confusion matrices (single or stacked)."""
    cm = np.asarray(cm, dtype=float)
    scores = per_class_scores(cm)
    present = scores["support"] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        balanced = np.where(present, scores["recall"], 0.0).sum(axis=-1) / present.sum(axis=-1)
        accuracy = np.diagonal(cm, axis1=-2, axis2=-1).sum(axis=-1) / cm.sum(axis=(-2, -1))
    return {
        "accuracy": accuracy,
        "macro_f1": scores["f1"].mean(axis=-1),
        "balanced_accuracy": balanced,
        "log_loss": np.asarray(loss_sum, dtype=float) / np.asarray(count, dtype=float),
    }


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Headline metrics, per-class scores and the confusion matrix.

    The predicted class is the most probable class of ``proba`` (pass
    masked probabilities to get legal predictions).
    """
    y_true = np.asarray(y_true, dtype=int)
    cm = confusion(y_true, proba.argmax(axis=1))
    headline = metrics_from_counts(cm, log_loss_terms(y_true, proba).sum(), len(y_true))
    scores = per_class_scores(cm)
    return {
        **{k: float(v) for k, v in headline.items()},
        "rows": int(len(y_true)),
        "per_class": {
            name: {k: float(scores[k][i]) for k in ("precision", "recall", "f1", "support")}
            for i, name in enumerate(CLASS_ORDER)
        },
        "confusion": cm.tolist(),
    }


@dataclass(frozen=True)
class MatchStats:
    """Per-match sufficient statistics for bootstrapping."""

    matches: tuple[str, ...]
    confusion: np.ndarray  # (matches, 5, 5)
    loss_sum: np.ndarray  # (matches,)
    count: np.ndarray  # (matches,)


def match_stats(y_true: np.ndarray, proba: np.ndarray, match_ids: np.ndarray) -> MatchStats:
    y_true = np.asarray(y_true, dtype=int)
    match_ids = np.asarray(match_ids)
    losses = log_loss_terms(y_true, proba)
    predictions = proba.argmax(axis=1)
    names = tuple(sorted(set(match_ids.tolist())))
    cms, sums, counts = [], [], []
    for name in names:
        rows = match_ids == name
        cms.append(confusion(y_true[rows], predictions[rows]))
        sums.append(losses[rows].sum())
        counts.append(rows.sum())
    return MatchStats(names, np.array(cms), np.array(sums), np.array(counts))


def resample_weights(match_count: int, resamples: int, seed: int) -> np.ndarray:
    """(resamples, matches) multiplicity of each match in each bootstrap resample."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, match_count, size=(resamples, match_count))
    return np.stack([np.bincount(row, minlength=match_count) for row in draws])


def bootstrap_metrics(stats: MatchStats, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Each metric for every resample (arrays of length ``resamples``)."""
    cm = np.tensordot(weights, stats.confusion, axes=1)
    return metrics_from_counts(cm, weights @ stats.loss_sum, weights @ stats.count)


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, ignoring undefined resamples."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    tail = (1 - level) / 2 * 100
    return float(np.percentile(samples, tail)), float(np.percentile(samples, 100 - tail))


def summarize(stats: MatchStats, weights: np.ndarray) -> dict[str, dict[str, float]]:
    """Point estimate (all matches once) and 95% bootstrap interval per metric."""
    point = metrics_from_counts(stats.confusion.sum(axis=0), stats.loss_sum.sum(), stats.count.sum())
    boot = bootstrap_metrics(stats, weights)
    return {m: {"value": float(point[m]), "ci_low": interval(boot[m])[0], "ci_high": interval(boot[m])[1]}
            for m in METRICS}


def paired_difference(first: MatchStats, second: MatchStats, weights: np.ndarray) -> dict[str, dict[str, float]]:
    """first minus second, per metric, on identical resampled matches."""
    if first.matches != second.matches:
        raise ValueError("Paired comparison needs the same matches in the same order.")
    point_a = metrics_from_counts(first.confusion.sum(axis=0), first.loss_sum.sum(), first.count.sum())
    point_b = metrics_from_counts(second.confusion.sum(axis=0), second.loss_sum.sum(), second.count.sum())
    boot_a, boot_b = bootstrap_metrics(first, weights), bootstrap_metrics(second, weights)
    out = {}
    for m in METRICS:
        diff = boot_a[m] - boot_b[m]
        low, high = interval(diff)
        out[m] = {"value": float(point_a[m] - point_b[m]), "ci_low": low, "ci_high": high,
                  "share_positive": float(np.mean(diff[np.isfinite(diff)] > 0))}
    return out
