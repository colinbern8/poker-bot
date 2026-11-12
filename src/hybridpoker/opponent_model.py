"""Supervised opponent-action models (Phase 3, Stage 3).

Pipeline for one feature set:

1. :func:`feature_matrix` builds Feature Set A or B from public ``obs_``
   columns (and, for B, earlier-deal actions), with history priors fitted
   on **training rows only**.
2. :class:`FeaturePreprocessor` is fitted on the training matrix only. Tree
   models need no scaling or imputation, so its job is to (a) fix the
   column order, (b) drop columns that are constant in training data
   (they carry no information), and (c) reject missing or infinite
   values instead of silently filling them.
3. :func:`make_model` builds a model with a fixed random seed.
4. :func:`class_probabilities` returns probabilities in the fixed
   ``CLASS_ORDER``, even if a model never saw some class (that class then
   gets probability 0 and is reported, rather than silently dropped).

Targets are integers 0-4 in ``CLASS_ORDER``:
fold, check_call, raise_half_pot, raise_pot, all_in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from hybridpoker.action_probabilities import CLASS_ORDER, legal_mask
from hybridpoker.features import feature_set_a
from hybridpoker.leakage import audit_feature_names
from hybridpoker.opponent_history import HistoryPriors, feature_set_b
from hybridpoker.seeding import derive_seed

FEATURE_SETS = ("A", "B")
MODEL_NAMES = ("dummy", "random_forest", "hist_gradient_boosting")
MASTER_SEED = 3000
N_CLASSES = len(CLASS_ORDER)

RF_GRID: dict[str, list[Any]] = {
    "n_estimators": [300],
    "max_depth": [None, 12, 24],
    "min_samples_leaf": [1, 5, 20],
    "max_features": ["sqrt", 0.5],
    "class_weight": [None, "balanced"],
}
HGB_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [200, 400],
    "max_leaf_nodes": [15, 31, 63],
    "l2_regularization": [0.0, 1.0],
    "class_weight": [None, "balanced"],
}


def encode_targets(actions: pd.Series) -> np.ndarray:
    """Action names -> integers in ``CLASS_ORDER``. Unknown names raise."""
    index = {name: i for i, name in enumerate(CLASS_ORDER)}
    unknown = set(actions) - set(index)
    if unknown:
        raise ValueError(f"Unknown action classes: {sorted(unknown)}")
    return actions.map(index).to_numpy(dtype=int)


def feature_matrix(frame: pd.DataFrame, feature_set: str, priors: HistoryPriors | None) -> pd.DataFrame:
    """Feature Set A or B for every row of ``frame`` (same index)."""
    if feature_set == "A":
        return feature_set_a(frame)
    if feature_set == "B":
        if priors is None:
            raise ValueError("Feature Set B needs history priors fitted on training rows.")
        return feature_set_b(frame, priors)
    raise ValueError(f"Unknown feature set {feature_set!r}; choose from {FEATURE_SETS}.")


@dataclass
class FeaturePreprocessor:
    """Column selection fitted on training data only."""

    columns: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame) -> FeaturePreprocessor:
        self._check_values(train)
        names = audit_feature_names(list(train.columns))
        if not names.passed:
            raise ValueError(f"Forbidden feature names: {names.detail}")
        self.dropped_constant = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
        self.columns = [c for c in train.columns if c not in self.dropped_constant]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            raise RuntimeError("FeaturePreprocessor must be fitted first.")
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        self._check_values(frame)
        return frame.loc[:, self.columns].to_numpy(dtype=float)

    @staticmethod
    def _check_values(frame: pd.DataFrame) -> None:
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Feature matrix contains missing or infinite values.")


def model_seed(model_name: str, feature_set: str) -> int:
    """Fixed random seed for a model, derived from the master seed (fits in 32 bits)."""
    return derive_seed(MASTER_SEED, "model", model_name, feature_set) % (2**32)


def make_model(model_name: str, params: dict[str, Any], seed: int):
    """Build an unfitted model. HistGradientBoosting never uses internal early stopping,
    because that would hold out random rows and break match-level splitting."""
    if model_name == "dummy":
        return DummyClassifier(strategy="prior")
    if model_name == "random_forest":
        # n_jobs=1: with several threads, scikit-learn adds the trees' probabilities
        # in thread-completion order, so results differ in the last digits between
        # runs. One thread makes predictions exactly reproducible.
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params)
    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}.")


def grid(model_name: str) -> list[dict[str, Any]]:
    """All hyperparameter combinations, in a fixed order."""
    if model_name == "dummy":
        return [{}]
    space = {"random_forest": RF_GRID, "hist_gradient_boosting": HGB_GRID}[model_name]
    combos: list[dict[str, Any]] = [{}]
    for key, values in space.items():
        combos = [{**c, key: v} for c in combos for v in values]
    return combos


def class_probabilities(model, x: np.ndarray) -> np.ndarray:
    """Predicted probabilities as an (rows, 5) array in ``CLASS_ORDER``.

    A class the model never saw in training gets probability 0 here (the
    legal mask's floor later keeps log-loss finite). The task itself is
    never changed: all five classes are always scored.
    """
    raw = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    if not np.isin(classes, np.arange(N_CLASSES)).all():
        raise ValueError(f"Model classes {classes} are not a subset of 0..{N_CLASSES - 1}.")
    full = np.zeros((raw.shape[0], N_CLASSES))
    full[:, classes] = raw
    return full


def missing_training_classes(model) -> list[str]:
    """Names of classes the fitted model never saw (should be empty)."""
    seen = set(np.asarray(model.classes_).tolist())
    return [name for i, name in enumerate(CLASS_ORDER) if i not in seen]


def legal_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Public legal-action mask, (rows, 5), in ``CLASS_ORDER``."""
    return legal_mask(frame)
