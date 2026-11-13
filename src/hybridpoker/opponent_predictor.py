"""Predict an opponent's next action with the frozen Phase 3 model (Feature Set B).

Input: one or more opponent observations (usually hypothetical ones from
:mod:`hybridpoker.counterfactual`) and the opponent's history features from
:class:`hybridpoker.opponent_tracker.OpponentTracker`. The observations go
through the same ``observation_columns`` -> ``feature_set_a`` path as the
training log, the frozen preprocessor fixes the column order, and the
output is **always legal-masked and renormalized** in ``CLASS_ORDER``
(raw probabilities are kept for diagnostics).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hybridpoker.action_probabilities import CLASS_ORDER, MaskedProbabilities, legal_mask, mask_and_renormalize
from hybridpoker.features import feature_set_a
from hybridpoker.hand_log import observation_columns
from hybridpoker.observation import Observation
from hybridpoker.opponent_history import B_FEATURES
from hybridpoker.opponent_model import class_probabilities

PHASE4_MODEL_PATH = Path("models/phase3/random_forest_logloss_B.joblib")


class ModelHashMismatch(RuntimeError):
    """The model file is not the frozen one."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class OpponentPredictor:
    """Legal-masked action probabilities from a Feature Set B model bundle."""

    def __init__(self, bundle: dict) -> None:
        if tuple(bundle["class_order"]) != CLASS_ORDER:
            raise ValueError("Model bundle uses a different class order.")
        if bundle["feature_set"] != "B":
            raise ValueError("The Phase 4 predictor needs a Feature Set B model.")
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.priors = bundle["priors"]

    @classmethod
    def load(cls, path: Path = PHASE4_MODEL_PATH, expected_sha256: str | None = None) -> OpponentPredictor:
        if expected_sha256 is not None and file_sha256(Path(path)) != expected_sha256:
            raise ModelHashMismatch(f"{path} does not match the frozen SHA-256 hash.")
        return cls(joblib.load(path))

    def features(self, observations: list[Observation], history: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(feature matrix, public obs_ columns) for the given opponent observations."""
        obs_frame = pd.DataFrame([observation_columns(o) for o in observations])
        features = feature_set_a(obs_frame)
        for name in B_FEATURES:
            features[name] = float(history[name])
        return features, obs_frame

    def predict(self, observations: list[Observation], history: pd.Series) -> MaskedProbabilities:
        if not observations:
            raise ValueError("Nothing to predict.")
        features, obs_frame = self.features(observations, history)
        raw = class_probabilities(self.model, self.preprocessor.transform(features))
        return mask_and_renormalize(raw, legal_mask(obs_frame))

    def predict_masked(self, observations: list[Observation], history: pd.Series) -> np.ndarray:
        return self.predict(observations, history).masked
