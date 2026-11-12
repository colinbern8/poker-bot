"""Fixed class order and legal-action masking for predicted action probabilities.

Every model in Phase 3 predicts probabilities for the five abstract actions
in one fixed order (``CLASS_ORDER``). Before the probabilities are used or
scored, :func:`mask_and_renormalize`:

1. sets illegal actions (from the public ``obs_legal_*`` flags) to 0;
2. renormalizes the legal probabilities to sum to 1 (if a model gave all
   legal actions 0, they become equally likely);
3. raises every legal probability to at least ``floor`` (so log-loss stays
   finite when a model gave a legal action probability 0) and
   renormalizes again, so each row still sums to exactly 1.

The raw model probabilities are kept unchanged alongside the masked ones
for diagnostics (e.g. how much probability a model wasted on illegal
actions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hybridpoker.actions import ActionCategory

CLASS_ORDER: tuple[str, ...] = tuple(c.value for c in ActionCategory)
PROBABILITY_FLOOR = 1e-6
LEGAL_COLUMNS = tuple(f"obs_legal_{c}" for c in CLASS_ORDER)


@dataclass(frozen=True)
class MaskedProbabilities:
    """Raw and masked probabilities, both of shape (rows, 5) in ``CLASS_ORDER``."""

    raw: np.ndarray
    masked: np.ndarray
    legal: np.ndarray

    @property
    def illegal_mass(self) -> np.ndarray:
        """Raw probability each row put on illegal actions (a diagnostic)."""
        return np.where(self.legal, 0.0, self.raw).sum(axis=1)

    @property
    def predicted(self) -> np.ndarray:
        """The most probable legal action of each row, as class names."""
        return np.asarray(CLASS_ORDER)[self.masked.argmax(axis=1)]


def legal_mask(frame: pd.DataFrame) -> np.ndarray:
    """Boolean (rows, 5) legality matrix from the public ``obs_legal_*`` columns."""
    return frame.loc[:, list(LEGAL_COLUMNS)].astype(bool).to_numpy()


def mask_and_renormalize(
    raw: np.ndarray, legal: np.ndarray, floor: float = PROBABILITY_FLOOR
) -> MaskedProbabilities:
    """Zero illegal actions, apply a floor to legal ones, and renormalize."""
    raw = np.array(raw, dtype=float, copy=True)
    legal = np.asarray(legal, dtype=bool)
    if raw.ndim != 2 or raw.shape[1] != len(CLASS_ORDER):
        raise ValueError(f"Expected probabilities of shape (rows, {len(CLASS_ORDER)}), got {raw.shape}.")
    if legal.shape != raw.shape:
        raise ValueError(f"Legal mask shape {legal.shape} differs from probabilities {raw.shape}.")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Probabilities must be finite and non-negative.")
    if not legal.any(axis=1).all():
        raise ValueError("Every row needs at least one legal action.")
    if not 0.0 <= floor < 1.0 / len(CLASS_ORDER):
        raise ValueError("floor must be in [0, 1/5).")

    masked = np.where(legal, raw, 0.0)
    totals = masked.sum(axis=1, keepdims=True)
    uniform = legal / legal.sum(axis=1, keepdims=True)
    masked = np.where(totals > 0, masked / np.where(totals > 0, totals, 1.0), uniform)

    masked = np.where(legal, np.maximum(masked, floor), 0.0)
    masked = masked / masked.sum(axis=1, keepdims=True)
    raw.setflags(write=False)
    return MaskedProbabilities(raw=raw, masked=masked, legal=legal)
