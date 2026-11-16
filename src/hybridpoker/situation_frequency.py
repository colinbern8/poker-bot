"""Situation-aware action frequencies: the stronger frequency control of Phase 4.

The original Frequency EV Agent uses one global set of training-split action
shares for every situation (``frequency_prior.json``). This module conditions
those shares on the **public betting situation** the opponent faces:

* the street (preflop, flop, turn, river);
* whether the opponent faces a bet (``to_call > 0``);
* whether that bet put the bettor all-in (``to_call > 0`` and the bettor has
  no chips behind, the same definition as :mod:`hybridpoker.state_coverage`);
* raises so far this street, capped at 3 ("3" means "3 or more").

Only Phase 3 **train-split** rows are counted
(``results/phase3/situation_frequency_prior.json``, built by
``scripts/build_phase4_artifacts.py``). Test, held-out and Phase 4 results
are never used.

**Sparse situations: hierarchical shrinkage (fixed in advance).** Some
situations are rare in training data (a preflop all-in has 11 rows), so raw
frequencies would be noisy or undefined. Each situation is estimated from
its own rows and its *parent* situation, which drops the last condition::

    global -> street -> street|bet -> street|bet|all_in -> street|bet|all_in|raises

    p(node) = (counts(node) + K * p(parent)) / (rows(node) + K),   K = 30

* With no rows, a situation simply inherits its parent's shares (fallback).
* With many rows, it approaches its own observed frequencies.
* With ``K`` rows, own data and parent count equally.
* The global level is the plain training share (the original prior).

``K = 30`` equals the low-coverage threshold already used for diagnostics in
Stage 4B. Both the ordering of the hierarchy and ``K`` were fixed before any
Phase 4 development match was played and were not tuned on any result.

**What this baseline does not use:** the opponent's identity or history, hand
strength, the board texture, pot and bet sizes (other than "a bet exists" and
"it is all-in"), position, and previous streets. The ML model sees many of
these, so the ML-minus-situation comparison still includes their value; it
is a stronger, but not a complete, control.

Probabilities are always legal-masked afterwards (by the response model), so
for example a situation facing an all-in only keeps fold and call.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hybridpoker.action_probabilities import CLASS_ORDER

SITUATION_PRIOR_PATH = Path("results/phase3/situation_frequency_prior.json")
SMOOTHING_ROWS = 30
MAX_RAISES = 3
STREETS = ("preflop", "flop", "turn", "river")
LEVEL_NAMES = ("global", "street", "street|bet", "street|bet|all_in", "street|bet|all_in|raises")
_KEY_COLUMNS = ["obs_street", "obs_to_call", "obs_opponent_stack", "obs_raises_this_street"]


def situation_path(obs_columns: Mapping[str, Any]) -> tuple[str, ...]:
    """Keys from the most general to the most specific level for one ``obs_`` row.

    >>> situation_path({"obs_street": "flop", "obs_to_call": 4, "obs_opponent_stack": 0,
    ...                 "obs_raises_this_street": 5})
    ('global', 'flop', 'flop|bet=1', 'flop|bet=1|all_in=1', 'flop|bet=1|all_in=1|raises=3')
    """
    street = str(obs_columns["obs_street"])
    if street not in STREETS:
        raise ValueError(f"Unknown street {street!r}.")
    facing_bet = int(obs_columns["obs_to_call"] > 0)
    facing_all_in = int(facing_bet == 1 and obs_columns["obs_opponent_stack"] == 0)
    raises = min(int(obs_columns["obs_raises_this_street"]), MAX_RAISES)
    level2 = f"{street}|bet={facing_bet}"
    level3 = f"{level2}|all_in={facing_all_in}"
    return ("global", street, level2, level3, f"{level3}|raises={raises}")


def situation_key(obs_columns: Mapping[str, Any]) -> str:
    """The most specific situation key (the one counted in the artifact)."""
    return situation_path(obs_columns)[-1]


def situation_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Action counts per most-specific situation key, from logged ``obs_``/``act_`` rows."""
    keys = [situation_key(row) for row in frame[_KEY_COLUMNS].to_dict("records")]
    table = pd.crosstab(pd.Series(keys, name="key"), frame["act_category"].to_numpy())
    return {key: {c: int(table.loc[key, c]) if c in table.columns else 0 for c in CLASS_ORDER}
            for key in sorted(table.index)}


def _parent(key: str) -> str:
    return "global" if "|" not in key else key.rsplit("|", 1)[0]


class SituationFrequencies:
    """Smoothed action shares for every situation key (see the module text)."""

    def __init__(self, leaf_counts: Mapping[str, Mapping[str, int]], smoothing_rows: float = SMOOTHING_ROWS) -> None:
        if smoothing_rows <= 0:
            raise ValueError("smoothing_rows must be positive.")
        self.smoothing_rows = float(smoothing_rows)
        counts: dict[str, np.ndarray] = {}
        for key, row in leaf_counts.items():
            if key.count("|") != 3:
                raise ValueError(f"{key!r} is not a most-specific situation key.")
            vector = np.array([row.get(c, 0) for c in CLASS_ORDER], dtype=float)
            if (vector < 0).any():
                raise ValueError("Counts must be non-negative.")
            node = key
            while True:  # add the leaf's counts to itself and every ancestor
                counts[node] = counts.get(node, np.zeros(len(CLASS_ORDER))) + vector
                if node == "global":
                    break
                node = _parent(node)
        if "global" not in counts or counts["global"].sum() <= 0:
            raise ValueError("No training rows.")
        self.counts = counts
        self._cache: dict[str, np.ndarray] = {}

    def rows(self, key: str) -> int:
        return int(self.counts.get(key, np.zeros(1)).sum())

    def shares(self, key: str) -> np.ndarray:
        """Smoothed shares in ``CLASS_ORDER`` for any key in the hierarchy (unseen keys fall back)."""
        if key not in self._cache:
            if key == "global":
                total = self.counts["global"]
                value = total / total.sum()
            else:
                own = self.counts.get(key, np.zeros(len(CLASS_ORDER)))
                value = (own + self.smoothing_rows * self.shares(_parent(key))) / (own.sum() + self.smoothing_rows)
            self._cache[key] = value
        return self._cache[key].copy()

    def predict(self, obs_columns: Mapping[str, Any]) -> np.ndarray:
        """Unmasked shares for one opponent situation."""
        return self.shares(situation_key(obs_columns))

    @classmethod
    def load(cls, path: Path = SITUATION_PRIOR_PATH) -> SituationFrequencies:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["class_order"] != list(CLASS_ORDER):
            raise ValueError("Situation prior uses a different class order.")
        if data["max_raises"] != MAX_RAISES:
            raise ValueError("Situation prior uses a different raise cap.")
        return cls(data["counts"], data["smoothing_rows"])
