"""How well the training data covers a game situation (diagnostic only).

The opponent model is queried on *hypothetical* states created by the EV
agents' candidate bets, and some of those situations are rare in training
data (e.g. facing a preflop all-in). To make that visible, every situation
gets a coarse **coverage key**:

    street | whether the actor faces an all-in | raises so far this street (0, 1, 2, 3+)

``results/phase3/state_coverage.json`` counts training-split rows per key
(built by ``scripts/build_phase4_artifacts.py``). The agents only *record*
these counts; coverage never changes a decision.

The key is computed from the public ``obs_`` columns, the same columns for
logged training rows and for live observations (``observation_columns``),
so the table and the live lookup cannot disagree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

COVERAGE_PATH = Path("results/phase3/state_coverage.json")
LOW_COVERAGE_THRESHOLD = 30
MAX_RAISES = 3


def coverage_key(obs_columns: Mapping[str, Any]) -> str:
    """Coverage key from one row of public ``obs_`` columns."""
    facing_all_in = int(obs_columns["obs_to_call"] > 0 and obs_columns["obs_opponent_stack"] == 0)
    raises = min(int(obs_columns["obs_raises_this_street"]), MAX_RAISES)
    return f"{obs_columns['obs_street']}|facing_all_in={facing_all_in}|raises={raises}"


def coverage_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Rows per coverage key (sorted keys, for a stable JSON file)."""
    keys = [coverage_key(row) for row in frame[["obs_street", "obs_to_call", "obs_opponent_stack",
                                                 "obs_raises_this_street"]].to_dict("records")]
    counts = pd.Series(keys).value_counts()
    return {key: int(counts[key]) for key in sorted(counts.index)}


def load_coverage(path: Path = COVERAGE_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
