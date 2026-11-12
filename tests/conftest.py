"""Shared fixtures for the Phase 3 tests.

``tiny_dataset`` plays the four pilot pairings with only 4 deals each and
30 equity samples, so the whole Phase 3 pipeline (generation, loading,
features, audits) can be tested in a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from hybridpoker.dataset_generation import generate_match
from hybridpoker.dataset_loader import load_dataset, prior_rows
from hybridpoker.dataset_manifest import build_pilot_manifest, match_specs
from hybridpoker.opponent_history import HistoryPriors, fit_priors

TINY_DEALS = 4
TINY_SAMPLES = 30


def tiny_manifest() -> dict:
    manifest = build_pilot_manifest(deal_count=TINY_DEALS)
    for config in manifest["configs"]:
        config["equity_samples"] = TINY_SAMPLES
    return manifest


@dataclass
class TinyDataset:
    manifest: dict
    raw_dir: Path
    frame: pd.DataFrame
    priors: HistoryPriors


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> TinyDataset:
    manifest = tiny_manifest()
    raw_dir = tmp_path_factory.mktemp("tiny_pilot")
    for spec in match_specs(manifest):
        generate_match(spec, manifest, raw_dir)
    frame = load_dataset(manifest, raw_dir)
    return TinyDataset(manifest, raw_dir, frame, fit_priors(prior_rows(frame, manifest)))
