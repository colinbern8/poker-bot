"""Loading the frozen model and building the live agent (CI-safe: no real model needed).

The real model is exercised only by the opt-in acceptance test
(``tests/acceptance/test_real_model_demo.py``).
"""

from __future__ import annotations

import joblib
import pytest

from hybridpoker.agents.ev_agent import FROZEN_MODEL_SHA256, MLResponseModel
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH, ModelHashMismatch, OpponentPredictor
from hybridpoker.phase4_final import MANIFEST_PATH, load_manifest
from hybridpoker_demo.resources import (
    REPO_MARKER,
    ModelUnavailable,
    build_ml_agent,
    load_frozen_ml_model,
    repo_root,
)


def test_repo_root_is_found_from_any_working_folder(tmp_path, monkeypatch) -> None:
    root = repo_root()
    monkeypatch.chdir(tmp_path)
    assert repo_root() == root and (root / REPO_MARKER).is_file()


def test_missing_model_gives_restore_instructions(tmp_path) -> None:
    with pytest.raises(ModelUnavailable, match="restore_phase4_model.py"):
        load_frozen_ml_model(tmp_path)


def test_a_model_file_with_the_wrong_hash_is_refused(tmp_path, tiny_bundle) -> None:
    path = tmp_path / PHASE4_MODEL_PATH
    path.parent.mkdir(parents=True)
    joblib.dump(tiny_bundle, path)
    with pytest.raises(ModelHashMismatch):
        load_frozen_ml_model(tmp_path)


def test_live_agent_is_configured_exactly_like_the_evaluated_ml_hero(tiny_bundle, tmp_path, monkeypatch) -> None:
    """Same name, sample counts, tie-break and decision rule as ``hero_configs['ev_ml']`` of the manifest.

    A stand-in model labelled with the frozen hash is used, so only the
    construction is tested here; the acceptance test repeats this with the
    real file.
    """
    expected = load_manifest(MANIFEST_PATH)["hero_configs"]["ev_ml"]
    model = MLResponseModel(OpponentPredictor(tiny_bundle), model_sha256=FROZEN_MODEL_SHA256)
    monkeypatch.chdir(tmp_path)  # absolute paths: works from any folder
    agent = build_ml_agent(model)
    assert agent.config() == expected
    assert agent.coverage_counts and agent.tracker is not None and agent.tracker.observed_rows() == 0
    assert build_ml_agent(model).tracker is not agent.tracker  # a fresh tracker for every match
