"""The frozen Hybrid ML Agent for live play: model loading, agent construction, warm-up.

* Files are found from the **repository root**, not the current folder, so
  the demonstration works whichever folder it is started from.
* The model file is accepted only if its SHA-256 equals the frozen hash
  (``FROZEN_MODEL_SHA256``); otherwise ``ModelHashMismatch`` is raised. A
  missing file raises :class:`ModelUnavailable` with restore instructions.
* :func:`build_ml_agent` creates the agent exactly as the Phase 4 evaluation
  did (``make_ml_ev_agent``: name, sample counts, coverage table), but reuses a
  model loaded once. The model is never changed by play, so one loaded copy
  can serve every match; each match gets a **new** agent, and therefore a new,
  empty opponent tracker.
"""

from __future__ import annotations

from pathlib import Path

from hybridpoker.agents.ev_agent import EQUITY_SAMPLES, FROZEN_MODEL_SHA256, EVAgent, MLResponseModel
from hybridpoker.dataset_generation import warm_preflop_cache
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH
from hybridpoker.range_equity import PREFLOP_RANK_SAMPLES
from hybridpoker.state_coverage import COVERAGE_PATH, load_coverage

ML_AGENT_NAME = "ev_ml"
REPO_MARKER = Path("data/manifests/phase4_eval_manifest.json")
RESTORE_HINT = "python scripts/restore_phase4_model.py --from /path/to/random_forest_logloss_B.joblib"


class ModelUnavailable(RuntimeError):
    """The frozen model file is not in place."""


def repo_root() -> Path:
    """The repository folder (found from this file, or else the current folder)."""
    for candidate in (Path(__file__).resolve().parents[2], Path.cwd()):
        if (candidate / REPO_MARKER).is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find the repository root (looked for {REPO_MARKER}).")


def model_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / PHASE4_MODEL_PATH


def load_frozen_ml_model(root: Path | None = None) -> MLResponseModel:
    """Load Random Forest B #22 after checking its SHA-256 against the frozen hash."""
    path = model_path(root)
    if not path.is_file():
        raise ModelUnavailable(f"The frozen model {path} is missing. Restore it with:\n    {RESTORE_HINT}")
    return MLResponseModel.load(path, FROZEN_MODEL_SHA256)


def build_ml_agent(model: MLResponseModel, root: Path | None = None) -> EVAgent:
    """A fresh Hybrid ML Agent, configured exactly like the evaluated ``ev_ml`` hero."""
    coverage = load_coverage((root or repo_root()) / COVERAGE_PATH)
    return EVAgent(ML_AGENT_NAME, model, EQUITY_SAMPLES, coverage=coverage)


def warm_caches() -> float:
    """Fill the preflop equity tables the agent uses; returns the seconds taken.

    Cached values depend only on the hand class and sample count, so warming
    never changes a decision; it only removes a delay from the first hands.
    """
    return warm_preflop_cache([EQUITY_SAMPLES, PREFLOP_RANK_SAMPLES])
