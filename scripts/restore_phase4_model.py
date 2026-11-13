"""Restore and verify the frozen Phase 4 opponent model (Random Forest B #22).

The model file is git-ignored (CLAUDE.md: no model checkpoints in git), so a
fresh Codespace or cloud container does not have it. Options, fastest first:

    # 1. Copy a saved file into place, then verify its SHA-256 (seconds)
    python scripts/restore_phase4_model.py --from /path/to/random_forest_logloss_B.joblib

    # 2. Just verify the file already at models/phase3/
    python scripts/restore_phase4_model.py

    # 3. Rebuild from scratch when no copy exists (about 20 min on 4 cores):
    #    the dataset is deterministic, and the refit reproduces the same bytes
    python scripts/restore_phase4_model.py --rebuild

The model is accepted only if its SHA-256 equals the approved hash
(``FROZEN_MODEL_SHA256``). ``--rebuild`` regenerates the Phase 3 dataset from
its committed manifest, runs ``select_probability_model.py``, restores the
committed selection JSON (whose provenance fields would otherwise change),
and checks the hash. Tested with Python 3.12.3 and the pinned
requirements.txt; other library versions may produce different bytes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from hybridpoker.agents.ev_agent import FROZEN_MODEL_SHA256
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH, file_sha256

RAW = Path("data/raw/simulations/phase3_s3000")
MANIFEST = Path("data/manifests/phase3_manifest.json")
SELECTION = "results/phase3/probability_model_selection.json"


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="source", type=Path, help="a saved copy of the model file")
    group.add_argument("--rebuild", action="store_true", help="regenerate the dataset and refit (slow)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.source is not None:
        if file_sha256(args.source) != FROZEN_MODEL_SHA256:
            print(f"{args.source} does NOT match the frozen SHA-256; not copied.", file=sys.stderr)
            return 1
        PHASE4_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source, PHASE4_MODEL_PATH)
    elif args.rebuild:
        run("scripts/generate_dataset.py", "--manifest", str(MANIFEST), "--out", str(RAW),
            "--workers", str(args.workers))
        run("scripts/build_phase4_artifacts.py", "--check")
        PHASE4_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        run("scripts/select_probability_model.py")
        subprocess.run(["git", "checkout", "--", SELECTION], check=True)

    if not PHASE4_MODEL_PATH.exists():
        print(f"{PHASE4_MODEL_PATH} is missing. Use --from <copy> or --rebuild.", file=sys.stderr)
        return 1
    digest = file_sha256(PHASE4_MODEL_PATH)
    if digest != FROZEN_MODEL_SHA256:
        print(f"{PHASE4_MODEL_PATH}: SHA-256 {digest} does NOT match the frozen {FROZEN_MODEL_SHA256}.",
              file=sys.stderr)
        return 1
    print(f"{PHASE4_MODEL_PATH}: SHA-256 matches the frozen model ({digest}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
