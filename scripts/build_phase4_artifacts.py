"""Build the two small training-derived files the Phase 4 EV agents need.

    python scripts/build_phase4_artifacts.py           # write both files
    python scripts/build_phase4_artifacts.py --check   # fail if the committed files differ

Both come from the Phase 3 **train split only** (23,603 rows), so the agents
need no dataset at run time:

* results/phase3/frequency_prior.json: the global share of each action
  (the Dummy baseline). It is the Frequency EV Agent's opponent model,
  legal-masked at use. It ignores the betting situation entirely.
* results/phase3/state_coverage.json: training rows per coarse situation
  key (see hybridpoker.state_coverage), used only for diagnostics.

Requires the generated Phase 3 dataset (data/raw/simulations/phase3_s3000).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from hybridpoker.action_probabilities import CLASS_ORDER
from hybridpoker.dataset_loader import load_dataset
from hybridpoker.dataset_manifest import DEFAULT_MANIFEST_DIR, MAIN_MANIFEST_NAME, load_manifest, manifest_hash
from hybridpoker.state_coverage import COVERAGE_PATH, LOW_COVERAGE_THRESHOLD, coverage_counts

RAW = Path("data/raw/simulations/phase3_s3000")
PRIOR_PATH = Path("results/phase3/frequency_prior.json")
SOURCE_COMMIT = "a79526f5d40e7c757f7cce1721f8b48ade2cd9d0"


def build() -> dict[Path, str]:
    manifest = load_manifest(DEFAULT_MANIFEST_DIR / MAIN_MANIFEST_NAME)
    frame = load_dataset(manifest, RAW)
    train = frame[frame["id_split"] == "train"]
    counts = train["act_category"].value_counts()
    class_counts = {c: int(counts.get(c, 0)) for c in CLASS_ORDER}
    total = sum(class_counts.values())
    prior = {
        "description": "Global training-split action frequencies (the Dummy prior). Opponent model of the "
                       "Frequency EV Agent, legal-masked at use. Not conditioned on the betting situation.",
        "class_order": list(CLASS_ORDER),
        "counts": class_counts,
        "shares": [class_counts[c] / total for c in CLASS_ORDER],
        "rows": total,
        "split": "train",
        "manifest_hash": manifest_hash(manifest),
        "source_commit": SOURCE_COMMIT,
    }
    coverage = {
        "description": "Training-split rows per coverage key: street | facing_all_in | raises this street "
                       "(capped at 3). Diagnostic only.",
        "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
        "rows": int(len(train)),
        "split": "train",
        "manifest_hash": manifest_hash(manifest),
        "source_commit": SOURCE_COMMIT,
        "counts": coverage_counts(train),
    }
    assert np.isclose(sum(prior["shares"]), 1.0)
    return {PRIOR_PATH: json.dumps(prior, indent=2) + "\n", COVERAGE_PATH: json.dumps(coverage, indent=2) + "\n"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    status = 0
    for path, text in build().items():
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == text
            print(f"{path}: {'identical' if same else 'DIFFERENT OR MISSING'}")
            status |= 0 if same else 1
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
