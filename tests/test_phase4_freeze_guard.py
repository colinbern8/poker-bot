"""Guard: later phases leave the frozen Phase 4 code and its evaluation evidence untouched.

1. **Code, artifacts and pins (reused check).** The frozen
   ``phase4_final.check_manifest`` is run against the *committed* manifest.
   It verifies the manifest's own SHA-256, its seeds and opponents, and the
   SHA-256 of all 46 frozen files (package code, runner, priors, coverage
   table, ``requirements.txt``, ``pyproject.toml``). It also reports any new
   module under ``src/hybridpoker/``, which is why Phase 5 code lives in
   ``src/hybridpoker_demo/``. The model file's hash is pinned in the same
   manifest (checked here as a value; the file itself is outside git).
2. **Evaluation results (new, only what nothing else covers).** No existing
   check pins ``results/phase4/``, so their SHA-256 values (CRLF normalized to
   LF, as in the frozen check) are listed below. The final report, summary and
   run log are byte-identical to the copies on ``phase4-final-checkpoints``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from hybridpoker.agents.ev_agent import FROZEN_MODEL_SHA256
from hybridpoker.phase4_final import MANIFEST_PATH, check_manifest, load_manifest

RESULTS_DIR = Path("results/phase4")
RESULT_SHA256 = {
    "stage4b_examples.md": "1fce1553d2dd768b0854ab56043b87c8e00ebb44e84713c2771506fc062cfef0",
    "stage4b_rule_audit.json": "0af1734d53959b21f4fbd2a77036ef6c12e15c791f629d9963ecd3c7d13029d2",
    "stage4b_rule_audit.md": "2eadcdf5e4e59d303b531e99d91e613b43a5c0e2d0169c9da784cc8c6356095b",
    "stage4b_rule_audit_rows.csv": "a83b33101835190a99aa712ad020a5bec83364cfd4740921211ae797e7f7fc51",
    "stage4c_dev_dependence_sensitivity.json": "653fa70409636e8b416598439f274ff0ffa8f935c1fc6ea6bbcc54e274300677",
    "stage4c_dev_dependence_sensitivity.md": "6e39da60fa3f80e1f5f9cebb759188a858100eb9ad81e87022aecc8cafdafbb3",
    "stage4c_dev_report.md": "d0c1077c8efb7b3a1c4988f5da77bc69a1f594b8c21103b23e715aa6ef0875a6",
    "stage4c_dev_summary.json": "b011226e21a895fc9c03e0704c9dfdc293f2612da3778baee3796266a3f8b739",
    "stage4d_checkpoint_index.json": "dd49e522e2246f8d40a99b372afc8c1f80694ababd1d75be9682424515e33335",
    "stage4d_dependence_sensitivity.json": "31648b54a7c34bc44e37fe1d8e882a9d5499f030c3116d12505a9612a36d465e",
    "stage4d_dependence_sensitivity.md": "8a707cb6cda07b515852b389acdf96a3ac4edd2dd2b163a030326bd7ab2d50b9",
    "stage4d_final_report.md": "04cfbc8755b718c011c5069dbeb3319771bae663e7b3e0f8a3ea5c8ee6b60dc6",
    "stage4d_final_summary.json": "abd2c8b0ed8d7c59210473d0e5d68c16a648788936b423868bb9ab9aa9673864",
    "stage4d_run_log.json": "0dae4a7262b7b04ed80a6fbe79376201539339fb0d73e2c1408233c22442ea2a",
    "stage4d_verification.md": "179d00e7096f76917179ce50c73efd2608c98b363cbc2329a37751180e5f1b10",
}


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_committed_phase4_manifest_and_all_frozen_files_are_unchanged() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    assert len(manifest["frozen_files"]) == 46
    assert check_manifest(manifest) == []
    assert manifest["model"]["sha256"] == FROZEN_MODEL_SHA256
    assert manifest["freeze_tag"] == "phase4-freeze"


def test_phase4_evaluation_results_are_unchanged() -> None:
    present = sorted(p.name for p in RESULTS_DIR.iterdir())
    assert present == sorted(RESULT_SHA256)
    changed = [name for name, digest in RESULT_SHA256.items() if normalized_sha256(RESULTS_DIR / name) != digest]
    assert changed == []
