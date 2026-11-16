"""Write (or check) the Phase 4 final-evaluation manifest. Plays no hands.

    python scripts/build_phase4_manifest.py           # write data/manifests/phase4_eval_manifest.json
    python scripts/build_phase4_manifest.py --check   # fail if the committed manifest differs

The manifest pins master seed 4000 and every match seed, 500 deals per match,
the four heroes' exact configurations, the 16 opponents, the model hash, the
hash of every source file and training artifact, the library versions and
the analysis plan (see hybridpoker.phase4_final and docs/phase4_analysis_plan.md).
Needs the frozen model file (for the ML hero's configuration and hash check).
"""

from __future__ import annotations

import argparse
import sys

from hybridpoker.phase4_eval import HEROES, make_hero
from hybridpoker.phase4_final import MANIFEST_PATH, build_manifest, check_model, manifest_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest({name: make_hero(name).config() for name in HEROES})
    problems = check_model(manifest)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    text = manifest_text(manifest)
    if args.check:
        same = MANIFEST_PATH.exists() and MANIFEST_PATH.read_text(encoding="utf-8") == text
        print(f"{MANIFEST_PATH}: {'identical' if same else 'DIFFERENT OR MISSING'}")
        return 0 if same else 1
    MANIFEST_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST_PATH} (manifest_sha256 {manifest['manifest_sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
