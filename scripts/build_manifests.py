"""Build the Phase 3 dataset manifests from master seed 3000.

    python scripts/build_manifests.py           # write both manifests
    python scripts/build_manifests.py --check   # fail if the committed files differ

The manifests are small, deterministic JSON files committed to git under
data/manifests/. They define every configuration, match, seed and split.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from hybridpoker.dataset_manifest import (
    DEFAULT_MANIFEST_DIR,
    MAIN_MANIFEST_NAME,
    PILOT_MANIFEST_NAME,
    build_manifest,
    build_pilot_manifest,
    manifest_text,
    write_manifest,
)


def summarize(manifest: dict) -> None:
    print(f"\n{manifest['name']}: {len(manifest['matches'])} matches, "
          f"{manifest['deal_count']} deals each, split attempt {manifest['split_attempt']}")
    print("  splits:", dict(Counter(m["split"] for m in manifest["matches"])))
    for config in manifest["configs"]:
        print(f"  {config['name']:<20} pool={config['pool']:<8} looseness={config['looseness']:.3f} "
              f"aggression={config['aggression']:.3f} bluff_rate={config['bluff_rate']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only compare with the committed files")
    args = parser.parse_args()

    status = 0
    for name, manifest in ((MAIN_MANIFEST_NAME, build_manifest()), (PILOT_MANIFEST_NAME, build_pilot_manifest())):
        path = DEFAULT_MANIFEST_DIR / name
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == manifest_text(manifest)
            print(f"{path}: {'identical' if same else 'DIFFERENT OR MISSING'}")
            status |= 0 if same else 1
        else:
            write_manifest(manifest, path)
            print(f"wrote {path}")
            summarize(manifest)
    return status


if __name__ == "__main__":
    sys.exit(main())
