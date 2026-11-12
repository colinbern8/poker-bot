"""Generate match logs for a Phase 3 dataset manifest, in parallel.

    python scripts/generate_dataset.py --manifest data/manifests/phase3_pilot_manifest.json \
        --out data/raw/simulations/phase3_pilot_s3000 --workers 4

Each match is written to <out>/match_NN/ (ignored by git). Matches that
already have a complete run (metadata.json present) are skipped unless
--overwrite is given. A generation_summary.json with timings, software
versions and the git commit is written to <out>/ at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

from hybridpoker import __version__
from hybridpoker.dataset_generation import generate_match, warm_preflop_cache
from hybridpoker.dataset_manifest import load_manifest, manifest_hash, match_specs, run_dir_name
from hybridpoker.hand_log import _git_commit

_WARMUP_SECONDS = 0.0


def _init_worker(sample_counts: list[int]) -> None:
    global _WARMUP_SECONDS
    _WARMUP_SECONDS = warm_preflop_cache(sample_counts)


def _run(spec, manifest, out_root):
    result = generate_match(spec, manifest, out_root)
    result["worker_pid"] = os.getpid()
    result["worker_warmup_seconds"] = _WARMUP_SECONDS
    return result


def _working_tree_dirty() -> bool | None:
    """True if tracked files have uncommitted changes (None if git is unavailable)."""
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--matches", type=int, nargs="*", help="only these match numbers")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    specs = [s for s in match_specs(manifest) if args.matches is None or s.number in args.matches]
    todo = [s for s in specs if args.overwrite or not (args.out / run_dir_name(s) / "metadata.json").exists()]
    print(f"{manifest['name']}: {len(specs)} selected, {len(todo)} to generate, {args.workers} workers", flush=True)

    # Record provenance at the START: each match's metadata.json records the
    # commit at the moment it is written, which can differ if a commit is made
    # while generation runs.
    commit_at_start = _git_commit()
    dirty_at_start = _working_tree_dirty()
    samples = sorted({c["equity_samples"] for c in manifest["configs"]})
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(samples,)) as pool:
        futures = [pool.submit(_run, spec, manifest, args.out) for spec in todo]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(f"  match {r['number']:02d}: {r['hands']} hands, {r['decisions']} decisions, "
                  f"{r['seconds']:.1f}s ({r['decisions'] / r['seconds']:.1f} dec/s)", flush=True)
    wall = time.perf_counter() - start

    summary = {
        "manifest_name": manifest["name"],
        "manifest_hash": manifest_hash(manifest),
        "workers": args.workers,
        "wall_seconds": wall,
        "matches": sorted(results, key=lambda r: r["number"]),
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "versions": {
            "hybridpoker": __version__,
            "pokerkit": version("pokerkit"),
            "python": platform.python_version(),
            "git_commit_at_start": commit_at_start,
            "working_tree_dirty_at_start": dirty_at_start,
            "git_commit_at_end": _git_commit(),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    temporary = args.out / "generation_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.out / "generation_summary.json")
    print(f"done in {wall:.1f}s wall time", flush=True)


if __name__ == "__main__":
    main()
