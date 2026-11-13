"""Stage 4C development matches: four heroes, shared deals, seat swaps (master seed 5000).

    python scripts/run_phase4_dev.py --deals 60 --workers 4           # play and report
    python scripts/run_phase4_dev.py --deals 60 --report-only         # re-analyse saved records

Every hero plays one seat-swapped match of ``--deals`` deals against each of
the 16 opponent configurations. Against opponent ``i`` all heroes use the same
match seed ``derive_seed(5000, "dev", i)``, so they receive identical cards.

Outputs:

* data/processed/phase4_dev/records_d<deals>.json.gz (git-ignored): every hand
  and hero decision;
* results/phase4/stage4c_dev_report.md and stage4c_dev_summary.json:
  bb/100, paired differences, all-in statistics, low-coverage reliance,
  latency, errors, reproducibility checks and the sample-size estimate.

Development only: these results may guide fixes before the freeze; they are
not the final evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hybridpoker.agents.ev_agent import FROZEN_MODEL_SHA256
from hybridpoker.dataset_generation import warm_preflop_cache
from hybridpoker.hand_log import _git_commit
from hybridpoker.opponent_configs import build_configs
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH, file_sha256
from hybridpoker.phase4_eval import DEV_MASTER_SEED, HEROES, play_dev_match
from hybridpoker.phase4_report import REPLAY_DEALS, analyse

RAW_DIR = Path("data/processed/phase4_dev")
OUT = Path("results/phase4")


def _init_worker() -> None:
    warm_preflop_cache([500, 200])


def _dirty() -> bool | None:
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           check=True)
        return bool(r.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def play_all(deals: int, workers: int, heroes: tuple[str, ...]) -> dict:
    configs = build_configs()
    jobs = [(h, c, i, deals) for h in heroes for i, c in enumerate(configs)]
    replays = [(h, configs[i], i, REPLAY_DEALS) for h in heroes for i in (0, 10)]
    commit, dirty = _git_commit(), _dirty()
    start = time.perf_counter()
    results, replay_results = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(play_dev_match, *job): "main" for job in jobs}
        futures.update({pool.submit(play_dev_match, *job): "replay" for job in replays})
        for future in as_completed(futures):
            r = future.result()
            (results if futures[future] == "main" else replay_results).append(r)
            if futures[future] == "main":
                status = "ERROR" if r["error"] else f"{len(r['hands'])} hands"
                print(f"  {r['hero']:<13}{r['opponent']:<20}{status:<12}{r['seconds']:7.1f}s "
                      f"({len(results)}/{len(jobs)})", flush=True)
    return {"deals": deals, "master_seed": DEV_MASTER_SEED, "heroes": list(heroes),
            "opponents": [c.to_dict() for c in configs], "wall_seconds": time.perf_counter() - start,
            "workers": workers, "git_commit_at_start": commit, "working_tree_dirty_at_start": dirty,
            "model_sha256": file_sha256(PHASE4_MODEL_PATH), "python": platform.python_version(),
            "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
            "results": results, "replays": replay_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deals", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--heroes", nargs="*", default=list(HEROES))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    raw = RAW_DIR / f"records_d{args.deals}.json.gz"
    if args.report_only:
        data = json.loads(gzip.decompress(raw.read_bytes()))
    else:
        if file_sha256(PHASE4_MODEL_PATH) != FROZEN_MODEL_SHA256:
            print("The model file does not match the frozen SHA-256.", file=sys.stderr)
            return 1
        data = play_all(args.deals, args.workers, tuple(args.heroes))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(gzip.compress(json.dumps(data).encode()))
        print(f"wrote {raw}")
    summary, lines = analyse(data)
    summary.update({k: data[k] for k in ("master_seed", "git_commit_at_start", "working_tree_dirty_at_start",
                                         "model_sha256", "wall_seconds", "workers", "python", "machine")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage4c_dev_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (OUT / "stage4c_dev_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'stage4c_dev_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
