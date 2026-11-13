"""Back up, restore and validate Phase 4D match checkpoints (branch phase4-final-checkpoints).

Run from this backup worktree (for example /workspaces/phase4-checkpoints):

    python backup_checkpoints.py --repo /workspaces/poker-bot              # one backup now
    python backup_checkpoints.py --repo /workspaces/poker-bot --every 15   # back up every 15 min until done
    python backup_checkpoints.py --repo /workspaces/poker-bot --restore    # copy backed-up matches into the repo
    python backup_checkpoints.py --repo /workspaces/poker-bot --validate   # check the repo's matches, copy nothing

It never changes the running evaluation, the frozen commit or the evaluation
code. It only reads finished files in <repo>/data/processed/phase4_final/
(the runner writes each one atomically, so a file that exists is complete)
and the frozen manifest. Every match file is validated before it is copied
either way:

* hero and opponent are in the manifest, and the seed equals the manifest's
  match seed for that opponent;
* the deal count is the manifest's (500, or 6 for replays), no error was
  recorded, and every deal has both seatings;
* the stored result hash matches the hands.

Backed-up files are listed with their SHA-256 in checkpoints/INDEX.json. When
the run has finished, its report files (results/phase4/stage4d_final_*) and
run log are backed up too. This branch has no history in common with main
and must never be merged into it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "checkpoints"
BRANCH = "phase4-final-checkpoints"
FREEZE_COMMIT = "d50af2b04e00a58a2fe5104ae59da8f2f2ea7669"
TOTAL_MATCHES, TOTAL_REPLAYS = 64, 8
REPORT_FILES = ("results/phase4/stage4d_final_report.md", "results/phase4/stage4d_final_summary.json",
                "data/processed/phase4_final/run_log.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path, manifest: dict, replay: bool) -> tuple[dict | None, str | None]:
    try:
        record = json.loads(gzip.decompress(path.read_bytes()))
    except Exception as error:  # unreadable or incomplete
        return None, f"unreadable: {error}"
    seeds = {o["name"]: o["match_seed"] for o in manifest["opponents"]}
    deals = manifest["replays"]["deals"] if replay else manifest["deals_per_match"]
    if record.get("hero") not in manifest["heroes"]:
        return None, "unknown hero"
    if record.get("opponent") not in seeds or record.get("seed") != seeds[record["opponent"]]:
        return None, "opponent or seed differs from the manifest"
    if record.get("deals") != deals:
        return None, f"deal count {record.get('deals')} != {deals}"
    if record.get("error"):
        return None, "the match recorded an error"
    pairs = [(h["deal_index"], h["seat_swap"]) for h in record["hands"]]
    if pairs != [(i, s) for i in range(deals) for s in (0, 1)]:
        return None, "hands incomplete or out of order"
    digest = hashlib.sha256(repr([(h["deal_index"], h["seat_swap"], h["hero_net"])
                                  for h in record["hands"]]).encode()).hexdigest()
    if digest != record.get("result_hash"):
        return None, "result hash mismatch"
    return record, None


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True, check=True).stdout


def load_index() -> dict:
    path = DEST / "INDEX.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": {}}


def counts(names) -> tuple[int, int]:
    names = list(names)
    return sum(n.startswith("jobs/") for n in names), sum(n.startswith("replays/") for n in names)


def backup_once(repo: Path, manifest: dict) -> tuple[int, int, int]:
    """Copy new valid matches (and finished report files); commit and push. Returns (new, matches, replays)."""
    source = repo / "data/processed/phase4_final"
    index = load_index()
    added, problems = [], []
    for kind in ("jobs", "replays"):
        for path in sorted((source / kind).glob("*.json.gz")):
            name = f"{kind}/{path.name}"
            digest = sha256(path)
            if index["files"].get(name, {}).get("sha256") == digest:
                continue
            record, problem = validate(path, manifest, kind == "replays")
            if problem:
                problems.append(f"{name}: {problem}")
                continue
            (DEST / kind).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, DEST / name)
            index["files"][name] = {"sha256": digest, "hero": record["hero"], "opponent": record["opponent"],
                                    "deals": record["deals"], "hands": len(record["hands"]),
                                    "result_hash": record["result_hash"], "run_id": record.get("run_id")}
            added.append(name)
    reports = []
    for rel in REPORT_FILES:
        path = repo / rel
        target = DEST / "report" / Path(rel).name
        if path.exists() and (not target.exists() or sha256(target) != sha256(path)):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            reports.append(rel)
    for p in problems:
        print("NOT backed up:", p, flush=True)
    matches, replays = counts(index["files"])
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not added and not reports:
        print(f"{stamp} nothing new: {matches}/{TOTAL_MATCHES} matches, {replays}/{TOTAL_REPLAYS} replays backed up",
              flush=True)
        return 0, matches, replays
    index.update({"freeze_commit": FREEZE_COMMIT, "freeze_tag": "phase4-freeze",
                  "manifest_sha256": manifest["manifest_sha256"], "updated": stamp,
                  "complete": {"matches": matches, "of": TOTAL_MATCHES, "replays": replays, "replays_of": TOTAL_REPLAYS}})
    index["files"] = dict(sorted(index["files"].items()))
    (DEST / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    git("add", "checkpoints")
    git("commit", "-q", "-m",
        f"Back up Phase 4D checkpoints: {matches}/{TOTAL_MATCHES} matches, {replays}/{TOTAL_REPLAYS} replays"
        + (" + final report" if reports else "") + "\n\n"
        f"Generated data from the frozen run (tag phase4-freeze, {FREEZE_COMMIT[:7]}); validated against the "
        "manifest before copying. Backup branch only: never merge into main.")
    for attempt in range(4):
        try:
            git("push", "-q", "-u", "origin", BRANCH)
            print(f"{stamp} pushed {len(added)} new match file(s){' and the report' if reports else ''}: "
                  f"{matches}/{TOTAL_MATCHES} matches, {replays}/{TOTAL_REPLAYS} replays backed up", flush=True)
            break
        except subprocess.CalledProcessError as error:
            print("push failed, retrying:", error.stderr.strip(), flush=True)
            time.sleep(2 ** (attempt + 1))
    else:
        print("PUSH FAILED: the commit is saved locally and will be pushed with the next backup.", flush=True)
    return len(added), matches, replays


def restore(repo: Path, manifest: dict) -> int:
    """Copy backed-up matches into the repo after checking SHA-256 and the manifest."""
    index = load_index()
    target_root = repo / "data/processed/phase4_final"
    bad = 0
    for name, meta in index["files"].items():
        path = DEST / name
        if sha256(path) != meta["sha256"]:
            print(f"SHA-256 MISMATCH, not restored: {name}")
            bad += 1
            continue
        record, problem = validate(path, manifest, name.startswith("replays/"))
        if problem:
            print(f"INVALID, not restored: {name}: {problem}")
            bad += 1
            continue
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == meta["sha256"]:
            print(f"already present: {name}")
        else:
            shutil.copyfile(path, target)
            print(f"restored and verified: {name} ({record['hero']} vs {record['opponent']}, "
                  f"{len(record['hands'])} hands)")
    return 1 if bad else 0


def validate_repo(repo: Path, manifest: dict) -> int:
    source = repo / "data/processed/phase4_final"
    bad = 0
    files = [(k, p) for k in ("jobs", "replays") for p in sorted((source / k).glob("*.json.gz"))]
    for kind, path in files:
        record, problem = validate(path, manifest, kind == "replays")
        print(f"{'OK     ' if not problem else 'INVALID'} {kind}/{path.name}" + (f": {problem}" if problem else ""))
        bad += bool(problem)
    jobs = sum(k == "jobs" for k, _ in files)
    replays = sum(k == "replays" for k, _ in files)
    print(f"{jobs}/{TOTAL_MATCHES} matches and {replays}/{TOTAL_REPLAYS} replays on disk, {bad} invalid")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path("/workspaces/poker-bot"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--every", type=float, metavar="MINUTES", help="back up repeatedly until the run is complete")
    mode.add_argument("--restore", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.repo / "data/manifests/phase4_eval_manifest.json").read_text(encoding="utf-8"))
    if args.restore:
        return restore(args.repo, manifest)
    if args.validate:
        return validate_repo(args.repo, manifest)
    if args.every is None:
        backup_once(args.repo, manifest)
        return 0
    while True:
        backup_once(args.repo, manifest)
        matches, replays = counts(load_index()["files"])
        report_done = (DEST / "report" / "stage4d_final_report.md").exists()
        if matches == TOTAL_MATCHES and replays == TOTAL_REPLAYS and report_done:
            print("All matches, replays and the final report are backed up. Stopping.", flush=True)
            return 0
        time.sleep(args.every * 60)


if __name__ == "__main__":
    sys.exit(main())
