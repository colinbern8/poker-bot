"""Back up completed Phase 4D match checkpoints to the orphan branch phase4-final-checkpoints.

    python backup_checkpoints.py            # run from this worktree directory

Read-only towards the running evaluation: it only reads finished files in
/home/user/poker-bot/data/processed/phase4_final/ (written atomically by the
runner, so a file that exists is complete) and the frozen manifest. Each file
is validated before it is copied:

* hero and opponent are in the manifest, and the seed equals the manifest's
  match seed for that opponent;
* the deal count is the manifest's (500, or 6 for replays), no error was
  recorded, and every deal has both seatings;
* the stored result hash matches the hands.

Valid files are copied to checkpoints/, listed with their SHA-256 in
checkpoints/INDEX.json, committed and pushed. Invalid files are reported and
not copied. Nothing here changes the running simulation, the frozen commit or
the evaluation code. This branch has no history in common with main and must
never be merged into it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path("/home/user/poker-bot/data/processed/phase4_final")
MANIFEST = Path("/home/user/poker-bot/data/manifests/phase4_eval_manifest.json")
HERE = Path(__file__).resolve().parent
DEST = HERE / "checkpoints"
BRANCH = "phase4-final-checkpoints"
FREEZE_COMMIT = "d50af2b04e00a58a2fe5104ae59da8f2f2ea7669"


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


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    index_path = DEST / "INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"files": {}}
    added, problems = [], []
    for kind in ("jobs", "replays"):
        for path in sorted((SOURCE / kind).glob("*.json.gz")):
            name = f"{kind}/{path.name}"
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if index["files"].get(name, {}).get("sha256") == sha:
                continue  # already backed up, unchanged
            record, problem = validate(path, manifest, kind == "replays")
            if problem:
                problems.append(f"{name}: {problem}")
                continue
            (DEST / kind).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, DEST / name)
            index["files"][name] = {"sha256": sha, "hero": record["hero"], "opponent": record["opponent"],
                                    "deals": record["deals"], "hands": len(record["hands"]),
                                    "result_hash": record["result_hash"], "run_id": record.get("run_id")}
            added.append(name)
    for p in problems:
        print("NOT backed up:", p)
    if not added:
        print("nothing new to back up")
        return 1 if problems else 0
    main_jobs = sum(1 for n in index["files"] if n.startswith("jobs/"))
    replays = sum(1 for n in index["files"] if n.startswith("replays/"))
    index.update({"freeze_commit": FREEZE_COMMIT, "freeze_tag": "phase4-freeze",
                  "manifest_sha256": manifest["manifest_sha256"],
                  "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "complete": {"matches": main_jobs, "of": 64, "replays": replays, "replays_of": 8}})
    index["files"] = dict(sorted(index["files"].items()))
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    git("add", "checkpoints")
    git("commit", "-q", "-m",
        f"Back up Phase 4D checkpoints: {main_jobs}/64 matches, {replays}/8 replays\n\n"
        f"Generated data from the frozen run (tag phase4-freeze, {FREEZE_COMMIT[:7]}); validated against the "
        f"manifest before copying. Backup branch only: never merge into main.\n\n"
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_0125wGhLBRqB1rZxpmKWNPBN")
    for attempt in range(4):
        try:
            git("push", "-q", "-u", "origin", BRANCH)
            print(f"pushed {len(added)} new file(s): {main_jobs}/64 matches, {replays}/8 replays")
            return 0
        except subprocess.CalledProcessError as error:
            print("push failed:", error.stderr.strip())
            time.sleep(2 ** (attempt + 1))
    return 2


if __name__ == "__main__":
    sys.exit(main())
