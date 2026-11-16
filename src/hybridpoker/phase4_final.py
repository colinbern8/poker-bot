"""Stage 4D: the frozen final evaluation (manifest, verification, resumable runs, primary analysis).

Everything that defines the final evaluation is written **before** any final
hand is played into ``data/manifests/phase4_eval_manifest.json``:

* master seed 4000 and every match seed (``derive_seed(4000, "eval", opponent, 0)``);
* 500 seat-swapped deals per match; the 4 heroes and their exact configurations;
* the 16 opponent configurations;
* the model's SHA-256 and the SHA-256 of every source file and training
  artifact the agents use (line endings normalized, so a Windows checkout
  hashes the same);
* the pinned library versions;
* the analysis plan (primary endpoint, interval, secondary comparisons,
  subgroups), which ``docs/phase4_analysis_plan.md`` explains in words.

A manifest cannot contain the hash of the commit that contains it, so the
code version is pinned twice: by the file hashes in the manifest, and by the
git tag ``phase4-freeze``, which must point at the checked-out commit before
the runner plays a single final hand (:func:`verify_for_run`).

Runs are **resumable**: each (hero, opponent) match is written to its own
file as soon as it finishes. Matches are fully seeded and independent, so a
restarted run produces exactly the same records.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable, Iterable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from hybridpoker.agents.ev_agent import FROZEN_MODEL_SHA256
from hybridpoker.opponent_configs import OpponentConfig, build_configs
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH
from hybridpoker.phase4_eval import (
    DEV_MASTER_SEED,
    FINAL_MASTER_SEED,
    HERO_LABELS,
    HEROES,
    Estimate,
    final_match_seed,
    make_hero,
    net_by_deal,
    play_match_records,
    to_bb100,
)
from hybridpoker.seeding import derive_seed

MANIFEST_PATH = Path("data/manifests/phase4_eval_manifest.json")
FINAL_DIR = Path("data/processed/phase4_final")
FREEZE_TAG = "phase4-freeze"
FINAL_DEALS = 500
REPLAY_OPPONENTS = (0, 10)
REPLAY_DEALS = 6
BOOTSTRAP_REPLICATES = 10_000
PINNED_PACKAGES = ("pokerkit", "numpy", "pandas", "scikit-learn", "joblib")
FROZEN_EXTRA_FILES = (
    "scripts/run_phase4_final.py",
    "scripts/build_phase4_manifest.py",
    "results/phase3/frequency_prior.json",
    "results/phase3/situation_frequency_prior.json",
    "results/phase3/state_coverage.json",
    "requirements.txt",
    "pyproject.toml",
)

ANALYSIS_PLAN: dict[str, Any] = {
    "document": "docs/phase4_analysis_plan.md",
    "unit": "deal pair: the hero's net chips over both seatings of one deal, in bb/100 "
            "(chips / 2 hands / 2 chips per big blind x 100)",
    "pairing": "paired differences between two heroes on the same deal pair against the same opponent "
               "(identical cards, seats and opponent random streams)",
    "primary": {
        "name": "ML EV minus Situation-Aware Frequency EV",
        "treatment": "ev_ml",
        "control": "ev_situation",
        "estimand": "equal-weight mean over the 16 opponents of the mean paired difference per deal pair (bb/100)",
        "interval": "95% normal interval: mean +/- 1.96 x sqrt(sum over opponents of SE_o^2) / 16, "
                    "SE_o = sample SD of the opponent's paired differences / sqrt(500)",
        "decision": "two-sided test at alpha = 0.05: the ML agent is better if the interval lies above 0, "
                    "worse if below 0; otherwise no difference is detected",
    },
    "secondary": [
        {"name": "ML EV minus Global Frequency EV", "treatment": "ev_ml", "control": "ev_global"},
        {"name": "ML EV minus ProbabilityAgent", "treatment": "ev_ml", "control": "probability"},
    ],
    "secondary_rule": "same estimator and 95% interval as the primary, not adjusted for multiplicity; "
                      "supporting evidence only",
    "subgroups": "training pool (12) vs held-out (4) opponents, and each style (4 opponents each), equal weight "
                 "within the group; exploratory, no confirmatory claims",
    "per_opponent": "descriptive only (16 comparisons: about 0.8 intervals would exclude 0 by chance at 5%)",
    "descriptive": "all six pairwise hero comparisons, win rates, all-in rates and all-in-hand results, "
                   "low-coverage reliance, decision latency",
    "sensitivity": f"stratified bootstrap of the primary (deal pairs resampled within each opponent, "
                   f"{BOOTSTRAP_REPLICATES} replicates, seed derive_seed(4000, 'bootstrap', 'primary')); "
                   "the decision uses the normal interval",
    "errors": "any match error or illegal action invalidates the run: no primary result is reported from an "
              "incomplete run; a fix is a documented protocol deviation and requires a new freeze",
    "no_tuning": "no agent, model, seed, sample-size or analysis change after the freeze; final results are "
                 "never used to tune anything",
}


# --- Hashing -------------------------------------------------------------------------------------------


def normalized_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF line endings normalized to LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def frozen_file_list(root: Path = Path(".")) -> list[str]:
    """Every source file of the package plus the runner, artifacts and dependency pins (sorted)."""
    sources = sorted(p.relative_to(root).as_posix() for p in (root / "src" / "hybridpoker").rglob("*.py"))
    return sorted({*sources, *FROZEN_EXTRA_FILES})


def manifest_hash(manifest: dict[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --- Manifest --------------------------------------------------------------------------------------------


def final_opponents(master_seed: int = FINAL_MASTER_SEED) -> list[dict[str, Any]]:
    return [{**c.to_dict(), "index": i, "match_seed": final_match_seed(c.name, 0, master_seed)}
            for i, c in enumerate(build_configs())]


def build_manifest(hero_configs: dict[str, dict[str, Any]], root: Path = Path(".")) -> dict[str, Any]:
    """The final-evaluation manifest. ``hero_configs``: ``make_hero(name).config()`` for every hero."""
    if list(hero_configs) != list(HEROES):
        raise ValueError(f"Hero configurations must be given for {HEROES} in that order.")
    manifest = {
        "name": "phase4_final_evaluation",
        "schema_version": 1,
        "freeze_tag": FREEZE_TAG,
        "note": "Final runs require the git tag phase4-freeze on the checked-out commit, a clean working tree "
                "and matching file, model and package hashes (scripts/run_phase4_final.py --check).",
        "master_seed": FINAL_MASTER_SEED,
        "seed_formula": 'derive_seed(4000, "eval", opponent_name, 0)',
        "deals_per_match": FINAL_DEALS,
        "seat_swap": True,
        "heroes": list(HEROES),
        "hero_labels": {h: HERO_LABELS[h] for h in HEROES},
        "hero_configs": hero_configs,
        "opponents": final_opponents(),
        "model": {"path": PHASE4_MODEL_PATH.as_posix(), "sha256": FROZEN_MODEL_SHA256},
        "frozen_files": {f: normalized_sha256(root / f) for f in frozen_file_list(root)},
        "environment": {"python": "3.12", "packages": {p: version(p) for p in PINNED_PACKAGES}},
        "replays": {"opponent_indices": list(REPLAY_OPPONENTS), "deals": REPLAY_DEALS},
        "analysis": ANALYSIS_PLAN,
        "development_reference": {"master_seed": DEV_MASTER_SEED, "report": "results/phase4/stage4c_dev_report.md"},
    }
    manifest["manifest_sha256"] = manifest_hash(manifest)
    return manifest


def manifest_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Verification ------------------------------------------------------------------------------------------


def check_manifest(manifest: dict[str, Any], root: Path = Path(".")) -> list[str]:
    """Problems with the manifest itself and with the files it pins (empty list = all good)."""
    problems = []
    if manifest.get("manifest_sha256") != manifest_hash(manifest):
        problems.append("manifest content does not match its manifest_sha256")
    if manifest.get("master_seed") != FINAL_MASTER_SEED:
        problems.append("master seed is not 4000")
    expected = final_opponents()
    if manifest.get("opponents") != expected:
        problems.append("opponent configurations or match seeds differ from build_configs()/final_match_seed")
    if manifest.get("heroes") != list(HEROES):
        problems.append("hero list differs")
    listed = manifest.get("frozen_files", {})
    current = frozen_file_list(root)
    for f in sorted(set(current) - set(listed)):
        problems.append(f"file not covered by the manifest: {f}")
    for f, digest in sorted(listed.items()):
        path = root / f
        if not path.exists():
            problems.append(f"missing frozen file: {f}")
        elif normalized_sha256(path) != digest:
            problems.append(f"changed since the freeze: {f}")
    return problems


def check_environment(manifest: dict[str, Any]) -> list[str]:
    problems = []
    if not platform.python_version().startswith(manifest["environment"]["python"] + "."):
        problems.append(f"Python {platform.python_version()} (manifest: {manifest['environment']['python']})")
    for package, pinned in manifest["environment"]["packages"].items():
        if version(package) != pinned:
            problems.append(f"{package} {version(package)} installed, {pinned} pinned")
    return problems


def check_model(manifest: dict[str, Any], root: Path = Path(".")) -> list[str]:
    from hybridpoker.opponent_predictor import file_sha256

    path = root / manifest["model"]["path"]
    if not path.exists():
        return [f"model file missing: {path} (scripts/restore_phase4_model.py)"]
    return [] if file_sha256(path) == manifest["model"]["sha256"] else ["model SHA-256 differs from the frozen hash"]


def check_heroes(manifest: dict[str, Any], names: Iterable[str] | None = None) -> list[str]:
    problems = []
    for name in names or manifest["heroes"]:
        if make_hero(name).config() != manifest["hero_configs"][name]:
            problems.append(f"hero configuration differs: {name}")
    return problems


def check_git(tag: str = FREEZE_TAG) -> list[str]:
    """The checked-out commit carries ``tag`` and tracked files are unmodified."""
    try:
        tags = subprocess.run(["git", "tag", "--points-at", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.split()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return [f"git unavailable: {error}"]
    problems = [] if tag in tags else [f"HEAD is not tagged {tag}"]
    if dirty:
        problems.append("tracked files have uncommitted changes")
    return problems


def verify_for_run(manifest: dict[str, Any], root: Path = Path(".")) -> dict[str, list[str]]:
    """Every pre-run check, by name. The runner plays only if every list is empty."""
    return {"manifest and frozen files": check_manifest(manifest, root),
            "environment": check_environment(manifest),
            "model": check_model(manifest, root),
            "hero configurations": check_heroes(manifest),
            "git tag and clean tree": check_git(manifest["freeze_tag"])}


# --- Seeds used elsewhere ---------------------------------------------------------------------------------


def earlier_match_seeds(manifest_dir: Path = Path("data/manifests"), dev_opponents: int = 16) -> dict[str, set[int]]:
    """Match seeds of the Phase 3 dataset, the Phase 3 pilot and the Stage 4C development run."""
    from hybridpoker.phase4_eval import dev_match_seed

    seeds = {"stage4c_dev": {dev_match_seed(i) for i in range(dev_opponents)}}
    for name in ("phase3_manifest.json", "phase3_pilot_manifest.json"):
        data = json.loads((manifest_dir / name).read_text(encoding="utf-8"))
        seeds[name] = {int(m["seed"]) for m in data["matches"]}
    return seeds


def deal_seeds(match_seed: int, deals: int) -> set[int]:
    return {derive_seed(match_seed, "deal", i) for i in range(deals)}


# --- Running (resumable) -----------------------------------------------------------------------------------


def job_path(out_dir: Path, hero: str, opponent: str, replay: bool = False) -> Path:
    return Path(out_dir) / ("replays" if replay else "jobs") / f"{hero}__{opponent}.json.gz"


def jobs_from_manifest(manifest: dict[str, Any]) -> tuple[list[tuple], list[tuple]]:
    """(main jobs, replay jobs) as (hero, OpponentConfig, seed, deals, index); costliest opponents first."""
    configs = [(OpponentConfig.from_dict({k: o[k] for k in OpponentConfig.__dataclass_fields__}), o) for o in
               manifest["opponents"]]
    cost = {"solid": 0, "maniac": 1, "calling_station": 2, "nit": 3}  # wall-time order only
    ordered = sorted(configs, key=lambda c: (cost[c[0].family], c[1]["index"]))
    main = [(h, c, o["match_seed"], manifest["deals_per_match"], o["index"]) for c, o in ordered
            for h in manifest["heroes"]]
    replays = [(h, configs[i][0], configs[i][1]["match_seed"], manifest["replays"]["deals"], i)
               for i in manifest["replays"]["opponent_indices"] for h in manifest["heroes"]]
    return main, replays


def _write_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(gzip.compress(json.dumps(record).encode()))
    os.replace(temporary, path)


def _job(args):
    hero, config, seed, deals, index = args
    return play_match_records(hero, config, seed, deals, index)


def run_jobs(main: list[tuple], replays: list[tuple], out_dir: Path, run_id: str, workers: int,
             initializer: Callable[[], None] | None = None, log: Callable[[str], None] = print) -> int:
    """Play every job whose file does not exist yet; returns the number of matches played."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    todo = [(job, False) for job in main if not job_path(out_dir, job[0], job[1].name).exists()]
    todo += [(job, True) for job in replays if not job_path(out_dir, job[0], job[1].name, True).exists()]
    log(f"{len(todo)} of {len(main) + len(replays)} matches to play ({len(main) + len(replays) - len(todo)} done)")
    if not todo:
        return 0
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as pool:
        futures = {pool.submit(_job, job): (job, replay) for job, replay in todo}
        for done, future in enumerate(as_completed(futures), 1):
            (hero, config, *_), replay = futures[future]
            record = {**future.result(), "run_id": run_id}
            _write_atomic(job_path(out_dir, hero, config.name, replay), record)
            status = "ERROR" if record["error"] else f"{len(record['hands'])} hands"
            log(f"  {'replay ' if replay else ''}{hero:<13}{config.name:<20}{status:<12}{record['seconds']:7.1f}s "
                f"({done}/{len(todo)})")
    return len(todo)


def collect(main: list[tuple], replays: list[tuple], out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    def read(path: Path) -> dict[str, Any]:
        return json.loads(gzip.decompress(path.read_bytes()))

    missing = [job_path(out_dir, j[0], j[1].name) for j in main if not job_path(out_dir, j[0], j[1].name).exists()]
    missing += [job_path(out_dir, j[0], j[1].name, True) for j in replays
                if not job_path(out_dir, j[0], j[1].name, True).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} matches have not been played, e.g. {missing[0]}")
    return {"results": [read(job_path(out_dir, j[0], j[1].name)) for j in main],
            "replays": [read(job_path(out_dir, j[0], j[1].name, True)) for j in replays]}


# --- Pre-specified primary analysis --------------------------------------------------------------------------


def paired_estimates(results: list[dict[str, Any]], treatment: str, control: str, deals: int,
                     opponents: list[str]) -> dict[str, Estimate]:
    by = {(r["hero"], r["opponent"]): r for r in results}
    return {o: Estimate.from_units(to_bb100(net_by_deal(by[(treatment, o)]["hands"], deals))
                                   - to_bb100(net_by_deal(by[(control, o)]["hands"], deals))) for o in opponents}


def stratified_bootstrap(per_opponent_units: list[np.ndarray], replicates: int, seed: int) -> tuple[float, float]:
    """95% percentile interval of the equal-weight mean, resampling units within each opponent."""
    rng = np.random.default_rng(seed)
    means = np.zeros(replicates)
    for units in per_opponent_units:
        n = len(units)
        means += units[rng.integers(0, n, size=(replicates, n))].mean(axis=1)
    means /= len(per_opponent_units)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def conclusion(estimate: Estimate, treatment_label: str, control_label: str) -> str:
    """The pre-specified wording of the primary decision (two-sided, alpha = 0.05)."""
    if estimate.low > 0:
        return (f"{treatment_label} won more than {control_label}: {estimate.mean:+.1f} bb/100, 95% CI "
                f"[{estimate.low:+.1f}, {estimate.high:+.1f}] excludes 0 (significant at the 5% level).")
    if estimate.high < 0:
        return (f"{treatment_label} won less than {control_label}: {estimate.mean:+.1f} bb/100, 95% CI "
                f"[{estimate.low:+.1f}, {estimate.high:+.1f}] excludes 0 (significant at the 5% level).")
    return (f"No significant difference was detected between {treatment_label} and {control_label}: "
            f"{estimate.mean:+.1f} bb/100, 95% CI [{estimate.low:+.1f}, {estimate.high:+.1f}] includes 0. "
            f"The data are compatible with effects in that range.")


def primary_analysis(results: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Primary and key secondary endpoints exactly as pre-specified (refuses incomplete or failed runs)."""
    errors = [f"{r['hero']} vs {r['opponent']}" for r in results if r["error"]]
    if errors:
        raise RuntimeError(f"The run is invalid: {len(errors)} matches failed ({', '.join(errors[:5])}).")
    deals = manifest["deals_per_match"]
    opponents = [o["name"] for o in manifest["opponents"]]
    for r in results:
        if len(r["hands"]) != 2 * deals:
            raise RuntimeError(f"{r['hero']} vs {r['opponent']}: {len(r['hands'])} hands, expected {2 * deals}.")
    labels = manifest["hero_labels"]
    out: dict[str, Any] = {}
    for role, spec in [("primary", manifest["analysis"]["primary"]),
                       *[("secondary", s) for s in manifest["analysis"]["secondary"]]]:
        per = paired_estimates(results, spec["treatment"], spec["control"], deals, opponents)
        overall = Estimate.equal_weight(list(per.values()))
        entry = {"role": role, "name": spec["name"], "estimate": overall.__dict__,
                 "conclusion": conclusion(overall, labels[spec["treatment"]], labels[spec["control"]])}
        if role == "primary":
            by = {(r["hero"], r["opponent"]): r for r in results}
            units = [to_bb100(net_by_deal(by[(spec["treatment"], o)]["hands"], deals))
                     - to_bb100(net_by_deal(by[(spec["control"], o)]["hands"], deals)) for o in opponents]
            entry["bootstrap_95"] = stratified_bootstrap(units, BOOTSTRAP_REPLICATES,
                                                         derive_seed(FINAL_MASTER_SEED, "bootstrap", "primary"))
        out[spec["name"]] = entry
    return out


def primary_lines(analysis: dict[str, Any]) -> list[str]:
    lines = ["## Pre-specified endpoints", ""]
    for name, entry in analysis.items():
        e = Estimate(**entry["estimate"])
        label = "Primary" if entry["role"] == "primary" else "Secondary"
        lines += [f"**{label}: {name}.** {entry['conclusion']}", ""]
        if "bootstrap_95" in entry:
            low, high = entry["bootstrap_95"]
            lines += [f"Sensitivity (stratified bootstrap, {BOOTSTRAP_REPLICATES:,} replicates): 95% interval "
                      f"[{low:+.1f}, {high:+.1f}]. Standard error {e.se:.1f} bb/100 from {e.units} deal pairs.", ""]
    return lines

