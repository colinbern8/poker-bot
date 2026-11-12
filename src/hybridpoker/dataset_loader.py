"""Load generated match logs and check them against the dataset manifest.

A run directory is accepted only if:

* it is complete (``metadata.json`` exists; Phase 2's atomic-write rule);
* its seed, deal count and both agents' full configurations equal the
  manifest's entry for that match;
* it was generated from the same manifest (matching manifest hash).

The loader adds a few bookkeeping columns. They use non-feature prefixes,
so ``feature_columns`` and the Phase 3 feature builders never see them:

``id_match_number``    Position of the match in the manifest.
``id_split``           train / validation / test / heldout / pilot_*.
``id_scored``          Whether this row counts for evaluation (in held-out
                       matches only the held-out player's rows do).
``label_actor_family`` Preset family of the acting configuration.
``label_actor_pool``   ``train`` or ``heldout`` pool of the acting configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hybridpoker.actions import ActionCategory
from hybridpoker.dataset_manifest import (
    SCORE_BOTH,
    MatchSpec,
    configs_by_name,
    manifest_hash,
    match_specs,
    run_dir_name,
)
from hybridpoker.opponent_history import player_role

TEXT_COLUMNS = ("obs_board", "obs_history")
BET_TO_COLUMNS = tuple(f"obs_bet_to_{c.value}" for c in ActionCategory)


class DatasetError(RuntimeError):
    """A run is missing, incomplete, or does not match the manifest."""


def read_decisions(path: Path) -> pd.DataFrame:
    """Read ``decisions.csv`` with the same column types the simulator wrote.

    Empty boards and histories become ``""`` (CSV stores them as blanks),
    and legal bet sizes stay whole numbers (empty where illegal).
    """
    frame = pd.read_csv(path)
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str)
    return frame.astype({column: "Int64" for column in BET_TO_COLUMNS})


def _expected_agent(config) -> dict[str, Any]:
    return {"class": "StyleAgent", "name": config.name, **config.parameters().__dict__}


def check_run_metadata(metadata: dict[str, Any], spec: MatchSpec, manifest: dict[str, Any]) -> None:
    """Raise ``DatasetError`` if a run's metadata disagrees with the manifest."""
    configs = configs_by_name(manifest)
    problems = []
    if metadata.get("seed") != spec.seed:
        problems.append(f"seed {metadata.get('seed')} != {spec.seed}")
    if metadata.get("deal_count") != spec.deal_count:
        problems.append(f"deal_count {metadata.get('deal_count')} != {spec.deal_count}")
    if metadata.get("hand_count") != 2 * spec.deal_count:
        problems.append(f"hand_count {metadata.get('hand_count')} != {2 * spec.deal_count}")
    for role, name in (("hero", spec.hero), ("villain", spec.villain)):
        if metadata.get("agents", {}).get(role) != _expected_agent(configs[name]):
            problems.append(f"{role} agent does not match configuration {name!r}")
    extra = metadata.get("extra", {})
    if extra.get("manifest_hash") != manifest_hash(manifest):
        problems.append("manifest hash differs")
    if extra.get("match_number") != spec.number:
        problems.append(f"match_number {extra.get('match_number')} != {spec.number}")
    if problems:
        raise DatasetError(f"Run for match {spec.number} is inconsistent: " + "; ".join(problems))


def load_match(run_dir: Path, spec: MatchSpec, manifest: dict[str, Any]) -> pd.DataFrame:
    """Load and validate one match; add the bookkeeping columns."""
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise DatasetError(f"Run {run_dir} is missing or incomplete (no metadata.json).")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    check_run_metadata(metadata, spec, manifest)

    frame = read_decisions(run_dir / "decisions.csv")
    if set(frame["id_match"]) != {metadata["match_id"]}:
        raise DatasetError(f"Run {run_dir} has unexpected match ids.")

    configs = configs_by_name(manifest)
    frame["id_match_number"] = spec.number
    frame["id_split"] = spec.split
    roles = player_role(frame)
    frame["id_scored"] = True if spec.scored_role == SCORE_BOTH else (roles == spec.scored_role)
    frame["label_actor_family"] = frame["label_actor_agent"].map(lambda n: configs[n].family)
    frame["label_actor_pool"] = frame["label_actor_agent"].map(lambda n: configs[n].pool)
    return frame


def prior_rows(frame: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    """Only the rows allowed to fit history priors (``train``; ``pilot_pool`` in the pilot)."""
    return frame[frame["id_split"].isin(manifest["prior_splits"])]


def load_dataset(
    manifest: dict[str, Any], raw_dir: Path, numbers: list[int] | None = None
) -> pd.DataFrame:
    """Load every match in the manifest (or only ``numbers``) from ``raw_dir``."""
    frames = []
    for spec in match_specs(manifest):
        if numbers is not None and spec.number not in numbers:
            continue
        frames.append(load_match(Path(raw_dir) / run_dir_name(spec), spec, manifest))
    if not frames:
        raise DatasetError("No matches selected.")
    return pd.concat(frames, ignore_index=True)
