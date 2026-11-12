"""The Phase 3 dataset manifest: which matches to play, with which seeds.

The manifest is a small JSON file committed to git. It lists every opponent
configuration and every match (who plays whom, the match seed, and which
split the match belongs to). Anyone with the manifest and the code can
regenerate the exact same dataset.

Design (see ``docs/phase3_plan.md``, section 4):

* **36 training-pool matches** (numbers 0-35). The 12 training-pool
  configurations are paired with a round-robin "circle method" schedule;
  the first 6 rounds give 36 distinct pairs in which every configuration
  plays exactly 6 matches.
* **Splits** of those 36 matches: 26 train, 5 validation, 5 test, drawn by
  a seeded shuffle that is redrawn until it is balanced (every
  configuration in at least 3 train matches; validation and test each
  contain all four style families).
* **12 held-out matches** (numbers 36-47). Each of the 4 held-out
  configurations plays one training-pool configuration from each of the
  other three families. Only the held-out player's decisions are scored.
* **Pilot**: 4 separate matches with their own seeds, used only to check
  the pipeline.

The manifest contains no timestamps or software versions, so building it
twice gives byte-identical JSON. Those details are recorded per run in
``metadata.json`` and in the generation summary instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hybridpoker.opponent_configs import (
    FAMILY_ORDER,
    HELDOUT_POOL,
    MASTER_SEED,
    TRAIN_POOL,
    OpponentConfig,
    build_configs,
)
from hybridpoker.seeding import derive_seed

MANIFEST_SCHEMA_VERSION = 1
DEALS_PER_MATCH = 60
TRAIN_POOL_ROUNDS = 6
SPLIT_SIZES = (("train", 26), ("validation", 5), ("test", 5))
MIN_TRAIN_APPEARANCES = 3
HELDOUT_SPLIT = "heldout"
PILOT_POOL_SPLIT = "pilot_pool"
PILOT_HELDOUT_SPLIT = "pilot_heldout"
SCORE_BOTH = "both"
ROLES = ("hero", "villain")
MAX_SPLIT_ATTEMPTS = 10_000

DEFAULT_MANIFEST_DIR = Path("data/manifests")
MAIN_MANIFEST_NAME = "phase3_manifest.json"
PILOT_MANIFEST_NAME = "phase3_pilot_manifest.json"


@dataclass(frozen=True)
class MatchSpec:
    """One match to play.

    Attributes:
        number: Position in the manifest (0-47 for the main dataset).
        seed: The match master seed passed to ``run_match``.
        hero, villain: Configuration names. "hero" and "villain" are only
            Phase 2's names for the two players; both are bots here.
        split: ``train``, ``validation``, ``test``, ``heldout``,
            ``pilot_pool`` or ``pilot_heldout``.
        scored_role: Whose decisions count for evaluation: ``both``, or
            the role of the held-out configuration in a held-out match.
    """

    number: int
    seed: int
    hero: str
    villain: str
    split: str
    deal_count: int
    scored_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Pairings -----------------------------------------------------------------------


def round_robin_rounds(items: Sequence[str]) -> list[list[tuple[str, str]]]:
    """All rounds of a round-robin tournament (the "circle method").

    With n (even) items there are n - 1 rounds of n / 2 pairs. Every item
    plays once per round and every pair occurs exactly once overall.
    """
    players = list(items)
    if len(players) % 2 or len(players) < 2:
        raise ValueError("The circle method needs an even number (>= 2) of items.")
    if len(set(players)) != len(players):
        raise ValueError("Items must be distinct.")
    rounds = []
    for _ in range(len(players) - 1):
        n = len(players)
        rounds.append([(players[i], players[n - 1 - i]) for i in range(n // 2)])
        players = [players[0], players[-1], *players[1:-1]]  # rotate all but the first
    return rounds


def training_pool_pairings(
    configs: Sequence[OpponentConfig], master_seed: int = MASTER_SEED
) -> list[tuple[str, str]]:
    """36 distinct pairs in which every training-pool configuration plays 6 times."""
    names = [c.name for c in configs if c.pool == TRAIN_POOL]
    random.Random(derive_seed(master_seed, "pairing")).shuffle(names)
    rounds = round_robin_rounds(names)[:TRAIN_POOL_ROUNDS]
    return [pair for round_pairs in rounds for pair in round_pairs]


def ordered_roles(pair: tuple[str, str], *seed_parts: object) -> tuple[str, str]:
    """A seeded coin flip deciding which configuration is "hero"."""
    first, second = pair
    if random.Random(derive_seed(*seed_parts)).random() < 0.5:
        return first, second
    return second, first


def split_is_balanced(
    pairs: Sequence[tuple[str, str]], splits: Sequence[str], configs: Sequence[OpponentConfig]
) -> bool:
    """The acceptance rule for a split draw (plan section 4.3)."""
    family = {c.name: c.family for c in configs}
    train_counts = Counter(name for pair, split in zip(pairs, splits) if split == "train" for name in pair)
    pool = [c.name for c in configs if c.pool == TRAIN_POOL]
    if any(train_counts[name] < MIN_TRAIN_APPEARANCES for name in pool):
        return False
    for split_name in ("validation", "test"):
        families = {family[name] for pair, s in zip(pairs, splits) if s == split_name for name in pair}
        if families != set(FAMILY_ORDER):
            return False
    return True


def assign_splits(
    pairs: Sequence[tuple[str, str]],
    configs: Sequence[OpponentConfig],
    master_seed: int = MASTER_SEED,
) -> tuple[list[str], int]:
    """Seeded 26/5/5 split of the training-pool matches.

    Returns the split of each pair (same order as ``pairs``) and the
    accepted attempt number.
    """
    if len(pairs) != sum(size for _, size in SPLIT_SIZES):
        raise ValueError("Pair count does not match the split sizes.")
    for attempt in range(MAX_SPLIT_ATTEMPTS):
        order = list(range(len(pairs)))
        random.Random(derive_seed(master_seed, "split", attempt)).shuffle(order)
        splits = [""] * len(pairs)
        position = 0
        for split_name, size in SPLIT_SIZES:
            for index in order[position : position + size]:
                splits[index] = split_name
            position += size
        if split_is_balanced(pairs, splits, configs):
            return splits, attempt
    raise RuntimeError("No balanced split found.")


def heldout_pairings(
    configs: Sequence[OpponentConfig], master_seed: int = MASTER_SEED
) -> list[tuple[str, str]]:
    """(held-out config, training-pool counterpart) for the 12 held-out matches.

    Each held-out configuration meets one training-pool configuration from
    each of the other three families, in ``FAMILY_ORDER``.
    """
    pairs = []
    for heldout in (c for c in configs if c.pool == HELDOUT_POOL):
        for other_family in FAMILY_ORDER:
            if other_family == heldout.family:
                continue
            candidates = [c.name for c in configs if c.pool == TRAIN_POOL and c.family == other_family]
            rng = random.Random(derive_seed(master_seed, "heldout_pairing", heldout.name, other_family))
            pairs.append((heldout.name, rng.choice(candidates)))
    return pairs


# --- Manifests -----------------------------------------------------------------------


def _match_spec(
    number: int, seed: int, pair: tuple[str, str], roles_seed: tuple, split: str,
    deal_count: int, heldout_name: str | None,
) -> MatchSpec:
    hero, villain = ordered_roles(pair, *roles_seed)
    if heldout_name is None:
        scored = SCORE_BOTH
    else:
        scored = "hero" if hero == heldout_name else "villain"
    return MatchSpec(number, seed, hero, villain, split, deal_count, scored)


def build_manifest(master_seed: int = MASTER_SEED, deal_count: int = DEALS_PER_MATCH) -> dict[str, Any]:
    """The main Phase 3 manifest: 36 training-pool and 12 held-out matches."""
    configs = build_configs(master_seed)
    pool_pairs = training_pool_pairings(configs, master_seed)
    splits, attempt = assign_splits(pool_pairs, configs, master_seed)

    matches = []
    for number, (pair, split) in enumerate(zip(pool_pairs, splits)):
        matches.append(
            _match_spec(number, derive_seed(master_seed, "match", number), pair,
                        (master_seed, "roles", number), split, deal_count, None)
        )
    for offset, pair in enumerate(heldout_pairings(configs, master_seed)):
        number = len(pool_pairs) + offset
        matches.append(
            _match_spec(number, derive_seed(master_seed, "match", number), pair,
                        (master_seed, "roles", number), HELDOUT_SPLIT, deal_count, pair[0])
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": "phase3",
        "master_seed": master_seed,
        "deal_count": deal_count,
        "seat_swap": True,
        "split_attempt": attempt,
        "prior_splits": ["train"],
        "configs": [c.to_dict() for c in configs],
        "matches": [m.to_dict() for m in matches],
    }


def build_pilot_manifest(master_seed: int = MASTER_SEED, deal_count: int = DEALS_PER_MATCH) -> dict[str, Any]:
    """The 4-match pilot: 3 training-pool matches covering all four families
    and 1 held-out configuration against a training-pool configuration.

    Pilot seeds come from ``derive_seed(master_seed, "pilot", i)`` and never
    overlap the main dataset. Pilot data is never used for final results.
    """
    configs = build_configs(master_seed)
    rng = random.Random(derive_seed(master_seed, "pilot_pairing"))
    families = list(FAMILY_ORDER)
    rng.shuffle(families)

    def pool_config(family: str) -> str:
        return rng.choice([c.name for c in configs if c.pool == TRAIN_POOL and c.family == family])

    pool_pairs = [
        (pool_config(families[0]), pool_config(families[1])),
        (pool_config(families[2]), pool_config(families[3])),
        (pool_config(families[1]), pool_config(families[3])),
    ]
    heldout = rng.choice([c for c in configs if c.pool == HELDOUT_POOL])
    counterpart_family = rng.choice([f for f in FAMILY_ORDER if f != heldout.family])
    heldout_pair = (heldout.name, pool_config(counterpart_family))

    matches = [
        _match_spec(i, derive_seed(master_seed, "pilot", i), pair,
                    (master_seed, "pilot_roles", i), PILOT_POOL_SPLIT, deal_count, None)
        for i, pair in enumerate(pool_pairs)
    ]
    matches.append(
        _match_spec(3, derive_seed(master_seed, "pilot", 3), heldout_pair,
                    (master_seed, "pilot_roles", 3), PILOT_HELDOUT_SPLIT, deal_count, heldout.name)
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": "phase3_pilot",
        "master_seed": master_seed,
        "deal_count": deal_count,
        "seat_swap": True,
        "split_attempt": None,
        "prior_splits": [PILOT_POOL_SPLIT],
        "configs": [c.to_dict() for c in configs],
        "matches": [m.to_dict() for m in matches],
    }


def manifest_text(manifest: dict[str, Any]) -> str:
    """Canonical JSON text (sorted keys), so equal manifests are byte-identical."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(manifest_text(manifest).encode("utf-8")).hexdigest()


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(manifest_text(manifest), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def match_specs(manifest: dict[str, Any]) -> list[MatchSpec]:
    return [MatchSpec(**m) for m in manifest["matches"]]


def configs_by_name(manifest: dict[str, Any]) -> dict[str, OpponentConfig]:
    return {c["name"]: OpponentConfig.from_dict(c) for c in manifest["configs"]}


def run_dir_name(spec: MatchSpec) -> str:
    return f"match_{spec.number:02d}"
