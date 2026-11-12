"""Tests for hybridpoker.dataset_manifest: pairings, balance, seeds and splits."""

from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from hybridpoker.dataset_manifest import (
    DEFAULT_MANIFEST_DIR,
    HELDOUT_SPLIT,
    MAIN_MANIFEST_NAME,
    PILOT_MANIFEST_NAME,
    build_manifest,
    build_pilot_manifest,
    configs_by_name,
    manifest_text,
    round_robin_rounds,
    split_is_balanced,
)
from hybridpoker.opponent_configs import FAMILY_ORDER, build_configs

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest():
    return build_manifest()


@pytest.fixture(scope="module")
def pilot():
    return build_pilot_manifest()


def pool_matches(manifest):
    return [m for m in manifest["matches"] if m["split"] != HELDOUT_SPLIT]


def heldout_matches(manifest):
    return [m for m in manifest["matches"] if m["split"] == HELDOUT_SPLIT]


def pair(match):
    return frozenset((match["hero"], match["villain"]))


# --- Round robin ------------------------------------------------------------------------


def test_round_robin_covers_every_pair_once() -> None:
    items = [f"c{i}" for i in range(12)]
    rounds = round_robin_rounds(items)
    assert len(rounds) == 11
    for round_pairs in rounds:
        assert sorted(x for p in round_pairs for x in p) == sorted(items)  # everyone plays once
    all_pairs = [frozenset(p) for r in rounds for p in r]
    assert len(all_pairs) == len(set(all_pairs)) == len(list(combinations(items, 2)))


@pytest.mark.parametrize("items", [["a", "b", "c"], ["a", "a"], []])
def test_round_robin_rejects_bad_input(items) -> None:
    with pytest.raises(ValueError):
        round_robin_rounds(items)


# --- Training-pool pairings: balance and no repeats ----------------------------------------


def test_match_counts_and_numbering(manifest) -> None:
    assert [m["number"] for m in manifest["matches"]] == list(range(48))
    assert len(pool_matches(manifest)) == 36
    assert len(heldout_matches(manifest)) == 12
    assert all(m["deal_count"] == 60 for m in manifest["matches"])
    assert manifest["master_seed"] == 3000


def test_training_pool_pairings_are_balanced_without_repeats(manifest) -> None:
    configs = configs_by_name(manifest)
    matches = pool_matches(manifest)
    pairs = [pair(m) for m in matches]
    assert all(len(p) == 2 for p in pairs)  # no configuration plays itself
    assert len(set(pairs)) == 36  # no repeated pair
    appearances = Counter(name for p in pairs for name in p)
    pool_names = {n for n, c in configs.items() if c.pool == "train"}
    assert set(appearances) == pool_names
    assert set(appearances.values()) == {6}  # every configuration plays exactly 6 matches


def test_split_sizes_and_balance(manifest) -> None:
    configs = configs_by_name(manifest)
    splits = Counter(m["split"] for m in manifest["matches"])
    assert splits == {"train": 26, "validation": 5, "test": 5, "heldout": 12}

    train_counts = Counter(n for m in manifest["matches"] if m["split"] == "train" for n in pair(m))
    assert all(train_counts[n] >= 3 for n, c in configs.items() if c.pool == "train")
    for split in ("validation", "test"):
        families = {configs[n].family for m in manifest["matches"] if m["split"] == split for n in pair(m)}
        assert families == set(FAMILY_ORDER), split


def test_appearances_per_split_add_up_for_every_config(manifest) -> None:
    configs = configs_by_name(manifest)
    per_split = Counter((n, m["split"]) for m in pool_matches(manifest) for n in pair(m))
    for name, config in configs.items():
        total = sum(per_split[(name, s)] for s in ("train", "validation", "test"))
        assert total == (6 if config.pool == "train" else 0), name


def test_split_balance_rule_rejects_unbalanced_draws() -> None:
    configs = build_configs()
    names = [c.name for c in configs if c.pool == "train"]
    pairs = [(names[i % 12], names[(i + 1) % 12]) for i in range(36)]
    # Everything in validation/test comes from the first two configurations only.
    splits = ["validation"] * 5 + ["test"] * 5 + ["train"] * 26
    pairs = [(names[0], names[1])] * 10 + pairs[10:]
    assert not split_is_balanced(pairs, splits, configs)


# --- Held-out matches ------------------------------------------------------------------------


def test_heldout_matches_face_three_training_pool_families(manifest) -> None:
    configs = configs_by_name(manifest)
    heldout_names = {n for n, c in configs.items() if c.pool == "heldout"}
    by_config: dict[str, list[str]] = {}
    for m in heldout_matches(manifest):
        scored = m[m["scored_role"]]
        other = m["villain" if m["scored_role"] == "hero" else "hero"]
        assert scored in heldout_names
        assert configs[other].pool == "train"
        by_config.setdefault(scored, []).append(configs[other].family)
    assert set(by_config) == heldout_names
    for name, families in by_config.items():
        assert len(families) == 3
        assert len(set(families)) == 3
        assert configs[name].family not in families
    assert len({pair(m) for m in heldout_matches(manifest)}) == 12


def test_heldout_configs_never_in_training_pool_matches(manifest) -> None:
    configs = configs_by_name(manifest)
    for m in pool_matches(manifest):
        assert configs[m["hero"]].pool == configs[m["villain"]].pool == "train"
        assert m["scored_role"] == "both"


# --- Seeds and reproducibility --------------------------------------------------------------


def test_all_match_seeds_are_distinct(manifest, pilot) -> None:
    seeds = [m["seed"] for m in manifest["matches"] + pilot["matches"]]
    assert len(seeds) == len(set(seeds)) == 52


def test_manifests_are_byte_identical_and_match_committed_files(manifest, pilot) -> None:
    assert manifest_text(build_manifest()) == manifest_text(manifest)
    assert manifest_text(build_pilot_manifest()) == manifest_text(pilot)
    committed = REPO / DEFAULT_MANIFEST_DIR
    assert (committed / MAIN_MANIFEST_NAME).read_text(encoding="utf-8") == manifest_text(manifest)
    assert (committed / PILOT_MANIFEST_NAME).read_text(encoding="utf-8") == manifest_text(pilot)


def test_manifest_contains_no_timestamps_or_versions(manifest) -> None:
    text = manifest_text(manifest)
    for word in ("created", "utc", "git", "python", "pokerkit"):
        assert word not in text.lower()


# --- Pilot ----------------------------------------------------------------------------------


def test_pilot_design(pilot) -> None:
    configs = configs_by_name(pilot)
    matches = pilot["matches"]
    assert [m["split"] for m in matches] == ["pilot_pool"] * 3 + ["pilot_heldout"]
    pool = matches[:3]
    assert len({pair(m) for m in pool}) == 3
    assert {configs[n].family for m in pool for n in pair(m)} == set(FAMILY_ORDER)
    assert all(configs[n].pool == "train" for m in pool for n in pair(m))
    heldout = matches[3]
    scored = heldout[heldout["scored_role"]]
    other = heldout["villain" if heldout["scored_role"] == "hero" else "hero"]
    assert configs[scored].pool == "heldout" and configs[other].pool == "train"
    assert configs[scored].family != configs[other].family
    assert pilot["prior_splits"] == ["pilot_pool"]
