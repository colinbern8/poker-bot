"""Tests for hybridpoker.dataset_generation and hybridpoker.dataset_loader."""

import json
import shutil

import pytest

from hybridpoker.dataset_generation import generate_match, preflop_classes
from hybridpoker.dataset_loader import DatasetError, load_dataset, load_match, prior_rows
from hybridpoker.dataset_manifest import manifest_hash, match_specs, run_dir_name
from hybridpoker.hand_log import feature_columns

from conftest import TINY_DEALS, tiny_manifest


def test_preflop_classes_are_the_169_starting_hands() -> None:
    classes = preflop_classes()
    assert len(classes) == len(set(classes)) == 169
    assert {"AA", "AKs", "AKo", "72o", "22"} <= set(classes)


def test_loaded_dataset_has_all_matches_and_bookkeeping(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    manifest = tiny_dataset.manifest
    assert sorted(frame["id_match_number"].unique()) == [0, 1, 2, 3]
    assert frame["id_match"].nunique() == 4
    assert set(frame["id_split"]) == {"pilot_pool", "pilot_heldout"}
    assert frame.groupby("id_match")["id_hand_index"].nunique().eq(2 * TINY_DEALS).all()
    # Bookkeeping columns are never selected as model inputs.
    for column in ("id_match_number", "id_split", "id_scored", "label_actor_family", "label_actor_pool"):
        assert column not in feature_columns(frame)
    # Text columns survive the CSV round trip as strings.
    assert (frame.loc[frame["obs_street"] == "preflop", "obs_board"] == "").all()
    assert frame["obs_bet_to_fold"].dtype == "Int64"
    assert len(prior_rows(frame, manifest)) == (frame["id_split"] == "pilot_pool").sum()


def test_only_heldout_player_is_scored_in_heldout_match(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    pool = frame[frame["id_split"] == "pilot_pool"]
    heldout = frame[frame["id_split"] == "pilot_heldout"]
    assert pool["id_scored"].all()
    assert (heldout.loc[heldout["id_scored"], "label_actor_pool"] == "heldout").all()
    assert (heldout.loc[~heldout["id_scored"], "label_actor_pool"] == "train").all()
    assert heldout["id_scored"].any() and (~heldout["id_scored"]).any()


def test_run_metadata_records_manifest(tiny_dataset) -> None:
    spec = match_specs(tiny_dataset.manifest)[3]
    metadata = json.loads((tiny_dataset.raw_dir / run_dir_name(spec) / "metadata.json").read_text())
    assert metadata["extra"]["manifest_hash"] == manifest_hash(tiny_dataset.manifest)
    assert metadata["extra"]["match_number"] == 3
    assert metadata["extra"]["split"] == "pilot_heldout"
    assert metadata["agents"]["hero"]["name"] == spec.hero


def test_regenerated_match_is_byte_identical(tiny_dataset, tmp_path) -> None:
    spec = match_specs(tiny_dataset.manifest)[0]
    generate_match(spec, tiny_dataset.manifest, tmp_path)
    original = (tiny_dataset.raw_dir / run_dir_name(spec) / "decisions.csv").read_bytes()
    assert (tmp_path / run_dir_name(spec) / "decisions.csv").read_bytes() == original


@pytest.fixture
def copied_run(tiny_dataset, tmp_path):
    spec = match_specs(tiny_dataset.manifest)[1]
    target = tmp_path / run_dir_name(spec)
    shutil.copytree(tiny_dataset.raw_dir / run_dir_name(spec), target)
    return spec, target


def _edit_metadata(run_dir, edit) -> None:
    path = run_dir / "metadata.json"
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def test_incomplete_run_is_refused(tiny_dataset, copied_run) -> None:
    spec, run_dir = copied_run
    (run_dir / "metadata.json").unlink()
    with pytest.raises(DatasetError, match="incomplete"):
        load_match(run_dir, spec, tiny_dataset.manifest)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda d: d.update(seed=d["seed"] + 1), "seed"),
        (lambda d: d["agents"]["hero"].update(looseness=0.123), "hero agent"),
        (lambda d: d["extra"].update(manifest_hash="0" * 64), "manifest hash"),
        (lambda d: d["extra"].update(match_number=99), "match_number"),
        (lambda d: d.update(hand_count=1), "hand_count"),
    ],
)
def test_runs_inconsistent_with_manifest_are_refused(tiny_dataset, copied_run, edit, message) -> None:
    spec, run_dir = copied_run
    _edit_metadata(run_dir, edit)
    with pytest.raises(DatasetError, match=message):
        load_match(run_dir, spec, tiny_dataset.manifest)


def test_run_from_a_different_manifest_is_refused(tiny_dataset) -> None:
    other = tiny_manifest()
    other["configs"][0]["equity_samples"] = 31  # any change alters the hash
    with pytest.raises(DatasetError):
        load_dataset(other, tiny_dataset.raw_dir)
