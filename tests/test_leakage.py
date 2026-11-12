"""Tests for hybridpoker.leakage: audits L1-L11 pass on the real pipeline and
**fail** on deliberately leaky inputs (so a passing audit means something)."""

import copy

import numpy as np
import pandas as pd
import pytest

from hybridpoker import leakage as L
from hybridpoker.dataset_manifest import build_manifest, build_pilot_manifest, manifest_text
from hybridpoker.features import RAISE_VALUES, feature_set_a
from hybridpoker.opponent_history import B_FEATURES, feature_set_b, history_features


@pytest.fixture(scope="module")
def builders(tiny_dataset):
    return {"A": feature_set_a, "B": lambda f: feature_set_b(f, tiny_dataset.priors)}


# --- The real pipeline passes every audit ----------------------------------------------


def test_real_pipeline_passes_all_audits(tiny_dataset, builders) -> None:
    frame, priors, manifest = tiny_dataset.frame, tiny_dataset.priors, tiny_dataset.manifest
    results = [
        L.audit_feature_names(list(feature_set_b(frame, priors).columns)),
        L.audit_scrambling(frame, builders),
        L.audit_future_truncation(frame, builders, sample=40),
        L.audit_history_uses_earlier_deals(frame, priors),
        L.audit_mirror_hand_excluded(frame, priors),
        L.audit_public_replay(frame),
        L.audit_player_role(frame),
        L.audit_split_integrity(frame, [manifest]),
        L.audit_seed_uniqueness([build_manifest(), build_pilot_manifest()], frame),
        L.audit_train_only_priors(frame, manifest),
        L.audit_manifest_reproducible({"phase3": manifest_text(build_manifest()),
                                       "phase3_pilot": manifest_text(build_pilot_manifest())}),
    ]
    failed = [r for r in results if not r.passed]
    assert not failed, failed
    assert [r.audit_id for r in results] == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11a"]


# --- Each audit catches a deliberate leak ----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["label_actor_agent", "obs_pot_size", "b_maniac_share", "a_looseness_hint", "a_hole_rank", "id_split", "pot"],
)
def test_l1_flags_forbidden_names(name) -> None:
    assert not L.audit_feature_names(["a_pot_bb", name]).passed


def test_l1_flags_duplicates() -> None:
    assert not L.audit_feature_names(["a_pot_bb", "a_pot_bb"]).passed


def test_l2_catches_a_builder_that_reads_labels(tiny_dataset) -> None:
    def leaky(frame):
        out = feature_set_a(frame)
        out["a_extra"] = (frame["label_actor_family"] == "maniac").astype(float)
        return out

    assert not L.audit_scrambling(tiny_dataset.frame, {"leaky": leaky}).passed


def test_l2_catches_a_builder_that_reads_hidden_cards(tiny_dataset) -> None:
    def leaky(frame):
        out = feature_set_a(frame)
        out["a_extra"] = frame["hidden_actor_hole_cards"].str.contains("A").astype(float)
        return out

    assert not L.audit_scrambling(tiny_dataset.frame, {"leaky": leaky}).passed


def test_l3_catches_a_builder_that_peeks_at_the_next_decision(tiny_dataset) -> None:
    def leaky(frame):
        out = feature_set_a(frame)
        ordered = L.time_order(frame)
        next_raise = ordered.groupby("id_match")["act_category"].shift(-1).isin(RAISE_VALUES).astype(float)
        out["a_extra"] = next_raise.reindex(frame.index)
        return out

    assert not L.audit_future_truncation(tiny_dataset.frame, {"leaky": leaky}).passed


def _inclusive_history(frame, priors):
    """A WRONG history builder that also counts the current deal."""
    shifted = frame.copy()
    shifted["id_deal_index"] = shifted["id_deal_index"] - 1  # a copy of the current deal now looks "earlier"
    later = frame.copy()
    combined = pd.concat([shifted, later], keys=["copy", "orig"])
    features = history_features(combined, priors).loc["orig"]
    return features


def test_l4_l5_catch_history_that_includes_the_current_deal(tiny_dataset, monkeypatch) -> None:
    monkeypatch.setattr(L, "history_features", _inclusive_history)
    frame, priors = tiny_dataset.frame, tiny_dataset.priors
    assert not L.audit_history_uses_earlier_deals(frame, priors).passed
    assert not L.audit_mirror_hand_excluded(frame, priors).passed


def test_l6_catches_an_observation_that_contains_private_information(tiny_dataset) -> None:
    frame = tiny_dataset.frame.copy()
    # Pretend the logger leaked hand strength into the "public" pot odds column.
    frame["obs_pot_odds"] = frame["obs_pot_odds"] + frame["hidden_actor_hole_cards"].str.count("A") * 0.01
    assert not L.audit_public_replay(frame).passed


def test_l7_catches_wrong_role(tiny_dataset) -> None:
    frame = tiny_dataset.frame.copy()
    frame.loc[frame.index[0], "label_actor_role"] = "villain" if frame.iloc[0]["label_actor_role"] == "hero" else "hero"
    assert not L.audit_player_role(frame).passed


def test_l8_catches_heldout_config_in_training_split(tiny_dataset) -> None:
    manifest = copy.deepcopy(tiny_dataset.manifest)
    heldout = next(c["name"] for c in manifest["configs"] if c["pool"] == "heldout")
    manifest["matches"][0]["hero"] = heldout
    assert not L.audit_split_integrity(tiny_dataset.frame, [manifest]).passed


def test_l8_catches_wrong_scored_rows(tiny_dataset) -> None:
    frame = tiny_dataset.frame.copy()
    frame.loc[frame["id_split"] == "pilot_heldout", "id_scored"] = True
    assert not L.audit_split_integrity(frame, [tiny_dataset.manifest]).passed


def test_l8_catches_match_in_two_splits(tiny_dataset) -> None:
    frame = tiny_dataset.frame.copy()
    first_match = frame["id_match"] == frame["id_match"].iloc[0]
    frame.loc[first_match & (frame["id_deal_index"] == 0), "id_split"] = "pilot_heldout"
    assert not L.audit_split_integrity(frame, [tiny_dataset.manifest]).passed


def test_l9_catches_repeated_seed() -> None:
    manifest = build_pilot_manifest()
    manifest["matches"][1]["seed"] = manifest["matches"][0]["seed"]
    assert not L.audit_seed_uniqueness([manifest]).passed


def test_l10_catches_priors_fitted_on_all_rows(tiny_dataset, monkeypatch) -> None:
    monkeypatch.setattr(L, "prior_rows", lambda frame, manifest: frame)  # wrong: uses every split
    assert not L.audit_train_only_priors(tiny_dataset.frame, tiny_dataset.manifest).passed


def test_l11_catches_changed_manifest_and_bytes() -> None:
    assert not L.audit_manifest_reproducible({"phase3_pilot": "{}\n"}).passed
    assert not L.audit_regenerated_bytes(b"abc", b"abd", "match_00").passed
    assert L.audit_regenerated_bytes(b"abc", b"abc", "match_00").passed


def test_scrambling_changes_private_columns_but_not_keys(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    scrambled = L.scramble_private_columns(frame, seed=1)
    for key in ("id_match", "id_deal_index", "id_seat_swap"):
        pd.testing.assert_series_equal(scrambled[key], frame[key])
    assert not scrambled["hidden_actor_hole_cards"].equals(frame["hidden_actor_hole_cards"])
    assert not scrambled["label_actor_agent"].equals(frame["label_actor_agent"])
    obs = [c for c in frame.columns if c.startswith("obs_")]
    pd.testing.assert_frame_equal(scrambled[obs], frame[obs])


def test_b_feature_names_pass_allowlist() -> None:
    assert L.audit_feature_names(list(B_FEATURES)).passed
    assert np.all([name.startswith("b_") for name in B_FEATURES])
