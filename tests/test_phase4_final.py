"""Tests for the Stage 4D freeze tooling (hybridpoker.phase4_final).

No final (master seed 4000) hand is ever played here: runs use other seeds,
and seed-4000 checks only compare seed numbers.
"""

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hybridpoker.agents.ev_agent import FROZEN_MODEL_SHA256
from hybridpoker.opponent_configs import build_configs
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH
from hybridpoker.phase4_eval import HEROES, Estimate, final_match_seed, net_by_deal, play_match_records
from hybridpoker.phase4_final import (
    ANALYSIS_PLAN,
    FINAL_DEALS,
    MANIFEST_PATH,
    build_manifest,
    check_git,
    check_heroes,
    check_manifest,
    collect,
    conclusion,
    deal_seeds,
    earlier_match_seeds,
    final_opponents,
    frozen_file_list,
    job_path,
    load_manifest,
    manifest_hash,
    normalized_sha256,
    primary_analysis,
    run_jobs,
    stratified_bootstrap,
)
from hybridpoker.seeding import derive_seed

TEST_MASTER = 9999  # never 4000 or 5000


# --- Seeds ---------------------------------------------------------------------------------------------


def test_final_seeds_follow_the_plan_and_are_distinct() -> None:
    opponents = final_opponents()
    assert [o["name"] for o in opponents] == [c.name for c in build_configs()]
    for o in opponents:
        assert o["match_seed"] == derive_seed(4000, "eval", o["name"], 0) == final_match_seed(o["name"])
    assert len({o["match_seed"] for o in opponents}) == 16


def test_final_match_seeds_never_overlap_earlier_seeds() -> None:
    final = {o["match_seed"] for o in final_opponents()}
    for name, seeds in earlier_match_seeds().items():
        assert final.isdisjoint(seeds), name
    assert min(final) > 10**6  # Phase 2 matches and tests used small integer seeds


def test_final_deal_seeds_never_overlap_earlier_deal_seeds() -> None:
    final = set().union(*(deal_seeds(o["match_seed"], FINAL_DEALS) for o in final_opponents()))
    assert len(final) == 16 * FINAL_DEALS
    manifests = {name: json.loads(Path("data/manifests", name).read_text(encoding="utf-8"))
                 for name in ("phase3_manifest.json", "phase3_pilot_manifest.json")}
    earlier = set()
    for seed in earlier_match_seeds()["stage4c_dev"]:
        earlier |= deal_seeds(seed, FINAL_DEALS)  # covers any development run up to 500 deals
    for data in manifests.values():
        for match in data["matches"]:
            earlier |= deal_seeds(int(match["seed"]), int(match["deal_count"]))
    assert final.isdisjoint(earlier)


# --- Manifest --------------------------------------------------------------------------------------------


def _fake_configs() -> dict:
    return {h: {"name": h} for h in HEROES}


def test_manifest_hash_detects_any_change() -> None:
    manifest = build_manifest(_fake_configs())
    assert manifest["manifest_sha256"] == manifest_hash(manifest)
    changed = {**manifest, "deals_per_match": 499}
    assert manifest_hash(changed) != manifest["manifest_sha256"]


def test_manifest_pins_seed_deals_heroes_and_analysis() -> None:
    manifest = build_manifest(_fake_configs())
    assert manifest["master_seed"] == 4000 and manifest["deals_per_match"] == 500 and manifest["seat_swap"]
    assert manifest["heroes"] == ["probability", "ev_global", "ev_situation", "ev_ml"]
    assert manifest["analysis"]["primary"]["treatment"] == "ev_ml"
    assert manifest["analysis"]["primary"]["control"] == "ev_situation"
    assert [(s["treatment"], s["control"]) for s in manifest["analysis"]["secondary"]] == \
        [("ev_ml", "ev_global"), ("ev_ml", "probability")]
    assert manifest["model"]["sha256"] == FROZEN_MODEL_SHA256
    assert set(manifest["frozen_files"]) == set(frozen_file_list())
    assert "src/hybridpoker/agents/ev_agent.py" in manifest["frozen_files"]
    assert "results/phase3/situation_frequency_prior.json" in manifest["frozen_files"]
    assert check_manifest(manifest) == []


def test_changed_missing_or_unlisted_files_are_detected(tmp_path) -> None:
    for f in frozen_file_list():
        target = tmp_path / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
    manifest = build_manifest(_fake_configs(), root=tmp_path)
    assert check_manifest(manifest, tmp_path) == []
    (tmp_path / "src/hybridpoker/ev.py").write_text("# changed\n", encoding="utf-8")
    (tmp_path / "results/phase3/frequency_prior.json").unlink()
    (tmp_path / "src/hybridpoker/new_module.py").write_text("x = 1\n", encoding="utf-8")
    problems = check_manifest(manifest, tmp_path)
    assert "changed since the freeze: src/hybridpoker/ev.py" in problems
    assert "missing frozen file: results/phase3/frequency_prior.json" in problems
    assert "file not covered by the manifest: src/hybridpoker/new_module.py" in problems


def test_line_endings_do_not_change_file_hashes(tmp_path) -> None:
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "b.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    assert normalized_sha256(tmp_path / "a.py") == normalized_sha256(tmp_path / "b.py")


def test_wrong_master_seed_is_detected() -> None:
    manifest = build_manifest(_fake_configs())
    manifest["master_seed"] = 5000
    manifest["manifest_sha256"] = manifest_hash(manifest)
    assert "master seed is not 4000" in check_manifest(manifest)


def test_missing_freeze_tag_blocks_the_run() -> None:
    assert any("is not tagged" in p for p in check_git("no-such-tag-for-tests"))


def test_committed_manifest_is_self_consistent_and_plans_the_approved_design() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest["manifest_sha256"] == manifest_hash(manifest)
    assert manifest["opponents"] == final_opponents()
    assert manifest["analysis"] == json.loads(json.dumps(ANALYSIS_PLAN))
    assert check_heroes(manifest, ["probability", "ev_global", "ev_situation"]) == []


@pytest.mark.skipif(not PHASE4_MODEL_PATH.exists(), reason="frozen model file is kept outside the repository")
def test_committed_manifest_ml_hero_configuration_matches() -> None:
    assert check_heroes(load_manifest(MANIFEST_PATH), ["ev_ml"]) == []


# --- Pairing and statistics -----------------------------------------------------------------------------


def test_identical_heroes_on_shared_deals_have_exactly_zero_paired_difference() -> None:
    """Common random numbers: the same agent twice on the same seed gives identical results deal by deal."""
    config = build_configs()[12]  # solid preset
    seed = derive_seed(TEST_MASTER, "eval", config.name, 0)
    a = play_match_records("probability", config, seed, 3, 12)
    b = play_match_records("probability", config, seed, 3, 12)
    diff = net_by_deal(a["hands"], 3) - net_by_deal(b["hands"], 3)
    assert np.array_equal(diff, np.zeros(3))
    est = Estimate.from_units(diff)
    assert est.mean == 0 and est.se == 0


def test_bootstrap_interval_is_close_to_the_normal_interval_for_large_samples() -> None:
    rng = np.random.default_rng(0)
    units = [rng.normal(10 * k, 300, 500) for k in range(4)]
    low, high = stratified_bootstrap(units, 4000, seed=1)
    normal = Estimate.equal_weight([Estimate.from_units(u) for u in units])
    assert abs(low - normal.low) < 3 and abs(high - normal.high) < 3


def test_conclusion_wording_follows_the_interval() -> None:
    assert "won more" in conclusion(Estimate(50, 10, 30, 70, 100, 1), "ML", "Control")
    assert "won less" in conclusion(Estimate(-50, 10, -70, -30, 100, 1), "ML", "Control")
    assert "No significant difference" in conclusion(Estimate(5, 10, -15, 25, 100, 1), "ML", "Control")


def _fake_result(hero, opponent, nets, error=None):
    hands = [{"deal_index": i // 2, "seat_swap": i % 2, "hero_net": n} for i, n in enumerate(nets)]
    return {"hero": hero, "opponent": opponent, "hands": hands, "error": error}


def _fake_run(deals=4, error=None):
    manifest = build_manifest(_fake_configs())
    manifest["deals_per_match"] = deals
    rng = np.random.default_rng(3)
    results = []
    for o in manifest["opponents"]:
        for h in HEROES:
            results.append(_fake_result(h, o["name"], list(rng.integers(-50, 50, 2 * deals))))
    if error:
        results[0]["error"] = error
    return results, manifest


def test_primary_analysis_matches_a_hand_computation() -> None:
    results, manifest = _fake_run()
    out = primary_analysis(results, manifest)
    primary = out["ML EV minus Situation-Aware Frequency EV"]
    by = {(r["hero"], r["opponent"]): r for r in results}
    per = []
    for o in manifest["opponents"]:
        d = (net_by_deal(by[("ev_ml", o["name"])]["hands"], 4) - net_by_deal(by[("ev_situation", o["name"])]["hands"], 4))
        per.append(Estimate.from_units(d / 2 / 2 * 100))
    expected = Estimate.equal_weight(per)
    assert primary["role"] == "primary"
    assert np.isclose(primary["estimate"]["mean"], expected.mean) and np.isclose(primary["estimate"]["se"], expected.se)
    assert [v["role"] for v in out.values()] == ["primary", "secondary", "secondary"]


def test_primary_analysis_refuses_failed_or_incomplete_runs() -> None:
    results, manifest = _fake_run(error="IllegalActionError")
    with pytest.raises(RuntimeError, match="invalid"):
        primary_analysis(results, manifest)
    results, manifest = _fake_run()
    results[5]["hands"] = results[5]["hands"][:-2]
    with pytest.raises(RuntimeError, match="hands"):
        primary_analysis(results, manifest)


# --- Resumable runs ---------------------------------------------------------------------------------------


def test_runs_resume_without_replaying_and_reproduce_exactly(tmp_path) -> None:
    configs = build_configs()
    jobs = [("probability", configs[i], derive_seed(TEST_MASTER, "eval", configs[i].name, 0), 2, i) for i in (0, 8)]
    replays = [("probability", configs[0], jobs[0][2], 1, 0)]
    assert run_jobs(jobs, replays, tmp_path, "first", workers=2, log=lambda m: None) == 3
    first = collect(jobs, replays, tmp_path)
    assert run_jobs(jobs, replays, tmp_path, "second", workers=2, log=lambda m: None) == 0
    path = job_path(tmp_path, "probability", configs[8].name)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        collect(jobs, replays, tmp_path)
    assert run_jobs(jobs, replays, tmp_path, "third", workers=1, log=lambda m: None) == 1
    again = json.loads(gzip.decompress(path.read_bytes()))
    before = first["results"][1]
    assert again["run_id"] == "third" and before["run_id"] == "first"
    assert again["hands"] == before["hands"] and again["result_hash"] == before["result_hash"]
    assert [d["category"] for d in again["decisions"]] == [d["category"] for d in before["decisions"]]
