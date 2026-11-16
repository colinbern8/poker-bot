"""Tests for the Stage 4C development-evaluation helpers (hybridpoker.phase4_eval)."""

import math

import numpy as np
import pytest

from hybridpoker import phase4_eval
from hybridpoker.agents.ev_agent import make_situation_ev_agent
from hybridpoker.metrics import win_rate
from hybridpoker.opponent_configs import build_configs
from hybridpoker.phase4_eval import (
    DEV_MASTER_SEED,
    Estimate,
    dev_match_seed,
    half_width,
    net_by_deal,
    pairs_needed,
    play_dev_match,
    to_bb100,
)
from hybridpoker.seeding import derive_seed


def test_dev_seeds_follow_the_plan_and_refuse_the_final_master_seed() -> None:
    assert dev_match_seed(3) == derive_seed(DEV_MASTER_SEED, "dev", 3)
    assert len({dev_match_seed(i) for i in range(16)}) == 16
    with pytest.raises(ValueError):
        dev_match_seed(0, master_seed=4000)


def test_estimate_matches_the_phase2_win_rate_for_deal_pairs() -> None:
    chips = [10, -4, 30, -50, 2, 0, 7]
    ours = Estimate.from_units(to_bb100(np.array(chips)))
    reference = win_rate(chips, hands_per_unit=2, big_blind=2)
    assert math.isclose(ours.mean, reference.bb_per_100) and math.isclose(ours.se, reference.standard_error)
    assert math.isclose(ours.low, reference.ci_low) and math.isclose(ours.high, reference.ci_high)


def test_equal_weight_average_and_its_standard_error() -> None:
    a = Estimate.from_units([1.0, 3.0, 5.0])
    b = Estimate.from_units([10.0, 10.0, 16.0])
    avg = Estimate.equal_weight([a, b])
    assert math.isclose(avg.mean, (a.mean + b.mean) / 2)
    assert math.isclose(avg.se, math.sqrt(a.se ** 2 + b.se ** 2) / 2)


def test_sample_size_formulas_are_inverse() -> None:
    sds = [300.0, 500.0, 800.0, 1200.0]
    n = pairs_needed(sds, 20.0)
    assert half_width(sds, n) <= 20.0 < half_width(sds, n - 1)


def test_net_by_deal_sums_both_seatings() -> None:
    hands = [{"deal_index": 0, "hero_net": 5}, {"deal_index": 0, "hero_net": -2}, {"deal_index": 1, "hero_net": 4}]
    assert net_by_deal(hands, 2).tolist() == [3, 4]


def test_probability_hero_match_records_and_replays_exactly() -> None:
    config = build_configs()[8]  # a nit preset
    first = play_dev_match("probability", config, 8, 3)
    again = play_dev_match("probability", config, 8, 2)  # a shorter replay is an exact prefix
    assert first["error"] is None and len(first["hands"]) == 6
    assert all(d["any_low_coverage"] is None for d in first["decisions"])
    assert [(h["hero_net"], h["cards_hash"]) for h in again["hands"]] == \
        [(h["hero_net"], h["cards_hash"]) for h in first["hands"][:4]]
    assert sum(h["hero_net"] for h in first["hands"]) == net_by_deal(first["hands"], 3).sum()


def test_heroes_get_identical_cards_and_ev_diagnostics_line_up(monkeypatch) -> None:
    def fast_hero(name):
        if name == "ev_situation":
            agent = make_situation_ev_agent("ev_situation", equity_samples=60)
            agent.range_samples = 100
            return agent
        return original(name)

    original = phase4_eval.make_hero
    monkeypatch.setattr(phase4_eval, "make_hero", fast_hero)
    config = build_configs()[4]  # maniac preset
    ev = play_dev_match("ev_situation", config, 4, 3)
    prob = play_dev_match("probability", config, 4, 3)
    assert ev["error"] is None and prob["error"] is None
    assert [h["cards_hash"] for h in ev["hands"]] == [h["cards_hash"] for h in prob["hands"]]
    assert all(isinstance(d["chosen_low_coverage"], bool) for d in ev["decisions"])
    assert all(d["latency"] > 0 for d in ev["decisions"])
    assert len(ev["decisions"]) > 0


def test_errors_are_recorded_not_raised(monkeypatch) -> None:
    def broken(name):
        raise RuntimeError("boom")

    monkeypatch.setattr(phase4_eval, "make_hero", broken)
    out = play_dev_match("probability", build_configs()[0], 0, 2)
    assert "boom" in out["error"] and out["hands"] == []
