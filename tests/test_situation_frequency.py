"""Tests for the situation-aware frequency baseline (hybridpoker.situation_frequency).

Small hand-made count tables check the situation keys, the hierarchical
shrinkage and the legal masking; the committed artifact is checked against
the training split's totals and the original global prior.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hybridpoker.action_probabilities import CLASS_ORDER
from hybridpoker.actions import ActionCategory as C
from hybridpoker.agents.ev_agent import (
    FREQUENCY_PRIOR_PATH,
    EVAgent,
    FrequencyResponseModel,
    SituationFrequencyResponseModel,
    make_situation_ev_agent,
)
from hybridpoker.hand_log import observation_columns
from hybridpoker.match import run_match
from hybridpoker.agents import make_agent
from hybridpoker.situation_frequency import (
    MAX_RAISES,
    SITUATION_PRIOR_PATH,
    SMOOTHING_ROWS,
    SituationFrequencies,
    situation_counts,
    situation_key,
    situation_path,
)
from test_ev_agent import FLOP_CHECK, FLOP_FACING_BET, PREFLOP_OPEN, spot


def obs(street="flop", to_call=0, opponent_stack=100, raises=0):
    return {"obs_street": street, "obs_to_call": to_call, "obs_opponent_stack": opponent_stack,
            "obs_raises_this_street": raises}


def counts(fold=0, check_call=0, raise_half_pot=0, raise_pot=0, all_in=0):
    return dict(zip(CLASS_ORDER, (fold, check_call, raise_half_pot, raise_pot, all_in)))


# --- Situation keys -------------------------------------------------------------------------


def test_keys_cover_street_bet_all_in_and_capped_raises() -> None:
    assert situation_key(obs("preflop", 1, 199, 0)) == "preflop|bet=1|all_in=0|raises=0"
    assert situation_key(obs("flop", 0, 150, 0)) == "flop|bet=0|all_in=0|raises=0"
    assert situation_key(obs("turn", 40, 0, 2)) == "turn|bet=1|all_in=1|raises=2"
    assert situation_key(obs("river", 10, 50, 7)) == f"river|bet=1|all_in=0|raises={MAX_RAISES}"


def test_all_in_needs_a_bet_to_face() -> None:
    """A bettor with no chips behind but nothing to call (e.g. after a call all-in) is not 'facing all-in'."""
    assert situation_key(obs("river", 0, 0, 0)) == "river|bet=0|all_in=0|raises=0"


def test_path_runs_from_global_to_most_specific() -> None:
    assert situation_path(obs("turn", 12, 30, 1)) == (
        "global", "turn", "turn|bet=1", "turn|bet=1|all_in=0", "turn|bet=1|all_in=0|raises=1")


def test_unknown_street_is_rejected() -> None:
    with pytest.raises(ValueError):
        situation_key(obs("showdown"))


def test_counts_from_logged_rows() -> None:
    frame = pd.DataFrame([{**obs("flop", 4, 50, 1), "act_category": "fold"},
                          {**obs("flop", 4, 50, 1), "act_category": "check_call"},
                          {**obs("flop", 0, 50, 0), "act_category": "raise_pot"}])
    table = situation_counts(frame)
    assert table == {"flop|bet=0|all_in=0|raises=0": counts(raise_pot=1),
                     "flop|bet=1|all_in=0|raises=1": counts(fold=1, check_call=1)}


# --- Hierarchical shrinkage ------------------------------------------------------------------


def test_global_level_equals_plain_training_shares() -> None:
    table = SituationFrequencies({"flop|bet=0|all_in=0|raises=0": counts(check_call=60, raise_pot=20),
                                  "river|bet=1|all_in=0|raises=1": counts(fold=10, check_call=10)})
    assert np.allclose(table.shares("global"), [0.1, 0.7, 0.0, 0.2, 0.0])


def test_unseen_situation_inherits_its_parent() -> None:
    table = SituationFrequencies({"flop|bet=1|all_in=0|raises=1": counts(fold=30, check_call=70)})
    assert table.rows("flop|bet=1|all_in=1|raises=1") == 0
    assert np.allclose(table.shares("flop|bet=1|all_in=1|raises=1"), table.shares("flop|bet=1|all_in=1"))
    assert np.allclose(table.shares("flop|bet=1|all_in=1"), table.shares("flop|bet=1"))
    # an unseen street falls all the way back to the global shares
    assert np.allclose(table.shares("preflop|bet=1|all_in=1|raises=3"), table.shares("global"))


def test_shrinkage_formula_by_hand() -> None:
    """Leaf with 10 rows (all calls) under a parent with 10 + 90 rows: K = 30 pseudo-rows."""
    table = SituationFrequencies({"river|bet=1|all_in=1|raises=1": counts(check_call=10),
                                  "river|bet=1|all_in=0|raises=1": counts(fold=45, check_call=45)}, 30)
    # every level above 'river|bet=1|all_in=*' holds both leaves (fold 45, call 55 of 100 rows),
    # so global, street and street|bet all have the same shares
    river_bet = table.shares("river|bet=1")
    assert np.allclose(river_bet, [0.45, 0.55, 0, 0, 0])
    all_in = table.shares("river|bet=1|all_in=1")
    expected = (np.array([0, 10, 0, 0, 0]) + 30 * river_bet) / (10 + 30)
    assert np.allclose(all_in, expected)
    leaf = table.shares("river|bet=1|all_in=1|raises=1")
    assert np.allclose(leaf, (np.array([0, 10, 0, 0, 0]) + 30 * all_in) / 40)


def test_well_covered_situations_approach_their_own_frequencies() -> None:
    table = SituationFrequencies({"turn|bet=0|all_in=0|raises=0": counts(check_call=9000, raise_pot=1000),
                                  "turn|bet=1|all_in=0|raises=1": counts(fold=5000, check_call=5000)})
    assert np.allclose(table.shares("turn|bet=0|all_in=0|raises=0"), [0, 0.9, 0, 0.1, 0], atol=0.002)


def test_shares_are_valid_probabilities_and_cached_copies() -> None:
    table = SituationFrequencies({"flop|bet=1|all_in=0|raises=2": counts(1, 2, 3, 4, 5)})
    shares = table.shares("flop|bet=1|all_in=0|raises=2")
    assert np.isclose(shares.sum(), 1.0) and (shares >= 0).all()
    shares[:] = 0  # callers cannot corrupt the cache
    assert np.isclose(table.shares("flop|bet=1|all_in=0|raises=2").sum(), 1.0)


def test_invalid_tables_are_rejected() -> None:
    with pytest.raises(ValueError):
        SituationFrequencies({"flop|bet=1": counts(1)})
    with pytest.raises(ValueError):
        SituationFrequencies({})
    with pytest.raises(ValueError):
        SituationFrequencies({"flop|bet=1|all_in=0|raises=0": counts(1)}, smoothing_rows=0)


# --- Response model inside the shared EV agent ---------------------------------------------


@pytest.fixture
def toy_model() -> SituationFrequencyResponseModel:
    table = {"flop|bet=1|all_in=0|raises=1": counts(fold=80, check_call=10, raise_pot=10),
             "flop|bet=1|all_in=1|raises=1": counts(fold=5, check_call=95),
             "preflop|bet=1|all_in=0|raises=0": counts(fold=40, check_call=60)}
    return SituationFrequencyResponseModel(SituationFrequencies(table, SMOOTHING_ROWS))


def test_predictions_are_legal_masked_per_hypothetical_state(toy_model) -> None:
    """After a hero all-in the opponent may only fold or call; after a half-pot bet raises stay legal."""
    agent = EVAgent("ev_situation", toy_model, equity_fn=lambda o, n, r: 0.5)
    agent.act(spot(FLOP_CHECK), random.Random(1))
    diag = agent.diagnostics[-1]
    all_in = np.array(diag.opponent_probabilities["all_in"])
    assert all_in[2:].sum() == 0 and np.isclose(all_in.sum(), 1.0)
    half = np.array(diag.opponent_probabilities["raise_half_pot"])
    assert half[2:].sum() > 0 and np.isclose(half.sum(), 1.0)
    # the all-in state uses the all-in situation (fold rate far below the non-all-in one)
    assert all_in[0] < half[0]


def test_different_situations_get_different_predictions_unlike_the_global_prior(toy_model) -> None:
    agent = EVAgent("ev_situation", toy_model, equity_fn=lambda o, n, r: 0.5)
    agent.act(spot(FLOP_CHECK), random.Random(1))
    agent.act(spot(PREFLOP_OPEN), random.Random(1))
    flop_fold = agent.diagnostics[0].opponent_probabilities["raise_pot"][0]
    pre_fold = agent.diagnostics[1].opponent_probabilities["raise_pot"][0]
    assert abs(flop_fold - pre_fold) > 0.05
    assert agent.diagnostics[0].coverage  # coverage diagnostics are still recorded


def test_situation_model_needs_no_history_and_has_no_tracker(toy_model) -> None:
    agent = EVAgent("ev_situation", toy_model)
    assert agent.tracker is None
    assert agent.config()["response_model"]["name"] == "situation_frequency_prior"


def test_prediction_reads_only_public_columns(toy_model) -> None:
    """Swapping the hero's hole cards cannot change the opponent prediction."""
    a = spot(FLOP_FACING_BET)
    b = spot(FLOP_FACING_BET, holes="KsKc" + "AsKh")
    pa = toy_model.predict([a], None).masked
    pb = toy_model.predict([b], None).masked
    assert observation_columns(a) == observation_columns(b) and np.array_equal(pa, pb)


# --- Committed artifact ------------------------------------------------------------------------


def test_artifact_matches_training_totals_and_the_global_prior() -> None:
    data = json.loads(Path(SITUATION_PRIOR_PATH).read_text(encoding="utf-8"))
    prior = json.loads(Path(FREQUENCY_PRIOR_PATH).read_text(encoding="utf-8"))
    assert data["split"] == "train" and data["class_order"] == list(CLASS_ORDER)
    assert data["smoothing_rows"] == SMOOTHING_ROWS and data["max_raises"] == MAX_RAISES
    assert data["manifest_hash"] == prior["manifest_hash"]
    totals = {c: sum(row[c] for row in data["counts"].values()) for c in CLASS_ORDER}
    assert totals == prior["counts"] and sum(totals.values()) == data["rows"] == prior["rows"]
    table = SituationFrequencies.load()
    assert np.allclose(table.shares("global"), FrequencyResponseModel.load().shares)


def test_situation_ev_agent_plays_a_legal_deterministic_match() -> None:
    def play():
        hero = make_situation_ev_agent(equity_samples=50)
        hero.range_samples = 100
        match = run_match(hero, make_agent("nit", 50), deal_count=3, seed=11)
        return [h.hero_net for h in match.hands], [d.chosen for d in hero.diagnostics]

    first, second = play(), play()
    assert first == second
    assert len(first[1]) > 0
