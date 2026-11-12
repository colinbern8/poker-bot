"""Tests for hypothetical opponent states (hybridpoker.counterfactual) and the
opponent predictor (hybridpoker.opponent_predictor).

Requirements covered (Phase 4 plan, item 3):
* hypothetical states contain only information observable to the acting player;
* pot, bet sizes, stacks, position and betting history are updated correctly;
* the opponent is the predicted actor;
* legal-action masks are respected;
* the approved opponent-history calculation is preserved.
"""

import dataclasses

import joblib
import numpy as np
import pytest
from pokerkit import Card

from hybridpoker.action_probabilities import CLASS_ORDER
from hybridpoker.actions import ActionCategory as C
from hybridpoker.agents import make_agent
from hybridpoker.counterfactual import ReplayMismatch, hypothetical_opponent_observation, replay_public_state
from hybridpoker.dataset_loader import prior_rows
from hybridpoker.game import create_heads_up_state
from hybridpoker.hand_log import observation_columns
from hybridpoker.match import run_match
from hybridpoker.observation import observe
from hybridpoker.opponent_history import B_FEATURES, feature_set_b
from hybridpoker.opponent_model import FeaturePreprocessor, encode_targets, make_model
from hybridpoker.opponent_predictor import ModelHashMismatch, OpponentPredictor, file_sha256
from hybridpoker.opponent_tracker import OpponentTracker
from hybridpoker.simulator import public_hand_summary


def first_observation():
    state = create_heads_up_state()
    for card in Card.parse("2c3dAsKh"):
        state.deal_hole(card)
    return observe(state, history=())


@pytest.fixture(scope="module")
def played():
    matches = [run_match(make_agent(h, 30), make_agent(v, 30), deal_count=8, seed=s)
               for h, v, s in (("probability", "maniac", 31), ("maniac", "solid", 32), ("random", "nit", 33))]
    return [hand.result for m in matches for hand in m.hands]


# --- Correct updates: compare with what actually happened --------------------------------


def test_hypothetical_equals_the_real_next_observation(played) -> None:
    """For every hero action followed by an opponent decision on the same street, the
    hypothetical state built from the hero's view equals the opponent's real observation."""
    checked = 0
    for result in played:
        records = result.decisions
        for before, after in zip(records, records[1:]):
            if after.seat == before.seat or after.observation.street != before.observation.street:
                continue
            hypothetical = hypothetical_opponent_observation(before.observation, before.action)
            assert hypothetical is not None
            assert observation_columns(hypothetical) == observation_columns(after.observation)
            checked += 1
    assert checked > 50


def test_hand_computed_preflop_raise() -> None:
    obs = first_observation()  # button to act: pot 3, owes 1
    half = obs.legal(C.RAISE_HALF_POT)
    assert half.bet_to == 4 and half.chips_added == 3
    h = hypothetical_opponent_observation(obs, half)
    assert h.position == "big_blind" != obs.position  # the opponent is the actor
    assert (h.pot_size, h.to_call, h.own_bet, h.opponent_bet) == (6, 2, 2, 4)
    assert (h.own_stack, h.opponent_stack) == (198, 196)
    assert h.history[-1].short() == "preflop:button:raise_half_pot:4"
    assert h.street == "preflop" and h.board_cards == ()


def test_actions_that_end_the_round_have_no_opponent_decision(played) -> None:
    obs = first_observation()
    assert hypothetical_opponent_observation(obs, obs.legal(C.FOLD)) is None
    # A call that closes a street: find one in played hands.
    for result in played:
        for i, record in enumerate(result.decisions):
            nxt = result.decisions[i + 1] if i + 1 < len(result.decisions) else None
            closes = nxt is None or nxt.observation.street != record.observation.street
            if record.action.category is C.CHECK_CALL and closes:
                assert hypothetical_opponent_observation(record.observation, record.action) is None
                return
    pytest.fail("no street-closing call found")


# --- Only public information ---------------------------------------------------------------


def test_hypothetical_contains_no_hole_cards_and_ignores_hero_cards() -> None:
    obs = first_observation()
    action = obs.legal(C.RAISE_POT)
    h = hypothetical_opponent_observation(obs, action)
    assert h.hole_cards == ()
    other_cards = dataclasses.replace(obs, hole_cards=tuple(Card.parse("7h7d")))
    assert hypothetical_opponent_observation(other_cards, action) == h  # hero's cards play no role


def test_replay_rejects_an_inconsistent_observation() -> None:
    obs = first_observation()
    replay_public_state(obs)  # consistent: no error
    with pytest.raises(ReplayMismatch):
        replay_public_state(dataclasses.replace(obs, pot_size=obs.pot_size + 1))
    with pytest.raises(ValueError):
        hypothetical_opponent_observation(obs, dataclasses.replace(obs.legal(C.RAISE_POT), bet_to=5))


# --- Legal-action masks -------------------------------------------------------------------


def test_after_hero_all_in_the_opponent_cannot_raise() -> None:
    obs = first_observation()
    h = hypothetical_opponent_observation(obs, obs.legal(C.ALL_IN))
    assert set(h.legal_categories) == {C.FOLD, C.CHECK_CALL}
    assert h.to_call == 198 and h.own_stack == 198


# --- Predictor: masks, actor, history ------------------------------------------------------


@pytest.fixture(scope="module")
def bundle(tiny_dataset):
    frame, priors = tiny_dataset.frame, tiny_dataset.priors
    train = prior_rows(frame, tiny_dataset.manifest)
    features = feature_set_b(train, priors)
    prep = FeaturePreprocessor().fit(features)
    model = make_model("random_forest", {"n_estimators": 20, "min_samples_leaf": 5}, seed=3)
    model.fit(prep.transform(features), encode_targets(train["act_category"]))
    return {"model": model, "preprocessor": prep, "priors": priors, "feature_set": "B", "class_order": CLASS_ORDER}


def test_predictor_respects_legal_masks_and_sums_to_one(bundle) -> None:
    obs = first_observation()
    tracker = OpponentTracker(bundle["priors"])
    hypotheticals = [hypothetical_opponent_observation(obs, obs.legal(c)) for c in (C.RAISE_HALF_POT, C.RAISE_POT, C.ALL_IN)]
    result = OpponentPredictor(bundle).predict(hypotheticals, tracker.history_features())
    np.testing.assert_allclose(result.masked.sum(axis=1), 1.0)
    assert (result.masked[2, 2:] == 0).all()  # after an all-in the opponent cannot raise
    assert (result.masked[:2, 0] > 0).all()  # folding to a bet is legal
    assert result.raw.shape == (3, 5)  # raw probabilities kept for diagnostics


def test_predictor_uses_tracker_history_for_every_hypothetical(bundle, tiny_dataset) -> None:
    tracker = OpponentTracker(bundle["priors"])
    match = run_match(make_agent("maniac", 30), make_agent("nit", 30), deal_count=3, seed=5)
    for hand in match.hands:
        tracker.observe_hand_end(public_hand_summary(hand.result, hand.hero_seat, hand.deal_index, hand.seat_swap, 2))
    history = tracker.history_features()
    obs = first_observation()
    hypotheticals = [hypothetical_opponent_observation(obs, obs.legal(c)) for c in (C.RAISE_HALF_POT, C.ALL_IN)]
    features, _ = OpponentPredictor(bundle).features(hypotheticals, history)
    for row in range(len(hypotheticals)):
        np.testing.assert_array_equal(features.loc[row, list(B_FEATURES)].to_numpy(dtype=float),
                                      history.to_numpy(dtype=float))
    assert history["b_log_prior_hands"] > 0


def test_predictor_rejects_wrong_bundles_and_hashes(bundle, tmp_path) -> None:
    with pytest.raises(ValueError):
        OpponentPredictor({**bundle, "class_order": tuple(reversed(CLASS_ORDER))})
    with pytest.raises(ValueError):
        OpponentPredictor({**bundle, "feature_set": "A"})
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    assert isinstance(OpponentPredictor.load(path, expected_sha256=file_sha256(path)), OpponentPredictor)
    with pytest.raises(ModelHashMismatch):
        OpponentPredictor.load(path, expected_sha256="0" * 64)
