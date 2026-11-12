"""Tests for the Phase 4 EV agents (hybridpoker.agents.ev_agent).

Controlled spots are built in PokerKit, equity is injected as a fixed
number, and the opponent model is a stub with known probabilities, so
every decision can be checked by hand.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pokerkit import Card, Deck

from hybridpoker.action_probabilities import CLASS_ORDER, mask_and_renormalize, legal_mask
from hybridpoker.actions import ActionCategory as C, apply_action
from hybridpoker.agents import make_agent
from hybridpoker.agents.ev_agent import (
    FROZEN_MODEL_SHA256,
    EVAgent,
    FrequencyResponseModel,
    MLResponseModel,
    ResponseModel,
    make_ml_ev_agent,
)
from hybridpoker.agents.probability import ProbabilityAgent
from hybridpoker.game import SEAT_POSITIONS, STREET_NAMES, create_heads_up_state
from hybridpoker.hand_log import decisions_frame, observation_columns, write_run
from hybridpoker.match import run_match
from hybridpoker.observation import PublicAction, observe
from hybridpoker.opponent_history import history_features
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH, OpponentPredictor
from hybridpoker.state_coverage import COVERAGE_PATH, coverage_counts, coverage_key, load_coverage
import pandas as pd

HOLES = "2c3dAsKh"
BOARD = "KdJc5s7h2s"


def spot(actions: list[C]):
    """Play ``actions`` from the start of a hand and return the next actor's observation."""
    state = create_heads_up_state()
    used = set(Card.parse(HOLES + BOARD))
    burns = iter([c for c in Deck.STANDARD if c not in used])
    board = iter(Card.parse(BOARD))
    for card in Card.parse(HOLES):
        state.deal_hole(card)
    history: list[PublicAction] = []

    def deal() -> None:
        while True:
            if state.can_burn_card():
                state.burn_card(next(burns))
            elif state.can_deal_board():
                state.deal_board(next(board))
            else:
                return

    for category in actions:
        deal()
        obs = observe(state, tuple(history))
        action = obs.legal(category)
        assert action is not None, f"{category} not legal"
        seat = state.actor_index
        street = STREET_NAMES[state.street_index]
        apply_action(state, action)
        history.append(PublicAction(street, SEAT_POSITIONS[seat], category, action.bet_to, action.chips_added,
                                    action.is_all_in))
    deal()
    return observe(state, tuple(history))


PREFLOP_OPEN = []                                              # button owes 1
FLOP_CHECK = [C.CHECK_CALL, C.CHECK_CALL]                      # big blind first to act on the flop
FLOP_FACING_BET = FLOP_CHECK + [C.CHECK_CALL, C.RAISE_HALF_POT]  # big blind faces a bet
FLOP_FACING_RAISE = FLOP_CHECK + [C.RAISE_HALF_POT, C.RAISE_POT]  # big blind bet, button raised
FACING_ALL_IN = [C.ALL_IN]                                     # big blind faces a preflop shove


class StubModel(ResponseModel):
    """Returns the same raw probabilities for every state, legal-masked like the real models.

    The stub uses no probability floor (floor=0), so its probabilities are exact and
    ties can be tested exactly. The real models keep the approved 1e-6 floor.
    """

    name = "stub"

    def __init__(self, raw) -> None:
        self.raw = np.asarray(raw, dtype=float)
        self.calls = 0

    def predict(self, observations, tracker):
        self.calls += 1
        frame = pd.DataFrame([observation_columns(o) for o in observations])
        return mask_and_renormalize(np.tile(self.raw, (len(observations), 1)), legal_mask(frame), floor=0.0)


ALWAYS_FOLD = [1, 0, 0, 0, 0]
ALWAYS_CALL = [0, 1, 0, 0, 0]
ALWAYS_RERAISE = [0, 0, 0.5, 0.5, 0]


def agent_with(raw, equity: float) -> EVAgent:
    return EVAgent("test", StubModel(raw), equity_fn=lambda obs, n, rng: equity, coverage=load_coverage())


def decide(actions, raw, equity):
    agent = agent_with(raw, equity)
    decision = agent.act(spot(actions), rng=None)
    return decision, agent.diagnostics[-1]


# --- Zero-cost checks and folds -------------------------------------------------------------


def test_free_check_beats_bets_that_get_reraised() -> None:
    decision, diag = decide(FLOP_CHECK, ALWAYS_RERAISE, 0.3)
    assert decision.category is C.CHECK_CALL
    assert diag.action_evs["check_call"] == pytest.approx(0.3 * 4)  # pot 4 after a limped preflop
    assert all(diag.action_evs[r] < 0 for r in ("raise_half_pot", "raise_pot", "all_in"))
    assert "fold" not in diag.action_evs  # never offered when checking is free


def test_zero_equity_still_checks_rather_than_folding() -> None:
    decision, _ = decide(FLOP_CHECK, ALWAYS_RERAISE, 0.0)
    assert decision.category is C.CHECK_CALL


# --- Calls and folds at pot odds ---------------------------------------------------------------


def test_call_or_fold_flips_at_break_even_equity() -> None:
    obs = spot(FLOP_FACING_BET)
    break_even = obs.to_call / (obs.pot_size + obs.to_call)
    assert decide(FLOP_FACING_BET, ALWAYS_RERAISE, break_even + 0.01)[0].category is C.CHECK_CALL
    assert decide(FLOP_FACING_BET, ALWAYS_RERAISE, break_even - 0.01)[0].category is C.FOLD
    # Exactly break-even: call and fold both have EV 0 -> the tie goes to fold (fewer chips).
    decision, diag = decide(FLOP_FACING_BET, ALWAYS_RERAISE, break_even)
    assert diag.action_evs["check_call"] == pytest.approx(0.0, abs=1e-9)
    assert decision.category is C.FOLD


def test_facing_a_raise_values_only_the_new_chips() -> None:
    obs = spot(FLOP_FACING_RAISE)
    _, diag = decide(FLOP_FACING_RAISE, ALWAYS_RERAISE, 0.45)
    assert diag.action_evs["check_call"] == pytest.approx(0.45 * (obs.pot_size + obs.to_call) - obs.to_call)
    assert diag.action_evs["fold"] == 0.0


# --- Bet sizes -------------------------------------------------------------------------------


def test_opponent_who_always_folds_gets_the_smallest_bet() -> None:
    decision, diag = decide(FLOP_CHECK, ALWAYS_FOLD, 0.5)
    for size in ("raise_half_pot", "raise_pot", "all_in"):
        assert diag.action_evs[size] == pytest.approx(4.0)  # every bet wins the pot of 4
    assert decision.category is C.RAISE_HALF_POT  # tie between bets -> fewest chips


def test_certain_win_ties_check_with_bets_and_checks() -> None:
    decision, _ = decide(FLOP_CHECK, ALWAYS_FOLD, 1.0)  # check EV 4 = bet EV 4 -> 0 chips wins the tie
    assert decision.category is C.CHECK_CALL


def test_always_calling_opponent_hand_computed_bet_values() -> None:
    obs = spot(FLOP_CHECK)
    _, diag = decide(FLOP_CHECK, ALWAYS_CALL, 0.9)
    for action in obs.legal_actions:
        if action.category in (C.RAISE_HALF_POT, C.RAISE_POT, C.ALL_IN):
            a = action.chips_added  # the opponent adds the same a to call (no bets yet this street)
            assert diag.opponent_probabilities[action.category.value][1] == 1.0
            assert diag.action_evs[action.category.value] == pytest.approx(0.9 * (4 + 2 * a) - a)
    assert diag.chosen == "all_in"  # EV grows with the bet when always called
    assert decide(FLOP_CHECK, ALWAYS_CALL, 0.4)[0].category is C.CHECK_CALL  # 0.4 x (4 + 2a) - a < 1.6


def test_opponent_who_always_reraises_is_never_bet_into() -> None:
    obs = spot(FLOP_CHECK)
    for equity in (0.2, 0.6, 0.95):
        decision, diag = decide(FLOP_CHECK, ALWAYS_RERAISE, equity)
        for size in ("raise_half_pot", "raise_pot"):
            assert diag.action_evs[size] == pytest.approx(-obs.legal(C(size)).chips_added)  # lose the bet
            assert decision.category.value != size
    # After an all-in a re-raise is impossible: the stub's re-raise mass is masked away and
    # (with nothing left on legal actions) becomes fold 0.5 / call 0.5. So all-in is valued
    # as 0.5 x pot + 0.5 x [e x (4 + 2 x 198) - 198] and can still be chosen with high equity.
    decision, diag = decide(FLOP_CHECK, ALWAYS_RERAISE, 0.2)
    assert diag.opponent_probabilities["all_in"][:2] == [0.5, 0.5]
    assert diag.action_evs["all_in"] == pytest.approx(0.5 * 4 + 0.5 * (0.2 * 400 - 198))
    assert decision.category is C.CHECK_CALL
    assert decide(FLOP_CHECK, ALWAYS_RERAISE, 0.6)[0].category is C.ALL_IN


# --- All-in ------------------------------------------------------------------------------------


def test_after_a_hero_all_in_the_opponent_cannot_reraise() -> None:
    _, diag = decide(PREFLOP_OPEN, [0, 0, 0, 0, 1], 0.5)  # stub says "all-in" for every state
    shove = diag.opponent_probabilities["all_in"]
    assert shove[2:] == [0.0, 0.0, 0.0]  # masked: no raise possible after an all-in
    assert sum(shove) == pytest.approx(1.0)
    assert diag.hypothetical_obs["all_in"]["obs_legal_raise_pot"] is False


def test_facing_an_all_in_only_fold_or_call_and_no_model_call() -> None:
    obs = spot(FACING_ALL_IN)
    assert set(obs.legal_categories) == {C.FOLD, C.CHECK_CALL}
    agent = agent_with(ALWAYS_CALL, 0.55)
    decision = agent.act(obs, rng=None)
    diag = agent.diagnostics[-1]
    assert agent.response_model.calls == 0  # no bet to evaluate, so no prediction is needed
    assert diag.opponent_probabilities == {} and diag.coverage == {}
    break_even = obs.to_call / (obs.pot_size + obs.to_call)
    assert decision.category is (C.CHECK_CALL if 0.55 > break_even else C.FOLD)


def test_uncalled_chips_are_returned_in_the_agent_ev_path() -> None:
    # Equal 100-big-blind stacks mean a shove can never exceed what the opponent can call
    # in real play, so the case is built by hand: the opponent has only 40 chips behind.
    import dataclasses
    from hybridpoker.ev import ResponseProbabilities, action_value
    obs = dataclasses.replace(spot(FLOP_CHECK), opponent_stack=40)
    shove = obs.legal(C.ALL_IN)  # adds 198 chips
    value = action_value(obs, shove, 0.5, ResponseProbabilities(0.0, 1.0, 0.0))
    # Only 40 can be called: hero risks 40, showdown pot 4 + 40 + 40 = 84 -> 0.5 x 84 - 40 = 2.
    assert value.ev == pytest.approx(2.0)


# --- Same rule for both agents -------------------------------------------------------------


class FrequencyViaPredictor:
    """A stand-in 'ML predictor' that returns exactly the frequency model's probabilities."""

    def __init__(self, frequency: FrequencyResponseModel, priors) -> None:
        self.frequency = frequency
        self.priors = priors

    def predict(self, observations, history):
        return self.frequency.predict(observations, None)


def test_identical_probabilities_give_identical_decisions(tiny_dataset) -> None:
    frequency = FrequencyResponseModel.load()
    agent_f = EVAgent("ev_frequency", frequency, equity_samples=60, coverage=load_coverage())
    agent_m = EVAgent("ev_ml", MLResponseModel(FrequencyViaPredictor(frequency, tiny_dataset.priors)),
                      equity_samples=60, coverage=load_coverage())
    match_f = run_match(agent_f, make_agent("maniac", 30), deal_count=6, seed=404)
    match_m = run_match(agent_m, make_agent("maniac", 30), deal_count=6, seed=404)
    actions_f = [(r.action, r.decision.reason) for h in match_f.hands for r in h.result.decisions]
    actions_m = [(r.action, r.decision.reason) for h in match_m.hands for r in h.result.decisions]
    assert actions_f == actions_m
    assert [h.result.payoffs for h in match_f.hands] == [h.result.payoffs for h in match_m.hands]
    assert [d.action_evs for d in agent_f.diagnostics] == [d.action_evs for d in agent_m.diagnostics]
    assert agent_m.tracker is not None and agent_f.tracker is None  # only the ML agent tracks history


def test_equity_is_computed_exactly_like_the_probability_agent() -> None:
    import random
    obs = spot(FLOP_CHECK)
    probability = ProbabilityAgent()
    ev = EVAgent("ev", StubModel(ALWAYS_CALL), coverage=load_coverage())
    for seed in (1, 2, 3):
        p = probability.act(obs, random.Random(seed)).equity
        e = ev.act(obs, random.Random(seed)).equity
        assert p == e  # same function, same 500 samples, same random stream


# --- Real matches: hypothetical states, history, diagnostics, determinism -----------------------


@pytest.fixture(scope="module")
def ml_match(tiny_bundle):
    agent = EVAgent("ev_ml", MLResponseModel(OpponentPredictor(tiny_bundle)), equity_samples=60,
                    coverage=load_coverage())
    match = run_match(agent, make_agent("maniac", 30), deal_count=8, seed=505)
    return agent, match


def test_hypothetical_state_matches_what_the_opponent_then_saw(ml_match) -> None:
    agent, match = ml_match
    hero_records = [(h, i) for h in match.hands for i, r in enumerate(h.result.decisions) if r.seat == h.hero_seat]
    assert len(hero_records) == len(agent.diagnostics)
    checked = 0
    for (hand, i), diag in zip(hero_records, agent.diagnostics):
        records = hand.result.decisions
        if diag.chosen in ("raise_half_pot", "raise_pot", "all_in") and i + 1 < len(records):
            nxt = records[i + 1]
            assert diag.hypothetical_obs[diag.chosen] == observation_columns(nxt.observation)
            checked += 1
    assert checked > 0


def test_ml_agent_history_equals_offline_training_features(ml_match, tiny_bundle) -> None:
    agent, match = ml_match
    frame = decisions_frame(match)
    offline = history_features(frame, tiny_bundle["priors"])
    villain_rows = frame[frame["label_actor_role"] == "villain"]
    hero_rows = frame[frame["label_actor_role"] == "hero"]
    checked = 0
    for (_, row), diag in zip(hero_rows.iterrows(), agent.diagnostics):
        same_deal = villain_rows[villain_rows["id_deal_index"] == row["id_deal_index"]]
        if same_deal.empty:
            continue
        expected_hands = int(round(np.expm1(offline.loc[same_deal.index[0], "b_log_prior_hands"])))
        assert diag.opponent_history_hands == expected_hands
        checked += 1
    assert checked > 5


def test_every_decision_records_complete_diagnostics(ml_match) -> None:
    agent, match = ml_match
    hero_obs = [r.observation for h in match.hands for r in h.result.decisions if r.seat == h.hero_seat]
    for obs, diag in zip(hero_obs, agent.diagnostics):
        assert 0.0 <= diag.equity <= 1.0
        assert set(diag.action_evs) == {c.value for c in obs.legal_categories}
        raises = {c.value for c in obs.legal_categories} & {"raise_half_pot", "raise_pot", "all_in"}
        assert set(diag.opponent_probabilities) == raises == set(diag.coverage) == set(diag.opponent_raw_probabilities)
        for name in raises:
            assert sum(diag.opponent_probabilities[name]) == pytest.approx(1.0)
            assert {"key", "training_rows", "low"} <= set(diag.coverage[name])
        assert diag.total_seconds >= diag.equity_seconds >= 0 and diag.model_seconds >= 0
        assert diag.chosen in diag.action_evs and diag.reason


def test_ev_agent_matches_are_legal_and_byte_identical(tmp_path) -> None:
    hashes = []
    for run in range(2):
        agent = EVAgent("ev_frequency", FrequencyResponseModel.load(), equity_samples=60, coverage=load_coverage())
        villain = make_agent("solid", 30)
        match = run_match(agent, villain, deal_count=6, seed=606)
        paths = write_run(match, agent, villain, tmp_path / str(run))
        hashes.append(hashlib.sha256(paths["decisions"].read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


# --- Artifacts and models --------------------------------------------------------------------


def test_frequency_prior_artifact() -> None:
    data = json.loads(Path("results/phase3/frequency_prior.json").read_text())
    assert data["class_order"] == list(CLASS_ORDER) and data["split"] == "train" and data["rows"] == 23603
    assert sum(data["shares"]) == pytest.approx(1.0)
    assert sum(data["counts"].values()) == data["rows"]
    assert "Not conditioned on the betting situation" in data["description"]
    with pytest.raises(ValueError):
        FrequencyResponseModel([0.5, 0.5])
    with pytest.raises(ValueError):
        FrequencyResponseModel([0.5, 0.6, 0, 0, 0])


def test_coverage_artifact_and_key(tiny_dataset) -> None:
    coverage = load_coverage(COVERAGE_PATH)
    assert sum(coverage["counts"].values()) == coverage["rows"] == 23603
    assert coverage["low_coverage_threshold"] == 30
    frame = tiny_dataset.frame
    counts = coverage_counts(frame)
    assert sum(counts.values()) == len(frame)
    row = frame.iloc[0]
    assert coverage_key(row) == f"{row['obs_street']}|facing_all_in=0|raises={min(int(row['obs_raises_this_street']), 3)}"


def test_ml_model_requires_a_tracker(tiny_bundle) -> None:
    with pytest.raises(ValueError):
        MLResponseModel(OpponentPredictor(tiny_bundle)).predict([spot(PREFLOP_OPEN)], None)


@pytest.mark.skipif(not PHASE4_MODEL_PATH.exists(), reason="frozen model file is kept outside the repository")
def test_real_frozen_model_loads_with_hash_and_plays() -> None:
    agent = make_ml_ev_agent(equity_samples=60)
    assert agent.response_model.config()["model_sha256"] == FROZEN_MODEL_SHA256
    run_match(agent, make_agent("solid", 30), deal_count=1, seed=7)
    assert agent.diagnostics
