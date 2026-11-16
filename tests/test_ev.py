"""Tests for hybridpoker.ev: the Phase 4 EV equations, on hand-computed spots.

Each example states the arithmetic so it can be checked on paper.
"""

import numpy as np
import pytest

from pokerkit import Card

from hybridpoker.actions import ActionCategory
from hybridpoker.ev import (
    ResponseProbabilities,
    action_value,
    ev_bet,
    ev_call,
    ev_check,
    ev_fold,
)
from hybridpoker.game import create_heads_up_state
from hybridpoker.observation import observe
from hybridpoker.pot_odds import call_expected_value

C = ActionCategory
ALWAYS_FOLD = ResponseProbabilities(1.0, 0.0, 0.0)
ALWAYS_CALL = ResponseProbabilities(0.0, 1.0, 0.0)
ALWAYS_RAISE = ResponseProbabilities(0.0, 0.0, 1.0)


# --- Fold and check --------------------------------------------------------------------


def test_fold_is_zero() -> None:
    assert ev_fold() == 0.0


def test_check_is_equity_share_of_pot() -> None:
    assert ev_check(0.6, 20) == pytest.approx(12.0)  # 0.6 x 20
    assert ev_check(0.0, 20) == 0.0
    assert ev_check(1.0, 20) == 20.0


# --- Call -----------------------------------------------------------------------------------


def test_call_matches_hand_example_and_phase1() -> None:
    # Pot 30 (includes the opponent's bet of 10), call 10, equity 0.4:
    # 0.4 x (30 + 10) - 10 = 6.
    assert ev_call(0.4, 30, 10, own_stack=150) == pytest.approx(6.0)
    for equity, pot, call in [(0.25, 30, 10), (0.5, 7, 4), (0.9, 100, 60)]:
        assert ev_call(equity, pot, call, own_stack=500) == pytest.approx(call_expected_value(equity, pot, call))


def test_call_is_zero_at_pot_odds() -> None:
    # Break-even equity = call / (pot + call) = 10 / 40.
    assert ev_call(0.25, 30, 10, own_stack=150) == pytest.approx(0.0)


def test_short_stack_call_returns_the_uncalled_chips() -> None:
    # Opponent bet 50 into a pot that is now 80; hero has only 20 behind.
    # Hero calls 20; 30 of the bet is returned, so the showdown pot is 80 - 30 + 20 = 70.
    # EV = 0.5 x 70 - 20 = 15.
    assert ev_call(0.5, 80, 50, own_stack=20) == pytest.approx(15.0)


def test_call_rejects_free_check_and_bad_equity() -> None:
    with pytest.raises(ValueError):
        ev_call(0.5, 10, 0, own_stack=100)
    with pytest.raises(ValueError):
        ev_check(1.2, 10)


# --- Bet / raise -----------------------------------------------------------------------------


def test_bet_when_opponent_always_folds_wins_the_pot() -> None:
    # Postflop bet of 10 into 20: opponent folds -> hero wins the 20 already in the pot.
    assert ev_bet(0.1, 20, 10, own_bet=0, opponent_bet=0, opponent_stack=190, response=ALWAYS_FOLD) == 20.0


def test_bet_when_opponent_always_calls() -> None:
    # Bet 10 into 20, opponent calls 10: showdown pot 40; EV = 0.6 x 40 - 10 = 14.
    assert ev_bet(0.6, 20, 10, 0, 0, 190, ALWAYS_CALL) == pytest.approx(14.0)
    # With certain win, hero gains the pot plus the opponent's call: 20 + 10 = 30.
    assert ev_bet(1.0, 20, 10, 0, 0, 190, ALWAYS_CALL) == pytest.approx(30.0)
    # With certain loss, hero loses the 10 added.
    assert ev_bet(0.0, 20, 10, 0, 0, 190, ALWAYS_CALL) == pytest.approx(-10.0)


def test_bet_when_opponent_always_reraises_loses_the_bet() -> None:
    assert ev_bet(0.9, 20, 10, 0, 0, 190, ALWAYS_RAISE) == -10.0


def test_bet_mixes_the_three_branches() -> None:
    # Bet 10 into 20, equity 0.5; P(fold) 0.3, P(call) 0.5, P(raise) 0.2:
    # 0.3 x 20 + 0.5 x (0.5 x 40 - 10) + 0.2 x (-10) = 6 + 5 - 2 = 9.
    response = ResponseProbabilities(0.3, 0.5, 0.2)
    assert ev_bet(0.5, 20, 10, 0, 0, 190, response) == pytest.approx(9.0)


def test_preflop_raise_from_the_button() -> None:
    # Button has 1 in, big blind 2, pot 3. Half-pot raise to 4 adds 3 chips;
    # the big blind must add 2 to call. Called pot = 3 + 3 + 2 = 8.
    # Always called with equity 0.55: 0.55 x 8 - 3 = 1.4.
    assert ev_bet(0.55, 3, 3, own_bet=1, opponent_bet=2, opponent_stack=198, response=ALWAYS_CALL) == pytest.approx(1.4)


def test_facing_a_raise_counts_only_new_chips() -> None:
    # Flop pot 10. Hero bet 4, opponent raised to 12: the pot is now 10 + 4 + 12 = 26.
    # Calling adds 8: EV = 0.4 x (26 + 8) - 8 = 5.6. Hero's earlier 4 chips are sunk.
    assert ev_call(0.4, 26, 8, own_stack=186) == pytest.approx(5.6)
    # Re-raising to 36 adds 32 (own bet 4 -> 36); the opponent must add 24 to call.
    # Always called: 0.4 x (26 + 32 + 24) - 32 = 0.8.
    assert ev_bet(0.4, 26, 32, own_bet=4, opponent_bet=12, opponent_stack=183,
                  response=ALWAYS_CALL) == pytest.approx(0.8)


def test_all_in_larger_than_opponent_stack_returns_uncalled_chips() -> None:
    # Hero shoves 100 into a pot of 20; opponent has only 40 behind.
    # Only 40 can be called: hero effectively risks 40, showdown pot 20 + 40 + 40 = 100.
    # EV when called = 0.5 x 100 - 40 = 10 (not 0.5 x 220 - 100).
    assert ev_bet(0.5, 20, 100, 0, 0, opponent_stack=40, response=ALWAYS_CALL) == pytest.approx(10.0)
    # A re-raise is impossible here, but if supplied the loss is the full amount added.
    assert ev_bet(0.5, 20, 100, 0, 0, 40, ALWAYS_RAISE) == -100.0


def test_bet_input_validation() -> None:
    with pytest.raises(ValueError):
        ev_bet(0.5, 20, 0, 0, 0, 100, ALWAYS_CALL)
    with pytest.raises(ValueError):
        ev_bet(0.5, 20, 5, own_bet=0, opponent_bet=10, opponent_stack=100, response=ALWAYS_CALL)  # not a raise


# --- Response probabilities ----------------------------------------------------------------


def test_responses_from_masked_action_probabilities() -> None:
    r = ResponseProbabilities.from_action_probabilities([0.2, 0.5, 0.1, 0.15, 0.05])
    assert (r.fold, r.call) == (0.2, 0.5)
    assert r.raise_ == pytest.approx(0.3)
    with pytest.raises(ValueError):
        ResponseProbabilities(0.5, 0.6, 0.0)  # sums to 1.1
    with pytest.raises(ValueError):
        ResponseProbabilities.from_action_probabilities([0.5, 0.5])
    with pytest.raises(ValueError):
        ResponseProbabilities(-0.1, 0.6, 0.5)


# --- On a real observation -------------------------------------------------------------------


def first_preflop_observation():
    state = create_heads_up_state()
    for card in Card.parse("2c3dAsKh"):
        state.deal_hole(card)
    return observe(state, history=())


def test_action_values_on_the_first_preflop_decision() -> None:
    obs = first_preflop_observation()
    assert (obs.pot_size, obs.to_call, obs.own_bet, obs.opponent_bet) == (3, 1, 1, 2)
    values = {}
    for action in obs.legal_actions:
        response = ResponseProbabilities(0.4, 0.5, 0.1) if action.category not in (C.FOLD, C.CHECK_CALL) else None
        values[action.category] = action_value(obs, action, 0.6, response)
    assert values[C.FOLD].ev == 0.0
    assert values[C.CHECK_CALL].ev == pytest.approx(0.6 * 4 - 1)  # call 1 into 3
    half = next(a for a in obs.legal_actions if a.category is C.RAISE_HALF_POT)
    expected = 0.4 * 3 + 0.5 * (0.6 * (3 + half.chips_added + (half.bet_to - 2)) - half.chips_added) - 0.1 * half.chips_added
    assert values[C.RAISE_HALF_POT].ev == pytest.approx(expected)
    assert "P(fold) 0.40" in values[C.RAISE_HALF_POT].explanation
    with pytest.raises(ValueError):
        action_value(obs, half, 0.6, None)  # bets need a response model


def test_every_legal_action_gets_a_finite_value() -> None:
    obs = first_preflop_observation()
    response = ResponseProbabilities(0.3, 0.6, 0.1)
    for action in obs.legal_actions:
        r = response if action.category not in (C.FOLD, C.CHECK_CALL) else None
        assert np.isfinite(action_value(obs, action, 0.5, r).ev)
