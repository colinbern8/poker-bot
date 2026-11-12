"""Tests for hybridpoker.actions and hybridpoker.observation.

Every scenario uses a real PokerKit state with fixed cards.
"""

import copy

import pytest
from pokerkit import Card, State

from hybridpoker.actions import ActionCategory, LegalAction, apply_action, legal_actions
from hybridpoker.game import BIG_BLIND_SEAT, BUTTON_SEAT, STARTING_STACK, create_heads_up_state
from hybridpoker.observation import observe

FOLD = ActionCategory.FOLD
CHECK_CALL = ActionCategory.CHECK_CALL
HALF = ActionCategory.RAISE_HALF_POT
POT = ActionCategory.RAISE_POT
ALL_IN = ActionCategory.ALL_IN

# Seat 0 receives As Qd, seat 1 receives Ks Jd (dealing alternates seats).
DECK = tuple(Card.parse("AsKsQdJd" "2c" "9c8h7h" "3c" "6s" "4c" "5s"))


def new_hand() -> State:
    state = create_heads_up_state()
    for card in DECK[:4]:
        state.deal_hole(card)
    return state


def deal_flop(state: State) -> None:
    state.burn_card(DECK[4])
    for card in DECK[5:8]:
        state.deal_board(card)


def as_dict(state: State) -> dict[ActionCategory, LegalAction]:
    return {action.category: action for action in legal_actions(state)}


# --- Legal actions and concrete amounts -------------------------------------


def test_button_preflop_facing_big_blind() -> None:
    state = new_hand()
    assert state.actor_index == BUTTON_SEAT
    actions = as_dict(state)

    assert list(actions) == [FOLD, CHECK_CALL, HALF, POT, ALL_IN]
    assert actions[FOLD] == LegalAction(FOLD, bet_to=1, chips_added=0, is_all_in=False)
    assert actions[CHECK_CALL] == LegalAction(CHECK_CALL, 2, 1, False)
    # Pot 3, to call 1: half-pot raise to 2 + 0.5 * 4 = 4; pot raise to 2 + 4 = 6.
    assert actions[HALF] == LegalAction(HALF, 4, 3, False)
    assert actions[POT] == LegalAction(POT, 6, 5, False)
    assert actions[ALL_IN] == LegalAction(ALL_IN, STARTING_STACK, STARTING_STACK - 1, True)


def test_big_blind_option_after_limp_has_no_fold() -> None:
    state = new_hand()
    state.check_or_call()  # button limps
    actions = as_dict(state)

    assert state.actor_index == BIG_BLIND_SEAT
    assert FOLD not in actions  # folding is never better than a free check
    assert actions[CHECK_CALL] == LegalAction(CHECK_CALL, 2, 0, False)
    assert actions[HALF].bet_to == 4  # 2 + 0.5 * 4
    assert actions[POT].bet_to == 6  # 2 + 4


def test_unopened_flop_bet_sizes() -> None:
    state = new_hand()
    state.check_or_call()
    state.check_or_call()
    deal_flop(state)
    actions = as_dict(state)

    assert state.total_pot_amount == 4
    assert actions[CHECK_CALL].chips_added == 0
    assert actions[HALF] == LegalAction(HALF, 2, 2, False)
    assert actions[POT] == LegalAction(POT, 4, 4, False)
    assert actions[ALL_IN] == LegalAction(ALL_IN, 198, 198, True)


def test_raise_sizes_when_facing_a_flop_bet() -> None:
    state = new_hand()
    state.check_or_call()
    state.check_or_call()
    deal_flop(state)
    state.complete_bet_or_raise_to(4)  # big blind bets the pot
    actions = as_dict(state)

    # Pot 8 (4 + bet 4), to call 4: half -> 4 + 0.5 * 12 = 10; pot -> 4 + 12 = 16.
    assert state.min_completion_betting_or_raising_to_amount == 8
    assert actions[CHECK_CALL] == LegalAction(CHECK_CALL, 4, 4, False)
    assert actions[HALF] == LegalAction(HALF, 10, 10, False)
    assert actions[POT] == LegalAction(POT, 16, 16, False)


def test_reraise_sizes_stay_within_the_legal_range() -> None:
    state = new_hand()
    state.check_or_call()
    state.check_or_call()
    deal_flop(state)
    state.complete_bet_or_raise_to(2)  # min bet into pot of 4
    state.complete_bet_or_raise_to(20)  # raise to 20
    actions = as_dict(state)

    # Pot 26, to call 18: half-pot target 20 + 0.5 * 44 = 42. The minimum
    # re-raise is 20 + 18 = 38. Heads-up, a half-pot raise always meets the
    # minimum; clamping matters at the top (see the all-in test below).
    minimum = state.min_completion_betting_or_raising_to_amount
    assert minimum == 38
    assert actions[HALF].bet_to == 42
    assert all(
        minimum <= actions[c].bet_to <= state.max_completion_betting_or_raising_to_amount
        for c in (HALF, POT, ALL_IN)
    )


def test_raises_are_clamped_to_all_in() -> None:
    state = new_hand()
    state.complete_bet_or_raise_to(120)  # button raises big
    actions = as_dict(state)

    # The minimum re-raise (238) exceeds the stack, so every raise is all-in.
    for category in (HALF, POT, ALL_IN):
        assert actions[category] == LegalAction(category, 200, 198, True)


def test_facing_all_in_only_fold_or_call() -> None:
    state = new_hand()
    state.complete_bet_or_raise_to(200)  # button shoves
    actions = as_dict(state)

    assert list(actions) == [FOLD, CHECK_CALL]
    assert actions[CHECK_CALL] == LegalAction(CHECK_CALL, 200, 198, True)


def test_no_actions_when_nobody_is_due_to_act() -> None:
    state = create_heads_up_state()  # waiting for hole cards
    with pytest.raises(ValueError):
        legal_actions(state)


# --- Applying actions ---------------------------------------------------------


@pytest.mark.parametrize("prepare", ["preflop", "facing_flop_bet", "facing_shove"])
def test_every_legal_action_is_accepted_and_moves_the_right_chips(prepare: str) -> None:
    base = new_hand()
    if prepare == "facing_flop_bet":
        base.check_or_call()
        base.check_or_call()
        deal_flop(base)
        base.complete_bet_or_raise_to(4)
    elif prepare == "facing_shove":
        base.complete_bet_or_raise_to(200)

    for action in legal_actions(base):
        state = copy.deepcopy(base)
        seat = state.actor_index
        stack_before = state.stacks[seat]
        pot_before = state.total_pot_amount

        apply_action(state, action)

        if action.category is FOLD:
            assert state.status is False  # heads-up: a fold ends the hand
            continue
        assert stack_before - state.stacks[seat] == action.chips_added
        # Chips added go into the pot (bets are part of total_pot_amount).
        assert state.total_pot_amount == pot_before + action.chips_added
        assert (state.stacks[seat] == 0) == action.is_all_in


# --- Observations ---------------------------------------------------------------


def test_observation_contains_only_own_hole_cards() -> None:
    state = new_hand()
    observation = observe(state, history=())

    assert observation.position == "button"
    assert observation.hole_cards == tuple(Card.parse("KsJd"))
    assert not set(Card.parse("AsQd")) & set(observation.hole_cards)
    assert "As" not in repr(observation) and "Qd" not in repr(observation)


def test_observation_chip_fields_preflop() -> None:
    observation = observe(new_hand(), history=())

    assert observation.street == "preflop"
    assert observation.pot_size == 3
    assert observation.to_call == 1
    assert observation.own_stack == 199
    assert observation.opponent_stack == 198
    assert observation.effective_stack == 198
    assert (observation.own_bet, observation.opponent_bet) == (1, 2)
    assert observation.big_blind == 2
    assert observation.can_raise
    assert observation.legal(FOLD) is not None


def test_observation_on_the_flop() -> None:
    state = new_hand()
    state.check_or_call()
    state.check_or_call()
    deal_flop(state)
    observation = observe(state, history=())

    assert observation.street == "flop"
    assert observation.position == "big_blind"  # big blind acts first postflop
    assert observation.board_cards == tuple(Card.parse("9c8h7h"))
    assert observation.hole_cards == tuple(Card.parse("AsQd"))
    assert observation.legal(FOLD) is None
