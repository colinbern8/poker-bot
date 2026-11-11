"""Phase 0 smoke tests.

These tests check that the development environment is set up correctly:
the right Python version, all required libraries importable, and PokerKit
able to run the start of a heads-up No-Limit Texas Hold'em hand.

They do NOT test any poker AI logic (that starts in later phases).
"""

import importlib
import sys

import pytest
from pokerkit import Automation, NoLimitTexasHoldem, State

# Game settings used by every PokerKit test below.
SMALL_BLIND = 1
BIG_BLIND = 2
STARTING_STACK = 200
PLAYER_COUNT = 2

# In heads-up play, PokerKit seats the big blind at index 0 and the
# small blind (who is also the button) at index 1.
BIG_BLIND_PLAYER = 0
SMALL_BLIND_PLAYER = 1


def create_heads_up_state() -> State:
    """Create a new heads-up No-Limit Hold'em hand with blinds posted.

    Automations are routine steps PokerKit performs for us (posting
    blinds, collecting bets, etc.). Dealing cards and player decisions
    are deliberately NOT automated, so tests control them explicitly.
    """
    return NoLimitTexasHoldem.create_state(
        automations=(
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        ante_trimming_status=True,
        raw_antes=0,
        raw_blinds_or_straddles=(SMALL_BLIND, BIG_BLIND),
        min_bet=BIG_BLIND,
        raw_starting_stacks=STARTING_STACK,
        player_count=PLAYER_COUNT,
    )


def deal_all_hole_cards(state: State) -> None:
    """Deal hole cards until PokerKit says no more are needed."""
    while state.can_deal_hole():
        state.deal_hole()


# --- Environment checks ---------------------------------------------------


def test_python_version_is_3_12() -> None:
    assert sys.version_info[:2] == (3, 12), (
        f"Expected Python 3.12, got {sys.version.split()[0]}. "
        "See README.md for how to use Python 3.12."
    )


@pytest.mark.parametrize(
    "module_name",
    ["pokerkit", "numpy", "pandas", "sklearn", "streamlit", "hybridpoker"],
)
def test_required_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_hybridpoker_has_version() -> None:
    import hybridpoker

    assert hybridpoker.__version__ == "0.0.1"


# --- PokerKit game initialization -----------------------------------------


def test_heads_up_game_initializes_with_blinds_posted() -> None:
    state = create_heads_up_state()

    assert state.status is True  # the hand is in progress
    assert state.player_count == PLAYER_COUNT
    assert state.street_index == 0  # preflop
    assert list(state.bets) == [BIG_BLIND, SMALL_BLIND]
    assert list(state.stacks) == [
        STARTING_STACK - BIG_BLIND,
        STARTING_STACK - SMALL_BLIND,
    ]
    assert state.total_pot_amount == SMALL_BLIND + BIG_BLIND


def test_each_player_receives_two_unique_hole_cards() -> None:
    state = create_heads_up_state()
    deal_all_hole_cards(state)

    assert [len(cards) for cards in state.hole_cards] == [2, 2]
    all_cards = [card for cards in state.hole_cards for card in cards]
    assert len(set(all_cards)) == 4  # no card dealt twice


# --- Legal actions --------------------------------------------------------


def test_preflop_legal_actions_for_small_blind() -> None:
    state = create_heads_up_state()
    deal_all_hole_cards(state)

    # Heads-up, the small blind (button) acts first before the flop.
    assert state.actor_index == SMALL_BLIND_PLAYER
    assert state.can_fold()
    assert state.can_check_or_call()
    assert state.checking_or_calling_amount == BIG_BLIND - SMALL_BLIND

    # Minimum raise is to 2x the big blind; maximum is all-in.
    assert state.min_completion_betting_or_raising_to_amount == 2 * BIG_BLIND
    assert state.max_completion_betting_or_raising_to_amount == STARTING_STACK
    assert state.can_complete_bet_or_raise_to(2 * BIG_BLIND)
    assert not state.can_complete_bet_or_raise_to(SMALL_BLIND)  # too small


def test_small_blind_call_then_big_blind_check_ends_preflop() -> None:
    state = create_heads_up_state()
    deal_all_hole_cards(state)

    state.check_or_call()  # small blind completes to the big blind
    assert list(state.stacks) == [
        STARTING_STACK - BIG_BLIND,
        STARTING_STACK - BIG_BLIND,
    ]
    assert state.actor_index == BIG_BLIND_PLAYER

    state.check_or_call()  # big blind checks
    assert state.can_deal_board()  # betting is over; the flop is next
    assert state.total_pot_amount == 2 * BIG_BLIND


def test_small_blind_fold_awards_pot_to_big_blind() -> None:
    state = create_heads_up_state()
    deal_all_hole_cards(state)

    state.fold()

    assert state.status is False  # the hand is over
    assert list(state.stacks) == [
        STARTING_STACK + SMALL_BLIND,
        STARTING_STACK - SMALL_BLIND,
    ]
