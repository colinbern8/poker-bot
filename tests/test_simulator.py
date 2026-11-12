"""Tests for hybridpoker.simulator.

Rigged decks are written in dealing order: seat 0, seat 1, seat 0, seat 1
(hole cards), then burn, flop x3, burn, turn, burn, river.
"""

import dataclasses
import random
from collections.abc import Callable

import pytest
from pokerkit import Card

from hybridpoker.actions import ActionCategory
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.game import BIG_BLIND, BIG_BLIND_SEAT, BUTTON_SEAT, SMALL_BLIND, STARTING_STACK
from hybridpoker.observation import Observation
from hybridpoker.simulator import HandResult, IllegalActionError, play_hand, shuffled_deck

C = ActionCategory


class FunctionAgent(Agent):
    """Test agent whose choice is a function of the observation."""

    def __init__(self, name: str, choose: Callable[[Observation, random.Random], ActionCategory]):
        super().__init__(name)
        self.choose = choose

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        return Decision(self.choose(observation, rng))


def always(category: ActionCategory) -> Callable[[Observation, random.Random], ActionCategory]:
    return lambda observation, rng: category


def random_legal(observation: Observation, rng: random.Random) -> ActionCategory:
    return rng.choice(observation.legal_categories)


def rigged_deck(hole_seat0: str, hole_seat1: str, board: str) -> list[Card]:
    a, b = list(Card.parse(hole_seat0)), list(Card.parse(hole_seat1))
    flop_turn_river = list(Card.parse(board))
    burns = list(Card.parse("2c3c4c"))
    used = set(a + b + flop_turn_river + burns)
    assert len(used) == 4 + 5 + 3, "rigged cards must be distinct"
    order = [a[0], b[0], a[1], b[1], burns[0], *flop_turn_river[:3], burns[1],
             flop_turn_river[3], burns[2], flop_turn_river[4]]
    rest = [card for card in shuffled_deck(0) if card not in used]
    return order + rest


def checkers() -> list[Agent]:
    return [FunctionAgent("bb", always(C.CHECK_CALL)), FunctionAgent("btn", always(C.CHECK_CALL))]


def cards_inside(value: object) -> list[Card]:
    """Recursively find Card objects in dataclasses, tuples and lists."""
    if isinstance(value, Card):
        return [value]
    if dataclasses.is_dataclass(value):
        return [c for f in dataclasses.fields(value) for c in cards_inside(getattr(value, f.name))]
    if isinstance(value, (tuple, list)):
        return [c for item in value for c in cards_inside(item)]
    return []


def contributions(hand: HandResult) -> list[int]:
    """Total chips each seat put in (blinds plus every action)."""
    total = [BIG_BLIND, SMALL_BLIND]
    for record in hand.decisions:
        total[record.seat] += record.action.chips_added
    return total


# --- Showdowns and payouts --------------------------------------------------------


def test_showdown_pays_the_better_hand() -> None:
    deck = rigged_deck("AsAh", "7d2h", "KcQdJh8s5d")
    hand = play_hand(checkers(), deal_seed=0, agent_seeds=(1, 2), deck=deck)

    assert hand.went_to_showdown
    assert hand.board_cards == tuple(Card.parse("KcQdJh8s5d"))
    assert hand.hole_cards == (tuple(Card.parse("AsAh")), tuple(Card.parse("7d2h")))
    assert hand.payoffs == (BIG_BLIND, -BIG_BLIND)  # pot of 4: each put in 2


def test_split_pot_returns_chips() -> None:
    deck = rigged_deck("7d2h", "8d3h", "AhKhQhJhTh")  # royal flush on board
    hand = play_hand(checkers(), deal_seed=0, agent_seeds=(1, 2), deck=deck)
    assert hand.went_to_showdown
    assert hand.payoffs == (0, 0)


def test_button_fold_preflop_loses_small_blind() -> None:
    agents = [FunctionAgent("bb", always(C.CHECK_CALL)), FunctionAgent("btn", always(C.FOLD))]
    hand = play_hand(agents, deal_seed=5, agent_seeds=(1, 2))

    assert not hand.went_to_showdown
    assert len(hand.decisions) == 1
    assert hand.payoffs == (SMALL_BLIND, -SMALL_BLIND)


def test_preflop_all_in_runs_out_the_board() -> None:
    deck = rigged_deck("KsKh", "QsQh", "9c8d5h4s2d")
    agents = [FunctionAgent("bb", always(C.CHECK_CALL)), FunctionAgent("btn", always(C.ALL_IN))]
    hand = play_hand(agents, deal_seed=0, agent_seeds=(1, 2), deck=deck)

    assert [r.action.category for r in hand.decisions] == [C.ALL_IN, C.CHECK_CALL]
    assert hand.decisions[1].action.is_all_in  # the call is all-in too
    assert len(hand.board_cards) == 5  # dealt with no further decisions
    assert hand.payoffs == (STARTING_STACK, -STARTING_STACK)  # kings beat queens


def test_uncalled_raise_is_returned() -> None:
    # The button raises; the big blind folds. The button only wins the blind.
    agents = [FunctionAgent("bb", always(C.FOLD)), FunctionAgent("btn", always(C.RAISE_POT))]
    hand = play_hand(agents, deal_seed=3, agent_seeds=(1, 2))
    assert hand.payoffs == (-BIG_BLIND, BIG_BLIND)


# --- Rules hold across many random hands -----------------------------------------


@pytest.fixture(scope="module")
def random_hands() -> list[HandResult]:
    agents = [FunctionAgent("r0", random_legal), FunctionAgent("r1", random_legal)]
    return [play_hand(agents, deal_seed=i, agent_seeds=(i, i + 10_000)) for i in range(400)]


def test_chips_are_conserved_and_bounded(random_hands: list[HandResult]) -> None:
    for hand in random_hands:
        assert sum(hand.payoffs) == 0
        assert all(abs(p) <= STARTING_STACK for p in hand.payoffs)


def test_payouts_match_chips_put_in(random_hands: list[HandResult]) -> None:
    for hand in random_hands:
        put_in = contributions(hand)
        if not hand.went_to_showdown:
            folder = hand.decisions[-1].seat
            assert hand.decisions[-1].action.category is C.FOLD
            assert hand.payoffs[folder] == -put_in[folder]
        else:
            # Showdown: equal amounts were matched; the loser loses all of it.
            assert put_in[0] == put_in[1]
            assert hand.payoffs in ((put_in[1], -put_in[0]), (-put_in[0], put_in[1]), (0, 0))


def test_every_action_was_legal_and_consistent(random_hands: list[HandResult]) -> None:
    for hand in random_hands:
        for record in hand.decisions:
            observation = record.observation
            assert record.action in observation.legal_actions
            assert record.action.chips_added <= observation.own_stack
            if record.action.category is C.FOLD:
                assert observation.to_call > 0


def test_board_size_matches_how_the_hand_ended(random_hands: list[HandResult]) -> None:
    for hand in random_hands:
        assert len(hand.board_cards) in (0, 3, 4, 5)
        if hand.went_to_showdown:
            assert len(hand.board_cards) == 5


def test_history_grows_by_one_public_action_per_decision(random_hands: list[HandResult]) -> None:
    for hand in random_hands:
        for index, record in enumerate(hand.decisions):
            assert len(record.observation.history) == index
            assert record.decision_index == index
        first = hand.decisions[0]
        assert first.seat == BUTTON_SEAT and first.observation.street == "preflop"


def test_observations_never_reveal_opponent_cards(random_hands: list[HandResult]) -> None:
    for hand in random_hands:
        for record in hand.decisions:
            own = hand.hole_cards[record.seat]
            opponent = hand.hole_cards[1 - record.seat]
            assert record.observation.hole_cards == own
            # Collect every Card object anywhere inside the observation.
            visible = set(cards_inside(record.observation))
            assert visible == set(own) | set(record.observation.board_cards)
            assert not visible & set(opponent)


# --- Reproducibility ------------------------------------------------------------------


def test_same_seeds_replay_the_same_hand() -> None:
    agents = [FunctionAgent("r0", random_legal), FunctionAgent("r1", random_legal)]
    first = play_hand(agents, deal_seed=99, agent_seeds=(5, 6))
    second = play_hand(agents, deal_seed=99, agent_seeds=(5, 6))
    assert first == second


def test_same_deal_seed_gives_same_cards_to_each_seat() -> None:
    a = play_hand(checkers(), deal_seed=7, agent_seeds=(1, 2))
    b = play_hand(
        [FunctionAgent("x", random_legal), FunctionAgent("y", random_legal)],
        deal_seed=7,
        agent_seeds=(3, 4),
    )
    assert a.hole_cards == b.hole_cards
    # The checkers always reach the river; the random hand may stop earlier.
    assert len(a.board_cards) == 5
    assert b.board_cards == a.board_cards[: len(b.board_cards)]


def test_different_deal_seeds_give_different_cards() -> None:
    assert shuffled_deck(1) != shuffled_deck(2)
    assert sorted(map(repr, shuffled_deck(1))) == sorted(map(repr, shuffled_deck(2)))


# --- Errors -------------------------------------------------------------------------------


def test_illegal_action_is_rejected() -> None:
    # Folding when checking is free is not offered, so it is illegal.
    agents = [FunctionAgent("bb", always(C.FOLD)), FunctionAgent("btn", always(C.CHECK_CALL))]
    with pytest.raises(IllegalActionError, match="fold"):
        play_hand(agents, deal_seed=1, agent_seeds=(1, 2))


def test_two_agents_required() -> None:
    with pytest.raises(ValueError):
        play_hand(checkers()[:1], deal_seed=1, agent_seeds=(1,))


def test_big_blind_is_seat_zero() -> None:
    hand = play_hand(checkers(), deal_seed=1, agent_seeds=(1, 2))
    assert hand.decisions[1].seat == BIG_BLIND_SEAT
    assert hand.decisions[1].observation.position == "big_blind"
