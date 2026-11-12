"""Play one heads-up hand between two agents using PokerKit.

PokerKit enforces the rules (blinds, betting order, minimum raises,
all-ins, side effects of each action, and showdown payouts). This module
adds what PokerKit does not provide:

* a deck shuffled from a seed, so every hand can be replayed exactly;
* the five abstract action categories (see ``actions.py``);
* a record of the game state before every decision.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from pokerkit import Card, Deck

from hybridpoker.actions import ActionCategory, LegalAction, apply_action
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.game import PLAYER_COUNT, SEAT_POSITIONS, STREET_NAMES, create_heads_up_state
from hybridpoker.observation import Observation, PublicAction, PublicHandSummary, observe

_STANDARD_DECK: tuple[Card, ...] = tuple(Deck.STANDARD)
_MAX_STEPS = 1000  # safety net; a heads-up hand needs far fewer


class IllegalActionError(RuntimeError):
    """Raised when an agent chooses an action category that is not legal."""


@dataclass(frozen=True)
class DecisionRecord:
    """The game state just before a decision, and the decision itself."""

    decision_index: int
    seat: int
    agent_name: str
    observation: Observation
    action: LegalAction
    decision: Decision


@dataclass(frozen=True)
class HandResult:
    """Everything that happened in one hand.

    ``payoffs`` are net chips won (+) or lost (-) per seat; they sum to 0
    because there is no rake.
    """

    deal_seed: int
    agent_names: tuple[str, str]
    hole_cards: tuple[tuple[Card, ...], tuple[Card, ...]]
    board_cards: tuple[Card, ...]
    payoffs: tuple[int, int]
    went_to_showdown: bool
    decisions: tuple[DecisionRecord, ...]


def public_history(result: HandResult) -> tuple[PublicAction, ...]:
    """Every action of a finished hand, as all players saw it."""
    return tuple(
        PublicAction(
            street=record.observation.street,
            position=record.observation.position,
            category=record.action.category,
            bet_to=record.action.bet_to,
            chips_added=record.action.chips_added,
            is_all_in=record.action.is_all_in,
        )
        for record in result.decisions
    )


def public_hand_summary(
    result: HandResult, seat: int, deal_index: int, seat_swap: int, hands_per_deal: int
) -> PublicHandSummary:
    """The public summary of a finished hand for the player in ``seat``.

    Hole cards, payoffs and showdown information are deliberately left out.
    """
    return PublicHandSummary(
        position=SEAT_POSITIONS[seat],
        history=public_history(result),
        board_cards=result.board_cards,
        deal_index=deal_index,
        seat_swap=seat_swap,
        hands_per_deal=hands_per_deal,
    )


def shuffled_deck(deal_seed: int) -> list[Card]:
    """Return the 52 cards in a seeded random order."""
    deck = list(_STANDARD_DECK)
    random.Random(deal_seed).shuffle(deck)
    return deck


def play_hand(
    agents: Sequence[Agent],
    deal_seed: int,
    agent_seeds: Sequence[int],
    deck: Sequence[Card] | None = None,
) -> HandResult:
    """Play one complete hand.

    Args:
        agents: The agent in seat 0 (big blind) and seat 1 (button).
        deal_seed: Seed for shuffling the deck. The same seed always deals
            the same cards to the same seats.
        agent_seeds: Seed for each seat's agent random generator.
        deck: Optional fixed card order (used by tests). Cards are dealt
            from the front: hole cards alternate seat 0, seat 1, seat 0,
            seat 1; then burn and board cards in order.
    """
    if len(agents) != PLAYER_COUNT or len(agent_seeds) != PLAYER_COUNT:
        raise ValueError("Exactly two agents and two agent seeds are required.")

    cards = list(deck) if deck is not None else shuffled_deck(deal_seed)
    rngs = [random.Random(seed) for seed in agent_seeds]
    state = create_heads_up_state()
    next_card = 0
    history: list[PublicAction] = []
    records: list[DecisionRecord] = []
    hole_cards: tuple[list[Card], list[Card]] = ([], [])
    board_cards: list[Card] = []
    folded = False

    for _ in range(_MAX_STEPS):
        if not state.status:
            break
        if state.can_deal_hole():
            hole_cards[state.hole_dealee_index].append(cards[next_card])
            state.deal_hole(cards[next_card])
            next_card += 1
        elif state.can_burn_card():
            state.burn_card(cards[next_card])
            next_card += 1
        elif state.can_deal_board():
            board_cards.append(cards[next_card])
            state.deal_board(cards[next_card])
            next_card += 1
        elif state.actor_index is not None:
            seat = state.actor_index
            observation = observe(state, tuple(history))
            decision = agents[seat].act(observation, rngs[seat])
            action = observation.legal(decision.category)
            if action is None:
                raise IllegalActionError(
                    f"{agents[seat].name} chose {decision.category.value!r}; "
                    f"legal: {[c.value for c in observation.legal_categories]}"
                )
            street = STREET_NAMES[state.street_index]
            apply_action(state, action)
            folded = folded or action.category is ActionCategory.FOLD
            records.append(
                DecisionRecord(len(records), seat, agents[seat].name, observation, action, decision)
            )
            history.append(
                PublicAction(
                    street=street,
                    position=SEAT_POSITIONS[seat],
                    category=action.category,
                    bet_to=action.bet_to,
                    chips_added=action.chips_added,
                    is_all_in=action.is_all_in,
                )
            )
        else:
            raise RuntimeError("PokerKit state is waiting for an unsupported operation.")
    else:
        raise RuntimeError("Hand did not finish within the step limit.")

    payoffs = tuple(int(p) for p in state.payoffs)
    if sum(payoffs) != 0:
        raise RuntimeError(f"Chips not conserved: payoffs {payoffs}.")

    return HandResult(
        deal_seed=deal_seed,
        agent_names=(agents[0].name, agents[1].name),
        hole_cards=(tuple(hole_cards[0]), tuple(hole_cards[1])),
        board_cards=tuple(board_cards),
        payoffs=(payoffs[0], payoffs[1]),
        went_to_showdown=not folded,
        decisions=tuple(records),
    )
