"""Hypothetical ("what if I bet?") observations for the opponent.

The EV agents need the opponent's predicted response to a bet or raise the
hero is only *considering*. The opponent model was trained on observed
states, so we build the state the opponent would face after the hero's
action and describe it exactly like a logged decision.

The state is rebuilt **from public information only**: the hero's
observation supplies the public betting history and board, the cards are
replayed in a fresh PokerKit game with **fake hole cards**, and the
resulting opponent observation has its hole cards removed. So the
hypothetical state cannot contain any private information, by
construction. :func:`replay_public_state` also checks that the rebuilt
state matches the hero's real observation (pot, stacks, bets, legal
actions) before it is used.

Limitation (stated in the report): these hypothetical states are produced
by the hero's *candidate* actions, which can differ from the actions seen
in training data (distribution shift).
"""

from __future__ import annotations

import dataclasses

from pokerkit import Deck, State

from hybridpoker.actions import LegalAction, apply_action
from hybridpoker.game import SEAT_POSITIONS, STREET_NAMES, create_heads_up_state
from hybridpoker.observation import Observation, PublicAction, observe


class ReplayMismatch(RuntimeError):
    """The public replay does not reproduce the hero's observation."""


PUBLIC_FIELDS = ("street", "position", "pot_size", "to_call", "own_stack", "opponent_stack", "effective_stack",
                 "own_bet", "opponent_bet", "legal_actions", "board_cards")


def replay_public_state(observation: Observation) -> State:
    """A PokerKit state at the hero's decision, rebuilt from public information only."""
    board = list(observation.board_cards)
    fake_cards = iter([card for card in Deck.STANDARD if card not in set(board)])
    board_cards = iter(board)
    state = create_heads_up_state()

    def deal_pending() -> None:
        while True:
            if state.can_deal_hole():
                state.deal_hole(next(fake_cards))
            elif state.can_burn_card():
                state.burn_card(next(fake_cards))
            elif state.can_deal_board():
                state.deal_board(next(board_cards))
            else:
                return

    for action in observation.history:
        deal_pending()
        legal = LegalAction(action.category, action.bet_to, action.chips_added, action.is_all_in)
        apply_action(state, legal)
    deal_pending()

    rebuilt = observe(state, observation.history)
    for field in PUBLIC_FIELDS:
        if getattr(rebuilt, field) != getattr(observation, field):
            raise ReplayMismatch(f"Public replay differs in {field!r}.")
    return state


def hypothetical_opponent_observation(observation: Observation, action: LegalAction) -> Observation | None:
    """The opponent's observation if the hero took ``action`` now.

    Returns None when the action ends the betting round or the hand (e.g. a
    call that closes the street, or a fold), because the opponent then has
    no decision to predict on this street.
    """
    if observation.legal(action.category) != action:
        raise ValueError(f"{action} is not one of the hero's legal actions.")
    state = replay_public_state(observation)
    hero_seat = state.actor_index
    street = STREET_NAMES[state.street_index]
    apply_action(state, action)
    if not state.status or state.actor_index is None or state.actor_index == hero_seat:
        return None
    if STREET_NAMES[state.street_index] != street:
        return None
    history = (*observation.history, PublicAction(street, SEAT_POSITIONS[hero_seat], action.category,
                                                  action.bet_to, action.chips_added, action.is_all_in))
    opponent_view = observe(state, history)
    return dataclasses.replace(opponent_view, hole_cards=())  # fake cards removed: nothing private remains
