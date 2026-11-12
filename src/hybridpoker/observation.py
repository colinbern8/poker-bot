"""What a player is allowed to see when making a decision.

An :class:`Observation` contains only information the acting player would
have at a real table: their own hole cards, the community cards, chip
amounts, legal actions, and the public betting history. The opponent's hole
cards are never included (CLAUDE.md rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerkit import Card, State

from hybridpoker.actions import ActionCategory, LegalAction, legal_actions
from hybridpoker.game import BIG_BLIND, SEAT_POSITIONS, STREET_NAMES, other_seat


@dataclass(frozen=True)
class PublicAction:
    """An action in the betting history, as every player sees it."""

    street: str
    position: str
    category: ActionCategory
    bet_to: int
    chips_added: int
    is_all_in: bool

    def short(self) -> str:
        """Compact text form, e.g. ``preflop:button:raise_pot:6``."""
        return f"{self.street}:{self.position}:{self.category.value}:{self.bet_to}"


@dataclass(frozen=True)
class Observation:
    """The acting player's view of the game just before a decision.

    Chip conventions (same as Phase 1):

    * ``pot_size``: all chips in the middle, including bets on this street,
      excluding the chips the player would add by calling.
    * ``to_call``: extra chips needed to call (0 means checking is free).
    * Stacks are chips *behind*, i.e. not yet put into the pot.
    * ``effective_stack``: the smaller of the two stacks behind. No player
      can win or lose more than this from now on.
    """

    hole_cards: tuple[Card, ...]
    board_cards: tuple[Card, ...]
    street: str
    position: str
    pot_size: int
    to_call: int
    own_stack: int
    opponent_stack: int
    effective_stack: int
    own_bet: int
    opponent_bet: int
    big_blind: int
    legal_actions: tuple[LegalAction, ...]
    history: tuple[PublicAction, ...]

    def legal(self, category: ActionCategory) -> LegalAction | None:
        """Return the legal action for ``category``, or None if illegal."""
        for action in self.legal_actions:
            if action.category is category:
                return action
        return None

    @property
    def legal_categories(self) -> tuple[ActionCategory, ...]:
        return tuple(action.category for action in self.legal_actions)

    @property
    def can_raise(self) -> bool:
        return self.legal(ActionCategory.ALL_IN) is not None


@dataclass(frozen=True)
class PublicHandSummary:
    """What every player publicly saw in a finished hand (Phase 4).

    Sent to agents after each hand so they can remember an opponent's
    behaviour across hands. It holds **public information only**: the full
    betting history (including actions after the receiving player's last
    decision, e.g. the opponent folding to a bet), the community cards that
    were dealt, the receiving player's position, and match-structure
    indices. It never contains hole cards, showdown cards or payoffs,
    matching the information used to train the opponent model.

    Attributes:
        position: The receiving player's position in this hand.
        history: Every action of the hand, in order.
        board_cards: Community cards dealt (empty if the hand ended preflop).
        deal_index: Which deal of the match this hand belongs to.
        seat_swap: 0 or 1, the seating of this deal (always 0 without swaps).
        hands_per_deal: 2 for seat-swapped matches, otherwise 1.
    """

    position: str
    history: tuple[PublicAction, ...]
    board_cards: tuple[Card, ...]
    deal_index: int
    seat_swap: int
    hands_per_deal: int


def board_cards_of(state: State) -> tuple[Card, ...]:
    """Flatten PokerKit's board representation for a single board."""
    return tuple(cards[0] for cards in state.board_cards)


def observe(state: State, history: tuple[PublicAction, ...]) -> Observation:
    """Build the observation for the player due to act.

    Only ``state.hole_cards[seat]`` is read for hole cards, so the
    opponent's cards cannot leak into the observation.
    """
    seat = state.actor_index
    if seat is None:
        raise ValueError("No player is due to act.")
    opponent = other_seat(seat)

    own_stack = state.stacks[seat]
    opponent_stack = state.stacks[opponent]
    return Observation(
        hole_cards=tuple(state.hole_cards[seat]),
        board_cards=board_cards_of(state),
        street=STREET_NAMES[state.street_index],
        position=SEAT_POSITIONS[seat],
        pot_size=state.total_pot_amount,
        to_call=state.checking_or_calling_amount,
        own_stack=own_stack,
        opponent_stack=opponent_stack,
        effective_stack=min(own_stack, opponent_stack),
        own_bet=state.bets[seat],
        opponent_bet=state.bets[opponent],
        big_blind=BIG_BLIND,
        legal_actions=legal_actions(state),
        history=history,
    )
