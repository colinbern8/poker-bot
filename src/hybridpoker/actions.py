"""Abstract action categories and their conversion to legal chip amounts.

Agents choose from five abstract categories. The simulator converts each
category into a concrete, legal PokerKit action:

=================  ===========================================================
Category           Concrete action
=================  ===========================================================
``fold``           Give up the hand. Only offered when facing a bet.
``check_call``     Check if nothing is owed, otherwise call.
``raise_half_pot`` Bet or raise by half the pot (after calling).
``raise_pot``      Bet or raise by the full pot (after calling).
``all_in``         Bet or raise the maximum allowed amount.
=================  ===========================================================

Raise sizes follow the standard "pot-size raise" rule. First imagine calling;
then add ``fraction`` times the pot that would result::

    raise_to = largest_bet + fraction * (pot_size + to_call)

Here ``pot_size`` includes every bet already made this street (the Phase 1
convention). The result is rounded to whole chips and clamped into PokerKit's
legal range ``[min_raise_to, max_raise_to]``. Clamping can make two categories
produce the same amount, e.g. a pot-size raise that exceeds the stack becomes
an all-in. The category records what the agent *intended*; the chip amounts
record what actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pokerkit import State


class ActionCategory(str, Enum):
    """The five abstract actions available to every agent."""

    FOLD = "fold"
    CHECK_CALL = "check_call"
    RAISE_HALF_POT = "raise_half_pot"
    RAISE_POT = "raise_pot"
    ALL_IN = "all_in"


RAISE_CATEGORIES = (
    ActionCategory.RAISE_HALF_POT,
    ActionCategory.RAISE_POT,
    ActionCategory.ALL_IN,
)
POT_FRACTIONS = {
    ActionCategory.RAISE_HALF_POT: 0.5,
    ActionCategory.RAISE_POT: 1.0,
}


@dataclass(frozen=True)
class LegalAction:
    """A legal action with its concrete chip amounts.

    Attributes:
        category: The abstract action category.
        bet_to: The acting player's total bet on this street after the
            action (PokerKit's "raise to" amount for raises).
        chips_added: Chips moved from the player's stack by this action.
        is_all_in: True if the action puts the player's whole stack in.
    """

    category: ActionCategory
    bet_to: int
    chips_added: int
    is_all_in: bool


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def legal_actions(state: State) -> tuple[LegalAction, ...]:
    """Return the legal abstract actions for the player due to act.

    The tuple is ordered like :class:`ActionCategory`. Folding is excluded
    when checking is free, since it can never be better than checking.
    """
    seat = state.actor_index
    if seat is None:
        raise ValueError("No player is due to act.")

    own_bet = state.bets[seat]
    stack = state.stacks[seat]
    largest_bet = max(state.bets)
    to_call = state.checking_or_calling_amount
    pot_size = state.total_pot_amount

    actions: list[LegalAction] = []

    if to_call > 0 and state.can_fold():
        actions.append(LegalAction(ActionCategory.FOLD, own_bet, 0, False))

    if state.can_check_or_call():
        actions.append(
            LegalAction(
                ActionCategory.CHECK_CALL,
                bet_to=own_bet + to_call,
                chips_added=to_call,
                is_all_in=stack > 0 and to_call == stack,
            )
        )

    if state.can_complete_bet_or_raise_to():
        min_to = state.min_completion_betting_or_raising_to_amount
        max_to = state.max_completion_betting_or_raising_to_amount
        targets = {
            category: largest_bet + fraction * (pot_size + to_call)
            for category, fraction in POT_FRACTIONS.items()
        }
        targets[ActionCategory.ALL_IN] = max_to
        for category in RAISE_CATEGORIES:
            bet_to = min(max(_round_half_up(targets[category]), min_to), max_to)
            chips_added = bet_to - own_bet
            actions.append(
                LegalAction(category, bet_to, chips_added, chips_added == stack)
            )

    return tuple(actions)


def apply_action(state: State, action: LegalAction) -> None:
    """Apply a legal abstract action to a PokerKit state."""
    if action.category is ActionCategory.FOLD:
        state.fold()
    elif action.category is ActionCategory.CHECK_CALL:
        state.check_or_call()
    else:
        state.complete_bet_or_raise_to(action.bet_to)
