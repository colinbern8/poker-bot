"""Fixed game settings for the Phase 2 heads-up simulator.

Heads-up No-Limit Texas Hold'em, blinds 1/2, no antes, no rake, and
100-big-blind stacks reset at the start of every hand.
"""

from __future__ import annotations

from pokerkit import Automation, Mode, NoLimitTexasHoldem, State

SMALL_BLIND = 1
BIG_BLIND = 2
STARTING_STACK = 100 * BIG_BLIND  # 200 chips
PLAYER_COUNT = 2

# PokerKit seats heads-up players this way: seat 0 posts the big blind,
# seat 1 posts the small blind, is the button, and acts first preflop.
BIG_BLIND_SEAT = 0
BUTTON_SEAT = 1
SEAT_POSITIONS = {BIG_BLIND_SEAT: "big_blind", BUTTON_SEAT: "button"}

STREET_NAMES = ("preflop", "flop", "turn", "river")

# Routine steps PokerKit performs for us. Card dealing and burning are NOT
# automated: the simulator deals from its own seeded deck so every hand can
# be reproduced exactly.
AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


def create_heads_up_state() -> State:
    """Create a new hand with blinds posted and no cards dealt yet."""
    return NoLimitTexasHoldem.create_state(
        automations=AUTOMATIONS,
        ante_trimming_status=True,
        raw_antes=0,
        raw_blinds_or_straddles=(SMALL_BLIND, BIG_BLIND),
        min_bet=BIG_BLIND,
        raw_starting_stacks=STARTING_STACK,
        player_count=PLAYER_COUNT,
        mode=Mode.CASH_GAME,
    )


def other_seat(seat: int) -> int:
    return 1 - seat
