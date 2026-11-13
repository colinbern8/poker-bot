"""Track an opponent's public actions during a match and build Feature Set B live.

The tracker belongs to one player (the hero) and records the *other*
player's actions from each end-of-hand ``PublicHandSummary``. To guarantee
that live features equal the training features, it does not re-implement
any counting: it builds a small table in the same format as the training
log and calls the same :func:`hybridpoker.opponent_history.history_features`
with the frozen training priors.

Two details of that format matter:

* **Facing a bet.** The summary's actions record ``bet_to`` (the player's
  total bet on the street afterwards) but not the amount the player faced.
  :func:`decisions_from_history` recovers it by replaying the street's
  bets from the blinds (preflop) or zero (later streets).
* **Deals and the mirror hand.** Training excluded both seatings of the
  current deal. In a seat-swapped match a deal's two hands are played one
  after the other (first seating 0, then 1), so after the summary of
  seating 0 of deal *d* the next hand is still deal *d*, and its mirror
  hand is automatically excluded by ``history_features`` (it only uses
  earlier deals).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hybridpoker.game import BIG_BLIND, BIG_BLIND_SEAT, BUTTON_SEAT, SEAT_POSITIONS, SMALL_BLIND
from hybridpoker.observation import PublicAction, PublicHandSummary
from hybridpoker.opponent_history import B_FEATURES, B_INPUT_COLUMNS, HistoryPriors, history_features

BUTTON = SEAT_POSITIONS[BUTTON_SEAT]
BIG_BLIND_POSITION = SEAT_POSITIONS[BIG_BLIND_SEAT]
LIVE_MATCH_ID = "live"


@dataclass(frozen=True)
class PublicDecision:
    """One action from a public history, with the amount the actor faced."""

    street: str
    position: str
    to_call: int
    category: str


def other_position(position: str) -> str:
    return BIG_BLIND_POSITION if position == BUTTON else BUTTON


def decisions_from_history(history: tuple[PublicAction, ...]) -> list[PublicDecision]:
    """Recover, for every action, how many chips the actor had to call.

    Heads-up: the button posts the small blind and the big blind posts the
    big blind; later streets start with no bets.
    """
    decisions = []
    street = None
    bets: dict[str, int] = {}
    for action in history:
        if action.street != street:
            street = action.street
            bets = {BUTTON: SMALL_BLIND, BIG_BLIND_POSITION: BIG_BLIND} if street == "preflop" else {BUTTON: 0, BIG_BLIND_POSITION: 0}
        to_call = max(bets.values()) - bets[action.position]
        decisions.append(PublicDecision(street, action.position, to_call, action.category.value))
        bets[action.position] = action.bet_to
    return decisions


class OpponentTracker:
    """Remembers one opponent's public actions and returns its Feature Set B history features."""

    def __init__(self, priors: HistoryPriors) -> None:
        self.priors = priors
        self._rows: list[dict] = []
        self._last: tuple[int, int, int] | None = None  # (deal_index, seat_swap, hands_per_deal)
        self._cache: dict[int, pd.Series] = {}

    def observe_hand_end(self, summary: PublicHandSummary) -> None:
        """Record the opponent's actions from a finished hand (called by the hero agent)."""
        opponent = other_position(summary.position)
        for decision in decisions_from_history(summary.history):
            if decision.position != opponent:
                continue
            self._rows.append({
                "id_match": LIVE_MATCH_ID,
                "id_deal_index": summary.deal_index,
                "id_seat_swap": summary.seat_swap,
                # One fixed role for every opponent row: position is set so that the
                # training role rule maps all of them to the same player.
                "obs_position": BUTTON if summary.seat_swap == 0 else BIG_BLIND_POSITION,
                "obs_street": decision.street,
                "obs_to_call": decision.to_call,
                "act_category": decision.category,
            })
        self._last = (summary.deal_index, summary.seat_swap, summary.hands_per_deal)
        self._cache.clear()

    def current_deal_index(self) -> int:
        """The deal of the hand about to be played (0 before any hand has finished)."""
        if self._last is None:
            return 0
        deal, seat_swap, hands_per_deal = self._last
        return deal if hands_per_deal == 2 and seat_swap == 0 else deal + 1

    def observed_rows(self) -> int:
        return len(self._rows)

    def history_features(self) -> pd.Series:
        """The opponent's ``b_`` features for a decision in the current deal."""
        deal = self.current_deal_index()
        if deal not in self._cache:
            current = {"id_match": LIVE_MATCH_ID, "id_deal_index": deal, "id_seat_swap": 0, "obs_position": BUTTON,
                       "obs_street": "preflop", "obs_to_call": 0, "act_category": "check_call"}  # placeholder action
            frame = pd.DataFrame([*self._rows, current], columns=list(B_INPUT_COLUMNS))
            self._cache[deal] = history_features(frame, self.priors).iloc[-1][list(B_FEATURES)].rename(None)
        return self._cache[deal].copy()
