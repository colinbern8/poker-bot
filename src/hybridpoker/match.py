"""Run many hands between a hero agent and a villain agent.

Seat-swapped ("duplicate") evaluation: every deal is played twice from the
same seed, once with the hero on the button and once with the hero in the
big blind. Each seat gets the same cards in both hands, so both agents
play both sides of every deal and much of the card luck cancels out.

All randomness comes from one master ``seed``:

* deal ``i`` uses ``derive_seed(seed, "deal", i)`` for its deck, shared by
  both seatings (paired seeds);
* each agent decision stream uses ``derive_seed(seed, "agent", i, swap, seat)``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from hybridpoker.agents.base import Agent
from hybridpoker.game import BIG_BLIND, BIG_BLIND_SEAT, BUTTON_SEAT
from hybridpoker.metrics import WinRate, win_rate
from hybridpoker.seeding import derive_seed
from hybridpoker.simulator import HandResult, play_hand


@dataclass(frozen=True)
class PlayedHand:
    """One hand of a match and the hero's place in it."""

    hand_index: int
    deal_index: int
    seat_swap: int  # 0: hero on the button; 1: hero in the big blind
    hero_seat: int
    result: HandResult

    @property
    def hero_net(self) -> int:
        return self.result.payoffs[self.hero_seat]


@dataclass(frozen=True)
class MatchResult:
    match_id: str
    hero_name: str
    villain_name: str
    seed: int
    deal_count: int
    seat_swap: bool
    hands: tuple[PlayedHand, ...]

    @property
    def hands_per_deal(self) -> int:
        return 2 if self.seat_swap else 1

    def hero_net_by_deal(self) -> list[int]:
        """Hero's net chips per deal (summed over both seatings if swapped)."""
        totals = [0] * self.deal_count
        for hand in self.hands:
            totals[hand.deal_index] += hand.hero_net
        return totals

    def win_rate(self, confidence: float = 0.95) -> WinRate:
        """Hero bb/100 with a CI that treats each deal (pair) as one unit."""
        return win_rate(self.hero_net_by_deal(), self.hands_per_deal, BIG_BLIND, confidence)

    def naive_win_rate(self, confidence: float = 0.95) -> WinRate:
        """bb/100 treating every hand as independent.

        For seat-swapped matches this ignores the pairing and is reported
        only to show how much the pairing reduces uncertainty. Use
        :meth:`win_rate` for conclusions.
        """
        return win_rate([h.hero_net for h in self.hands], 1, BIG_BLIND, confidence)


def make_match_id(hero: Agent, villain: Agent, deal_count: int, seed: int, seat_swap: bool) -> str:
    """An opaque, deterministic identifier for a match configuration.

    It is derived from both agents' full configurations, the seed, the deal
    count and the seat-swap setting, so different configurations get
    different ids. It deliberately does not spell out agent names, but it
    is still an identifier, never a model feature.
    """
    description = json.dumps(
        {
            "hero": hero.config(),
            "villain": villain.config(),
            "deal_count": deal_count,
            "seed": seed,
            "seat_swap": seat_swap,
        },
        sort_keys=True,
    )
    return f"m{derive_seed('match', description):016x}"


def run_match(
    hero: Agent,
    villain: Agent,
    deal_count: int,
    seed: int,
    seat_swap: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> MatchResult:
    """Play ``deal_count`` deals (two hands each if ``seat_swap``)."""
    if deal_count < 1:
        raise ValueError("deal_count must be at least 1.")

    hands: list[PlayedHand] = []
    swaps = (0, 1) if seat_swap else (0,)
    for deal_index in range(deal_count):
        deal_seed = derive_seed(seed, "deal", deal_index)
        for swap in swaps:
            hero_seat = BUTTON_SEAT if swap == 0 else BIG_BLIND_SEAT
            agents = [villain, villain]
            agents[hero_seat] = hero
            agent_seeds = (
                derive_seed(seed, "agent", deal_index, swap, 0),
                derive_seed(seed, "agent", deal_index, swap, 1),
            )
            result = play_hand(agents, deal_seed, agent_seeds)
            hands.append(PlayedHand(len(hands), deal_index, swap, hero_seat, result))
        if progress is not None:
            progress(deal_index + 1, deal_count)

    return MatchResult(
        match_id=make_match_id(hero, villain, deal_count, seed, seat_swap),
        hero_name=hero.name,
        villain_name=villain.name,
        seed=seed,
        deal_count=deal_count,
        seat_swap=seat_swap,
        hands=tuple(hands),
    )
