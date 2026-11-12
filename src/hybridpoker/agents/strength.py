"""Hand strength for agents: equity against one random opponent hand.

Before the flop, equity depends only on the hand's *class*: the two ranks
and whether the cards share a suit (169 classes, e.g. ``AKs``, ``AKo``,
``77``). Each class is computed once per sample count, from a fixed
representative hand and a seed derived from the class name, and then
cached. The cached value never depends on the order hands are played, so
runs stay reproducible.

After the flop, equity is estimated fresh with a seed drawn from the
agent's seeded random generator.
"""

from __future__ import annotations

import random
from functools import lru_cache

from pokerkit import Card

from hybridpoker.equity import estimate_equity
from hybridpoker.observation import Observation
from hybridpoker.seeding import derive_seed

RANK_ORDER = "23456789TJQKA"


def hand_class(hole_cards: tuple[Card, ...]) -> str:
    """Return the preflop class of two hole cards, e.g. ``"AKs"`` or ``"77"``."""
    first, second = hole_cards
    ranks = sorted(
        (first.rank.value, second.rank.value), key=RANK_ORDER.index, reverse=True
    )
    if ranks[0] == ranks[1]:
        return ranks[0] * 2
    suited = "s" if first.suit == second.suit else "o"
    return f"{ranks[0]}{ranks[1]}{suited}"


def _representative(hand_class_name: str) -> str:
    high, low = hand_class_name[0], hand_class_name[1]
    if len(hand_class_name) == 2:  # pair
        return f"{high}s{low}h"
    if hand_class_name[2] == "s":
        return f"{high}s{low}s"
    return f"{high}s{low}h"


@lru_cache(maxsize=None)
def preflop_equity(hand_class_name: str, sample_count: int) -> float:
    """Cached preflop equity of a hand class against one random hand."""
    seed = derive_seed("preflop", hand_class_name, sample_count)
    return estimate_equity(
        _representative(hand_class_name), (), sample_count=sample_count, seed=seed
    ).equity


def hand_strength(observation: Observation, sample_count: int, rng: random.Random) -> float:
    """Equity of the observed hand against one uniformly random opponent hand."""
    if not observation.board_cards:
        return preflop_equity(hand_class(observation.hole_cards), sample_count)
    return estimate_equity(
        observation.hole_cards,
        observation.board_cards,
        opponent_count=1,
        sample_count=sample_count,
        seed=rng.getrandbits(63),
    ).equity
