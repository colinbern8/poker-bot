"""Equity against an opponent's estimated continuing range (Phase 4, Stage 4B fix).

**Why.** In the EV of a bet, the "opponent calls" branch used the hero's
equity against a *random* hand. But an opponent who calls a bet is usually
stronger than a random hand, so large bets (especially all-in) were
overvalued.

**What (an approximation, not a reconstruction).** If the opponent model
predicts that the opponent continues (calls or re-raises) with probability
``q`` after a given bet, we assume the opponent continues with the
**strongest ``q`` fraction of the hands it could hold**, and value the
"called" branch with the hero's equity against those hands only. This is a
*strongest-hand-percentile* approximation. It is **not** a verified
reconstruction of the opponent's range: real players (and the scripted
bots' random bluffs) also continue with some weak hands, and earlier
actions in the hand already narrowed the opponent's range (ignored here).

**Which hands, ranked how (public information only):**

* Candidate opponent hands: every two-card combination not blocked by the
  board or the hero's own hole cards, equally likely. The opponent's real
  cards are never used.
* Preflop: ranked by the cached equity of the hand's class against a random
  hand (the same 169-class table, 200 samples, that the scripted bots use).
* Postflop: ranked by the made hand on the current board (PokerKit's
  ``StandardHighHand``). On the river this equals ranking by equity against
  a random hand; on the flop and turn it under-rates drawing hands (measured
  rank agreement with the bots' own equity criterion: 0.96 on a dry flop,
  0.87 on a draw-heavy flop), which makes the estimated range slightly too
  strong, i.e. errs toward caution.
* Ties are broken by a fixed card order, so results are reproducible.

**Low-noise estimate (common random numbers).** The agents use
:func:`range_adjustments`: one shared Monte Carlo run draws opponent hands
uniformly from *all* unblocked hands, together with random runouts, and
records each drawn hand's strength rank. From the **same** draws it
measures the hero's equity against the strongest ``q`` fraction and
against all hands, and returns the **difference**. The agent adds this
correction to its usual equity against a random hand. Because both parts
share their samples, the noise largely cancels when the range is wide, and
a full range (``q = 1``) gives a correction of exactly 0. Narrow ranges get
extra samples drawn from the range itself (at least ``MIN_TOP_SAMPLES``).
One run serves every bet size. On the river everything is exact.

**Cost.** Ranking all combinations takes about 6 ms on the flop and up to
about 220 ms on the river (cached per board and hero hand within a
process). Earlier streets use one shared run of ``samples`` runouts plus
any top-ups.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pokerkit import Card, Deck, StandardHighHand

from hybridpoker.agents.strength import hand_class, preflop_equity
from hybridpoker.equity import hero_pot_share

PREFLOP_RANK_SAMPLES = 200  # the bots' preflop equity table
RANGE_EQUITY_SAMPLES = 400  # one shared run for all bet sizes
MIN_TOP_SAMPLES = 60
_DECK: tuple[Card, ...] = tuple(Deck.STANDARD)
_FULL_BOARD = 5


@dataclass(frozen=True)
class OpponentRanking:
    """All unblocked opponent hands, strongest first."""

    combos: tuple[tuple[Card, Card], ...]
    river_values: tuple | None  # StandardHighHand per combo (river only), same order

    @property
    def size(self) -> int:
        return len(self.combos)


def _card_key(card: Card) -> int:
    return _DECK.index(card)


@lru_cache(maxsize=256)
def opponent_ranking(hero_hole: tuple[Card, ...], board: tuple[Card, ...]) -> OpponentRanking:
    """Unblocked opponent combinations ranked strongest first (public information only)."""
    blocked = set(hero_hole) | set(board)
    unseen = [c for c in _DECK if c not in blocked]
    combos = list(itertools.combinations(unseen, 2))
    if not board:
        strength = [preflop_equity(hand_class(c), PREFLOP_RANK_SAMPLES) for c in combos]
        values = None
    else:
        values = [StandardHighHand.from_game(c, board) for c in combos]
        strength = values
    order = sorted(range(len(combos)),
                   key=lambda i: (strength[i], -_card_key(combos[i][0]), -_card_key(combos[i][1])), reverse=True)
    ranked = tuple(combos[i] for i in order)
    river = tuple(values[i] for i in order) if values is not None and len(board) == _FULL_BOARD else None
    return OpponentRanking(ranked, river)


def top_count(ranking: OpponentRanking, fraction: float) -> int:
    """How many of the strongest hands make up the continuing range."""
    if not 0.0 <= fraction <= 1.0 + 1e-9:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
    return min(ranking.size, max(1, int(round(fraction * ranking.size))))


def equity_vs_top_fraction(
    hero_hole: tuple[Card, ...], board: tuple[Card, ...], fraction: float, samples: int, seed: int
) -> float:
    """Hero equity against the opponent's strongest ``fraction`` of unblocked hands.

    Exact on the river; Monte Carlo (``samples`` runouts, seeded) earlier.
    """
    hero_hole, board = tuple(hero_hole), tuple(board)
    ranking = opponent_ranking(hero_hole, board)
    k = top_count(ranking, fraction)
    subset = ranking.combos[:k]
    if ranking.river_values is not None:  # river: exact, no sampling needed
        hero_value = StandardHighHand.from_game(hero_hole, board)
        shares = [1.0 if hero_value > v else 0.5 if hero_value == v else 0.0 for v in ranking.river_values[:k]]
        return float(np.mean(shares))
    rng = random.Random(seed)
    blocked = set(hero_hole) | set(board)
    missing = _FULL_BOARD - len(board)
    total = 0.0
    for _ in range(samples):
        opponent = subset[rng.randrange(k)]
        remaining = [c for c in _DECK if c not in blocked and c not in opponent]
        runout = tuple(rng.sample(remaining, missing))
        total += hero_pot_share(hero_hole, [opponent], board + runout)
    return total / samples


def _share(hero_hole, opponent, board, missing, blocked, rng) -> float:
    remaining = [c for c in _DECK if c not in blocked and c not in opponent]
    runout = tuple(rng.sample(remaining, missing))
    return hero_pot_share(hero_hole, [opponent], board + runout)


def range_adjustments(
    hero_hole: tuple[Card, ...], board: tuple[Card, ...], fractions, samples: int, seed: int
) -> dict[float, float]:
    """For each fraction q: equity(strongest q of hands) minus equity(all hands).

    Both parts come from the same simulated draws (common random numbers),
    so the correction is low-noise, and exactly 0 when q covers every hand.
    Exact on the river.
    """
    hero_hole, board = tuple(hero_hole), tuple(board)
    ranking = opponent_ranking(hero_hole, board)
    counts = {q: top_count(ranking, q) for q in fractions}
    if ranking.river_values is not None:
        hero_value = StandardHighHand.from_game(hero_hole, board)
        shares = np.array([1.0 if hero_value > v else 0.5 if hero_value == v else 0.0 for v in ranking.river_values])
        overall = float(shares.mean())
        return {q: float(shares[:k].mean()) - overall for q, k in counts.items()}
    if all(k >= ranking.size for k in counts.values()):
        return {q: 0.0 for q in counts}
    rng = random.Random(seed)
    blocked = set(hero_hole) | set(board)
    missing = _FULL_BOARD - len(board)
    indices, shares = [], []
    for _ in range(samples):
        i = rng.randrange(ranking.size)  # uniform over all unblocked hands
        indices.append(i)
        shares.append(_share(hero_hole, ranking.combos[i], board, missing, blocked, rng))
    indices_arr, shares_arr = np.array(indices), np.array(shares)
    overall = float(shares_arr.mean())
    out = {}
    for q, k in sorted(counts.items(), key=lambda item: item[1]):
        if k >= ranking.size:
            out[q] = 0.0
            continue
        top = list(shares_arr[indices_arr < k])
        while len(top) < MIN_TOP_SAMPLES:  # narrow range: top up with draws from the range itself
            top.append(_share(hero_hole, ranking.combos[rng.randrange(k)], board, missing, blocked, rng))
        out[q] = float(np.mean(top)) - overall
    return out
