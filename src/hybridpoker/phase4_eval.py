"""Phase 4 Stage 4C: paired development matches and their statistics.

Design (``docs/phase4_plan.md`` section 6):

* **Heroes:** the ProbabilityAgent, the Global Frequency EV Agent, the
  Situation-Aware Frequency EV Agent and the ML EV Agent.
* **Shared deals.** Against opponent number ``i`` every hero plays the same
  seat-swapped match seed, ``derive_seed(master, "dev", i)``, so all heroes
  get identical cards in identical seats (common random numbers). Each hero
  starts the match with no memory; the ML agent learns the opponent's
  public history during the match, as in the final evaluation.
* **Unit of analysis: the deal pair** (both seatings of one deal). Paired
  differences between heroes are taken deal by deal against the same
  opponent.
* **Master seeds.** Development uses 5000. The final evaluation seed 4000 is
  refused here so that it cannot be used by accident before the freeze.

Everything a hero does is recorded per decision (category, all-in, latency,
and for EV agents the low-coverage flags of its candidate bets) and per hand
(net chips, whether the hero chose or called an all-in). Nothing here
changes a decision.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from hybridpoker.actions import ActionCategory
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.agents.ev_agent import EVAgent, make_frequency_ev_agent, make_ml_ev_agent, make_situation_ev_agent
from hybridpoker.agents.probability import ProbabilityAgent
from hybridpoker.game import BIG_BLIND
from hybridpoker.match import run_match
from hybridpoker.observation import Observation, PublicHandSummary
from hybridpoker.opponent_configs import OpponentConfig
from hybridpoker.seeding import derive_seed

DEV_MASTER_SEED = 5000
FINAL_MASTER_SEED = 4000
HEROES = ("probability", "ev_global", "ev_situation", "ev_ml")
HERO_LABELS = {"probability": "ProbabilityAgent", "ev_global": "Global Frequency EV",
               "ev_situation": "Situation-Aware Frequency EV", "ev_ml": "ML EV"}
# (treatment, control): the order in which paired differences are reported
COMPARISONS = (("ev_ml", "ev_situation"), ("ev_ml", "ev_global"), ("ev_ml", "probability"),
               ("ev_situation", "ev_global"), ("ev_situation", "probability"), ("ev_global", "probability"))


def dev_match_seed(opponent_index: int, master_seed: int = DEV_MASTER_SEED) -> int:
    """Match seed shared by every hero against opponent ``opponent_index``."""
    if master_seed == FINAL_MASTER_SEED:
        raise ValueError("Master seed 4000 is reserved for the frozen final evaluation.")
    return derive_seed(master_seed, "dev", opponent_index)


def make_hero(name: str) -> Agent:
    if name == "probability":
        return ProbabilityAgent()
    if name == "ev_global":
        return make_frequency_ev_agent("ev_global")
    if name == "ev_situation":
        return make_situation_ev_agent("ev_situation")
    if name == "ev_ml":
        return make_ml_ev_agent("ev_ml")
    raise ValueError(f"Unknown hero {name!r}; choose from {HEROES}.")


class TimedAgent(Agent):
    """Wraps an agent and records the wall-clock time of every decision."""

    def __init__(self, inner: Agent) -> None:
        super().__init__(inner.name)
        self.inner = inner
        self.latencies: list[float] = []

    def config(self) -> dict[str, Any]:
        return self.inner.config()

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        start = time.perf_counter()
        decision = self.inner.act(observation, rng)
        self.latencies.append(time.perf_counter() - start)
        return decision

    def observe_hand_end(self, summary: PublicHandSummary) -> None:
        self.inner.observe_hand_end(summary)


@dataclass(frozen=True)
class HeroDecision:
    deal_index: int
    seat_swap: int
    street: str
    category: str
    is_all_in: bool  # this action put the hero all-in (a shove or an all-in call)
    latency: float
    equity: float | None
    chosen_low_coverage: bool | None  # EV agents: the chosen bet led to a poorly covered state
    any_low_coverage: bool | None  # EV agents: some candidate bet led to a poorly covered state
    equity_seconds: float | None = None
    model_seconds: float | None = None
    range_seconds: float | None = None


@dataclass(frozen=True)
class HeroHand:
    deal_index: int
    seat_swap: int
    hero_net: int
    chose_all_in: bool  # the hero chose the all_in action at least once
    hero_all_in: bool  # the hero ended up all-in (shove or all-in call)
    showdown: bool
    cards_hash: str  # deck seed and hole cards, to check that deals are shared across heroes


def _cards_hash(result) -> str:
    """Deck seed and hole cards. The board is left out: how much of it is dealt depends on play."""
    text = f"{result.deal_seed}|" + "|".join("".join(map(str, cards)) for cards in result.hole_cards)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def play_dev_match(hero_name: str, opponent: OpponentConfig, opponent_index: int, deals: int,
                   master_seed: int = DEV_MASTER_SEED) -> dict[str, Any]:
    """Play one seat-swapped match and return per-hand and per-decision records (never raises)."""
    seed = dev_match_seed(opponent_index, master_seed)
    out: dict[str, Any] = {"hero": hero_name, "opponent": opponent.name, "family": opponent.family,
                           "pool": opponent.pool, "opponent_index": opponent_index, "seed": seed, "deals": deals,
                           "error": None, "hands": [], "decisions": []}
    start = time.perf_counter()
    try:
        inner = make_hero(hero_name)
        hero = TimedAgent(inner)
        match = run_match(hero, opponent.make_agent(), deals, seed, seat_swap=True)
        diagnostics = inner.diagnostics if isinstance(inner, EVAgent) else None
        k = 0
        for hand in match.hands:
            records = [r for r in hand.result.decisions if r.seat == hand.hero_seat]
            for record in records:
                diag = diagnostics[k] if diagnostics is not None else None
                chosen = record.action.category.value
                out["decisions"].append(asdict(HeroDecision(
                    deal_index=hand.deal_index, seat_swap=hand.seat_swap, street=record.observation.street,
                    category=chosen, is_all_in=record.action.is_all_in, latency=hero.latencies[k],
                    equity=record.decision.equity,
                    chosen_low_coverage=(bool(diag.coverage[chosen]["low"]) if chosen in diag.coverage else False)
                    if diag is not None else None,
                    any_low_coverage=any(c["low"] for c in diag.coverage.values()) if diag is not None else None,
                    equity_seconds=diag.equity_seconds if diag else None,
                    model_seconds=diag.model_seconds if diag else None,
                    range_seconds=diag.range_seconds if diag else None)))
                if diag is not None and diag.chosen != chosen:
                    raise RuntimeError("Diagnostics are out of step with the played decisions.")
                k += 1
            out["hands"].append(asdict(HeroHand(
                deal_index=hand.deal_index, seat_swap=hand.seat_swap, hero_net=hand.hero_net,
                chose_all_in=any(r.action.category is ActionCategory.ALL_IN for r in records),
                hero_all_in=any(r.action.is_all_in for r in records), showdown=hand.result.went_to_showdown,
                cards_hash=_cards_hash(hand.result))))
        if k != len(hero.latencies):
            raise RuntimeError("Hero decision count differs from the number of timed decisions.")
    except Exception:  # recorded and reported, never hidden
        out["error"] = traceback.format_exc()
    out["seconds"] = time.perf_counter() - start
    out["result_hash"] = hashlib.sha256(repr([(h["deal_index"], h["seat_swap"], h["hero_net"])
                                              for h in out["hands"]]).encode()).hexdigest()
    return out


# --- Statistics ----------------------------------------------------------------------------------


def net_by_deal(hands: Sequence[dict[str, Any]], deals: int) -> np.ndarray:
    """Hero net chips per deal pair (both seatings summed)."""
    totals = np.zeros(deals)
    for hand in hands:
        totals[hand["deal_index"]] += hand["hero_net"]
    return totals


def to_bb100(pair_chips: np.ndarray) -> np.ndarray:
    """Deal-pair chips -> bb/100 per pair (2 hands per pair)."""
    return np.asarray(pair_chips, dtype=float) / 2 / BIG_BLIND * 100


@dataclass(frozen=True)
class Estimate:
    mean: float
    se: float
    low: float
    high: float
    units: int
    sd: float

    @classmethod
    def from_units(cls, values: Sequence[float], confidence: float = 0.95) -> Estimate:
        values = np.asarray(values, dtype=float)
        n = len(values)
        if n < 2:
            raise ValueError("At least 2 units are needed.")
        sd = float(values.std(ddof=1))
        se = sd / math.sqrt(n)
        z = NormalDist().inv_cdf(0.5 + confidence / 2)
        mean = float(values.mean())
        return cls(mean, se, mean - z * se, mean + z * se, n, sd)

    @classmethod
    def equal_weight(cls, parts: Sequence[Estimate], confidence: float = 0.95) -> Estimate:
        """Average of independent per-opponent estimates, each with weight 1/k."""
        k = len(parts)
        mean = sum(p.mean for p in parts) / k
        se = math.sqrt(sum(p.se ** 2 for p in parts)) / k
        z = NormalDist().inv_cdf(0.5 + confidence / 2)
        sd = math.sqrt(sum(p.sd ** 2 for p in parts) / k)  # root-mean-square per-pair SD
        return cls(mean, se, mean - z * se, mean + z * se, sum(p.units for p in parts), sd)


def pairs_needed(per_opponent_sd: Sequence[float], half_width: float, confidence: float = 0.95) -> int:
    """Deal pairs per opponent so that the equal-weight average has the given CI half-width."""
    k = len(per_opponent_sd)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    return math.ceil((z * math.sqrt(sum(s ** 2 for s in per_opponent_sd)) / (k * half_width)) ** 2)


def half_width(per_opponent_sd: Sequence[float], pairs: int, confidence: float = 0.95) -> float:
    k = len(per_opponent_sd)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    return z * math.sqrt(sum(s ** 2 for s in per_opponent_sd) / pairs) / k
