"""Shared helpers for the Phase 5A tests (not a test module).

They build cheap AI opponents (low sample counts, the small ``tiny_bundle``
model instead of the frozen one) and play **reference** hands and matches with
the unchanged ``play_hand``, so the demonstration code can be compared with
them exactly.
"""

from __future__ import annotations

import dataclasses
import math
import random
import statistics

import numpy as np

from hybridpoker.actions import RAISE_CATEGORIES, ActionCategory
from hybridpoker.agents import make_agent
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.agents.ev_agent import EVAgent, MLResponseModel, SituationFrequencyResponseModel
from hybridpoker.agents.probability import ProbabilityAgent, ProbabilityParameters
from hybridpoker.observation import Observation
from hybridpoker.opponent_predictor import OpponentPredictor
from hybridpoker.simulator import HandResult, play_hand, public_hand_summary
from hybridpoker.state_coverage import COVERAGE_PATH, load_coverage
from hybridpoker_demo.seeds import HandSetup, hand_setup

SAMPLES = 40  # equity and range samples for test AIs (the real agent uses 500 and 400)
TIMING_FIELDS = ("equity_seconds", "model_seconds", "range_seconds", "total_seconds")


class ShoveEV(EVAgent):
    """The ML EV agent with its choice forced to all-in whenever all-in is legal.

    A test stand-in that reproduces an AI shove on demand (the real agent
    shoves only when its EV calculation says so). Everything else, including
    the diagnostics, is the unchanged agent.
    """

    @staticmethod
    def choose(values, chips):
        shove = next((v for v in values if v.category is ActionCategory.ALL_IN), None)
        return shove or EVAgent.choose(values, chips)


class AllLowCoverageEV(EVAgent):
    """The unchanged ML EV agent, except that every bet option is marked as low training coverage.

    Coverage is a diagnostic only (it does not change any decision), so this
    test stand-in plays exactly like the agent while making the "few similar
    training examples" warnings appear for every option it could bet with.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.low_coverage_threshold = 10**12


class ScriptedAggressorEV(EVAgent):
    """A test stand-in that ignores its EV calculation and follows a fixed script.

    It creates the post-flop betting spots of the Fold-availability tests and
    records no diagnostics:

    * preflop: check or call;
    * flop: bet or raise half the pot until it has made two bets or raises on the
      flop, then call;
    * turn and river: all-in when checked to, otherwise call.
    """

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        legal = observation.legal_categories
        choice = ActionCategory.CHECK_CALL
        if observation.street == "flop":
            mine = sum(a.street == "flop" and a.position == observation.position and a.category in RAISE_CATEGORIES
                       for a in observation.history)
            if mine < 2 and ActionCategory.RAISE_HALF_POT in legal:
                choice = ActionCategory.RAISE_HALF_POT
        elif observation.street != "preflop" and observation.to_call == 0 and ActionCategory.ALL_IN in legal:
            choice = ActionCategory.ALL_IN
        return Decision(choice, "scripted")


def ml_agent(bundle: dict, cls: type[EVAgent] = EVAgent) -> EVAgent:
    """An ML EV agent like the frozen one, with a small test model and few samples.

    ``cls`` may be a test subclass that counts calls or injects failures.
    """
    return cls("ev_ml", MLResponseModel(OpponentPredictor(bundle)), SAMPLES,
               coverage=load_coverage(COVERAGE_PATH), range_samples=SAMPLES)


def situation_agent() -> EVAgent:
    return EVAgent("ev_situation", SituationFrequencyResponseModel.load(), SAMPLES,
                   coverage=load_coverage(COVERAGE_PATH), range_samples=SAMPLES)


def probability_agent() -> ProbabilityAgent:
    return ProbabilityAgent(ProbabilityParameters(equity_samples=SAMPLES))


def stand_in(style: str) -> Agent:
    """A scripted bot that plays the human's seat in reference games."""
    return make_agent(style, 30)


class Probe(Agent):
    """Passes decisions through unchanged and records the AI's generator state after each one."""

    def __init__(self, inner: Agent) -> None:
        super().__init__(inner.name)
        self.inner = inner
        self.states: list[tuple] = []

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        decision = self.inner.act(observation, rng)
        self.states.append(rng.getstate())
        return decision


def seat_agents(setup: HandSetup, human: Agent, ai: Agent) -> list[Agent]:
    agents: list[Agent] = [human, human]
    agents[setup.ai_seat] = ai
    return agents


def reference_hand(ai: Agent, human: Agent, setup: HandSetup) -> tuple[HandResult, list[tuple]]:
    """One uninterrupted hand with the unchanged simulator; returns the result and the AI's rng states."""
    probe = Probe(ai)
    result = play_hand(seat_agents(setup, human, probe), setup.deal_seed, setup.agent_seeds)
    return result, probe.states


def human_categories(result: HandResult, setup: HandSetup) -> list:
    return [r.action.category for r in result.decisions if r.seat == setup.human_seat]


def notify(ai: Agent, result: HandResult, setup: HandSetup) -> None:
    """The end-of-hand message the controller sends (one hand per deal, no seat swap)."""
    ai.observe_hand_end(public_hand_summary(result, setup.ai_seat, setup.hand_index, 0, 1))


def reference_match(ai: Agent, style: str, match_seed: int, hands: int) -> list[dict]:
    """An uninterrupted match against a stand-in bot; records everything needed for comparison."""
    human = stand_in(style)
    out = []
    for h in range(hands):
        setup = hand_setup(match_seed, h)
        result, states = reference_hand(ai, human, setup)
        notify(ai, result, setup)
        out.append({"setup": setup, "result": result, "rng_states": states,
                    "tracker": tracker_state(ai)})
    return out


def play_controller_hand(controller, actions: list) -> None:
    """Answer every human turn of the current hand with ``actions`` (in order)."""
    from hybridpoker_demo.controller import HumanTurn

    given = 0
    while isinstance(controller.view, HumanTurn):
        controller.submit(controller.view.ticket, actions[given])
        given += 1
    assert given == len(actions)


def tracker_state(ai: Agent) -> dict | None:
    tracker = getattr(ai, "tracker", None)
    if tracker is None:
        return None
    return {"rows": [dict(r) for r in tracker._rows], "observed": tracker.observed_rows(),
            "features": tracker.history_features().to_dict()}


def without_timing(diagnostics) -> dict:
    data = dataclasses.asdict(diagnostics)
    for name in TIMING_FIELDS:
        data.pop(name)
    return data


def assert_same_diagnostics(actual: list, expected: list) -> None:
    """Equal diagnostics apart from wall-clock timings (NaN equals NaN)."""
    assert len(actual) == len(expected)
    np.testing.assert_equal([without_timing(d) for d in actual], [without_timing(d) for d in expected])


def assert_same_hand(actual: HandResult, expected: HandResult, ai_seat: int) -> None:
    """Same cards, board, payoffs, showdown, and every decision's state and action.

    The human seat's ``Decision`` objects and agent names differ by design
    (a person versus the stand-in bot); the AI's decisions must be identical.
    """
    assert actual.deal_seed == expected.deal_seed
    assert actual.hole_cards == expected.hole_cards
    assert actual.board_cards == expected.board_cards
    assert actual.payoffs == expected.payoffs
    assert actual.went_to_showdown == expected.went_to_showdown
    assert len(actual.decisions) == len(expected.decisions)
    for a, e in zip(actual.decisions, expected.decisions):
        assert (a.decision_index, a.seat, a.observation, a.action) == \
            (e.decision_index, e.seat, e.observation, e.action)
        if a.seat == ai_seat:
            assert a.decision == e.decision
            assert a.agent_name == e.agent_name


# --- Latency reporting for the opt-in acceptance tests (Phase 5C) --------------------------------------

P95_TARGET_SECONDS = 2.0       # AI decision time, 95th percentile: a target, not a pass/fail limit
STARTUP_TARGET_SECONDS = 90.0  # model load + SHA-256 check + cache warm-up: a target, not a limit
FEW_SAMPLES = 20               # below this, the 95th percentile is close to the maximum


def p95(seconds: list[float]) -> float:
    """95th percentile by the nearest-rank method: the smallest value >= 95% of the sample."""
    if not seconds:
        raise ValueError("No decision times to summarize.")
    ordered = sorted(seconds)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def latency_report(seconds: list[float], label: str) -> str:
    """One line: count, median, p95 and max decision time, and whether the p95 target was met."""
    value = p95(seconds)
    status = "met" if value < P95_TARGET_SECONDS else "NOT met (a target, not a test failure)"
    note = f"; only {len(seconds)} decisions, so p95 is close to the max" if len(seconds) < FEW_SAMPLES else ""
    return (f"{label}: {len(seconds)} AI decisions; median {1000 * statistics.median(seconds):.0f} ms, "
            f"p95 {1000 * value:.0f} ms, max {1000 * max(seconds):.0f} ms; "
            f"target p95 < {1000 * P95_TARGET_SECONDS:.0f} ms: {status}{note}")


def startup_report(seconds: float, label: str) -> str:
    status = "met" if seconds < STARTUP_TARGET_SECONDS else "NOT met (a target, not a test failure)"
    return f"{label}: {seconds:.1f} s; target < {STARTUP_TARGET_SECONDS:.0f} s: {status}"
