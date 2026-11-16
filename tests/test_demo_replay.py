"""Phase 5A go/no-go gate: interrupted, replayed hands equal uninterrupted ones.

A hand is played twice:

1. **Reference:** the unchanged ``play_hand`` with a stand-in bot in the
   human's seat, run straight through.
2. **Interrupted:** the human's seat is a :class:`ScriptedHumanAgent` fed the
   stand-in's actions one at a time. Every time the human must act, the hand
   stops and is replayed from the start; already-made AI decisions come from
   :class:`RecordingAgent`.

Both must agree on cards, actions, payoffs, every AI decision, the AI's
random-generator state after every decision, and the AI's diagnostics
(apart from timings). A negative control shows the test would notice a
replay that forgot to restore the generator.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from demo_support import (
    assert_same_diagnostics,
    assert_same_hand,
    human_categories,
    ml_agent,
    notify,
    probability_agent,
    reference_hand,
    reference_match,
    seat_agents,
    situation_agent,
    stand_in,
    tracker_state,
)
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.observation import Observation
from hybridpoker.simulator import play_hand
from hybridpoker_demo.human import HumanInputRequired, ScriptedHumanAgent
from hybridpoker_demo.recording import RecordingAgent, ReplayDivergence
from hybridpoker_demo.seeds import hand_setup

SEEDS = (11, 2024, 31337)
HANDS = 6
STYLES = ("maniac", "solid", "calling_station")


class Counting(Agent):
    """Counts real (non-replayed) decisions of the wrapped AI; exposes its diagnostics."""

    def __init__(self, inner: Agent) -> None:
        super().__init__(inner.name)
        self.inner = inner
        self.calls = 0

    @property
    def diagnostics(self):
        return getattr(self.inner, "diagnostics", None)

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        self.calls += 1
        return self.inner.act(observation, rng)


def play_interrupted(ai: Agent, setup, actions: list, extra_replays: int = 0, recorder=RecordingAgent):
    """Play one hand, stopping whenever the human must act, then resuming by replay.

    Returns the final result, the AI records and the human observations seen at each stop.
    """
    records: tuple = ()
    given: list = []
    seen: list[Observation] = []
    while True:
        for attempt in range(1 + extra_replays):
            human = ScriptedHumanAgent(given)
            wrapper = recorder(ai, records)
            try:
                result = play_hand(seat_agents(setup, human, wrapper), setup.deal_seed, setup.agent_seeds)
            except HumanInputRequired as stop:
                wrapper.assert_fully_replayed()
                assert human.used == len(given)
                if attempt == 0:
                    records = wrapper.records
                    seen.append(stop.observation)
                else:  # an identical replay adds nothing and sees the same state
                    assert wrapper.records == records
                    assert stop.observation == seen[-1]
                continue
            wrapper.assert_fully_replayed()
            assert human.used == len(given)
            return result, wrapper.records, seen
        given.append(actions[len(given)])


@pytest.fixture(params=["ml", "situation", "probability"])
def make_ai(request, tiny_bundle):
    return {"ml": lambda: ml_agent(tiny_bundle), "situation": situation_agent,
            "probability": probability_agent}[request.param]


@pytest.mark.parametrize("seed", SEEDS)
def test_interrupted_hands_equal_uninterrupted_hands(make_ai, seed) -> None:
    for h in range(HANDS):
        setup = hand_setup(seed, h)
        style = STYLES[(seed + h) % len(STYLES)]
        reference_ai, demo_ai = make_ai(), make_ai()
        expected, expected_states = reference_hand(reference_ai, stand_in(style), setup)

        counter = Counting(demo_ai)
        actual, records, seen = play_interrupted(counter, setup, human_categories(expected, setup), extra_replays=2)

        assert_same_hand(actual, expected, setup.ai_seat)
        assert [r.rng_state_after for r in records] == expected_states
        assert [r.decision for r in records] == [d.decision for d in expected.decisions if d.seat == setup.ai_seat]
        assert [r.observation for r in records] == \
            [d.observation for d in expected.decisions if d.seat == setup.ai_seat]
        # The human saw exactly the reference human's observations, and nothing was recomputed on replays.
        assert seen == [d.observation for d in expected.decisions if d.seat == setup.human_seat]
        assert counter.calls == len(records)
        if hasattr(demo_ai, "diagnostics"):
            assert_same_diagnostics(demo_ai.diagnostics, reference_ai.diagnostics)
            assert_same_diagnostics([r.diagnostics for r in records], reference_ai.diagnostics)


@pytest.mark.parametrize("seed", SEEDS)
def test_interrupted_match_keeps_opponent_tracking_identical(tiny_bundle, seed) -> None:
    """Several hands in a row: tracker rows and history features match after every hand."""
    expected = reference_match(ml_agent(tiny_bundle), STYLES[seed % 3], seed, HANDS)
    demo_ai = ml_agent(tiny_bundle)
    for h, ref in enumerate(expected):
        setup = ref["setup"]
        actual, records, _ = play_interrupted(demo_ai, setup, human_categories(ref["result"], setup))
        assert_same_hand(actual, ref["result"], setup.ai_seat)
        assert [r.rng_state_after for r in records] == ref["rng_states"]
        notify(demo_ai, actual, setup)
        assert tracker_state(demo_ai) == ref["tracker"]
    assert demo_ai.tracker.observed_rows() > 0


def test_replay_without_restoring_the_generator_is_detected(tiny_bundle) -> None:
    """Negative control: a replay that skips ``rng.setstate`` breaks equivalence for some hand."""

    class NoRestore(RecordingAgent):
        def act(self, observation, rng):
            if self._replayed < len(self._recorded):
                record = self._recorded[self._replayed]
                self._replayed += 1
                return record.decision
            return super().act(observation, rng)

    detected = 0
    for seed in SEEDS:
        for h in range(HANDS):
            setup = hand_setup(seed, h)
            reference_ai, demo_ai = ml_agent(tiny_bundle), ml_agent(tiny_bundle)
            expected, states = reference_hand(reference_ai, stand_in("maniac"), setup)
            try:
                actual, records, _ = play_interrupted(demo_ai, setup, human_categories(expected, setup),
                                                      recorder=NoRestore)
            except Exception:  # e.g. a different AI action makes a scripted human action illegal
                detected += 1
                continue
            same = actual == expected or (
                [r.rng_state_after for r in records] == states
                and [r.decision for r in records] == [d.decision for d in expected.decisions
                                                      if d.seat == setup.ai_seat])
            detected += not same
    assert detected > 0


def test_a_replay_that_sees_a_different_state_is_rejected(tiny_bundle) -> None:
    for h in range(HANDS):
        setup = hand_setup(7, h)
        expected, _ = reference_hand(ml_agent(tiny_bundle), stand_in("maniac"), setup)
        actions = human_categories(expected, setup)
        ai = ml_agent(tiny_bundle)
        _, records, _ = play_interrupted(ai, setup, actions)
        if len(records) < 2 or not actions:
            continue
        # A tampered record: the first AI decision claims a different observation.
        other = dataclasses.replace(records[0].observation, pot_size=records[0].observation.pot_size + 1)
        tampered = (dataclasses.replace(records[0], observation=other), *records[1:])
        with pytest.raises(ReplayDivergence):
            play_hand(seat_agents(setup, ScriptedHumanAgent(actions), RecordingAgent(ai, tampered)),
                      setup.deal_seed, setup.agent_seeds)
        # A replay that stops before reaching every recorded decision is also rejected
        # (with no human actions it stops at the human's first turn, before the second AI decision).
        wrapper = RecordingAgent(ai, records)
        with pytest.raises(HumanInputRequired):
            play_hand(seat_agents(setup, ScriptedHumanAgent([]), wrapper), setup.deal_seed, setup.agent_seeds)
        with pytest.raises(ReplayDivergence):
            wrapper.assert_fully_replayed()
        return
    pytest.fail("no hand with at least two AI decisions was found")


def test_recording_agent_never_forwards_end_of_hand_messages(tiny_bundle) -> None:
    with pytest.raises(RuntimeError):
        RecordingAgent(ml_agent(tiny_bundle)).observe_hand_end(None)
