"""Record the AI's decisions in a hand and replay them exactly.

Each time the human acts, the controller replays the current hand from the
start with the unchanged :func:`hybridpoker.simulator.play_hand`.
:class:`RecordingAgent` wraps the AI so that:

* a decision the AI **already made** in this hand is returned from the
  record instead of being computed again; the AI's random generator is set
  to the exact state it had after that decision (``rng.setstate``), and the
  replayed observation must equal the recorded one;
* a **new** decision is computed by the real AI, then recorded together with
  the generator state and a copy of the AI's diagnostics.

``play_hand`` creates a fresh generator from the same seed on every replay,
so restoring the saved state after decision *k* leaves the generator exactly
where an uninterrupted hand would have it. All of the AI's randomness comes
from that generator, so interrupted and uninterrupted hands are identical
(proved in ``tests/test_demo_replay.py``).
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hybridpoker.agents.base import Agent, Decision
from hybridpoker.observation import Observation, PublicHandSummary


class ReplayDivergence(RuntimeError):
    """A replayed hand did not retrace the recorded one."""


@dataclass(frozen=True)
class RecordedDecision:
    """One AI decision, with what is needed to replay it exactly.

    Attributes:
        observation: The AI's observation (it contains the AI's own hole
            cards, so it is never shown to the human during the hand).
        decision: The AI's decision.
        rng_state_after: ``rng.getstate()`` right after the decision.
        diagnostics: A copy of the AI's ``DecisionDiagnostics`` for this
            decision, or None if the AI keeps no diagnostics.
    """

    observation: Observation
    decision: Decision
    rng_state_after: tuple
    diagnostics: Any | None


class RecordingAgent(Agent):
    """Wraps the AI for one replay of one hand."""

    def __init__(self, inner: Agent, recorded: Sequence[RecordedDecision] = ()) -> None:
        super().__init__(inner.name)
        self.inner = inner
        self._recorded = tuple(recorded)
        self._replayed = 0
        self.new_records: list[RecordedDecision] = []

    @property
    def records(self) -> tuple[RecordedDecision, ...]:
        """Every decision of this hand so far: the recorded ones, then the new ones."""
        return (*self._recorded, *self.new_records)

    def assert_fully_replayed(self) -> None:
        """Raise if the replay ended before reaching every recorded decision."""
        if self._replayed != len(self._recorded):
            raise ReplayDivergence(
                f"The replay reached {self._replayed} of {len(self._recorded)} recorded AI decisions.")

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        if self._replayed < len(self._recorded):
            record = self._recorded[self._replayed]
            if record.observation != observation:
                raise ReplayDivergence(f"AI decision {self._replayed} saw a different game state on replay.")
            rng.setstate(record.rng_state_after)
            self._replayed += 1
            return record.decision

        diagnostics = getattr(self.inner, "diagnostics", None)
        before = len(diagnostics) if diagnostics is not None else 0
        decision = self.inner.act(observation, rng)
        diagnostic = None
        if diagnostics is not None and len(diagnostics) > before:
            diagnostic = copy.deepcopy(diagnostics[-1])
        self.new_records.append(RecordedDecision(observation, decision, rng.getstate(), diagnostic))
        return decision

    def observe_hand_end(self, summary: PublicHandSummary) -> None:
        raise RuntimeError("The controller tells the AI about a finished hand itself, exactly once.")

    def config(self) -> dict[str, Any]:
        return self.inner.config()
