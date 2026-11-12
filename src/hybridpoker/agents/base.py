"""The common interface every agent implements."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from hybridpoker.actions import ActionCategory
from hybridpoker.observation import Observation, PublicHandSummary


@dataclass(frozen=True)
class Decision:
    """An agent's chosen action plus an optional explanation.

    Attributes:
        category: The abstract action to take. Must be legal.
        reason: Human-readable explanation, e.g. "call: equity 0.42 >
            pot odds 0.25". Logged for explainability; never shown to the
            opponent.
        equity: The agent's own equity estimate, if it computed one.
    """

    category: ActionCategory
    reason: str = ""
    equity: float | None = None


class Agent(ABC):
    """A poker-playing agent.

    ``act`` receives only an :class:`Observation` (never the opponent's
    cards) and a seeded random generator supplied by the simulator, which
    keeps every decision reproducible.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def act(self, observation: Observation, rng: random.Random) -> Decision:
        """Choose a legal action for the given observation."""

    def observe_hand_end(self, summary: PublicHandSummary) -> None:
        """Receive the public summary of a finished hand.

        Optional; the default does nothing, so agents that ignore it behave
        exactly as before. Phase 4 agents use it to track an opponent's
        public actions across hands.
        """
        return None

    def config(self) -> dict[str, Any]:
        """Settings that define this agent, stored in run metadata only."""
        return {"class": type(self).__name__, "name": self.name}
