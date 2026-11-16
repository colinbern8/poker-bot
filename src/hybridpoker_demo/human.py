"""The human player as seen by the unchanged simulator.

:func:`hybridpoker.simulator.play_hand` asks each agent for a decision and
waits for the answer. A person cannot answer inside that call, so the human
is represented by :class:`ScriptedHumanAgent`: it replays the actions the
person has already chosen in this hand, and when it runs out it raises
:class:`HumanInputRequired`. The exception carries the human's own
:class:`~hybridpoker.observation.Observation` (their cards, the board, the
pot, the stacks and the legal actions), which is exactly what the screen may
show. ``play_hand`` does not catch exceptions, so it stops cleanly and
nothing of the unfinished hand is kept.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from hybridpoker.actions import ActionCategory
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.observation import Observation

HUMAN_NAME = "human"


class HumanInputRequired(Exception):
    """Raised inside ``play_hand`` when the human must choose an action."""

    def __init__(self, observation: Observation) -> None:
        super().__init__("The human player must act.")
        self.observation = observation


class ScriptedHumanAgent(Agent):
    """Plays back the human's chosen actions for one hand, then asks for the next one."""

    def __init__(self, actions: Sequence[ActionCategory], name: str = HUMAN_NAME) -> None:
        super().__init__(name)
        self._actions = tuple(ActionCategory(a) for a in actions)
        self._next = 0

    @property
    def used(self) -> int:
        """How many of the scripted actions have been played so far."""
        return self._next

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        if self._next == len(self._actions):
            raise HumanInputRequired(observation)
        category = self._actions[self._next]
        self._next += 1
        return Decision(category, reason="human")
