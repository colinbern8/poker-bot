"""The probability-based baseline agent.

This agent combines the Phase 1 math (equity and pot odds) with a simple,
explainable decision rule. It assumes the opponent holds a random hand and
ignores how the opponent has played. Adding that information through a
learned opponent model is the research question of later phases.

Decision rule:

* Facing a bet:
    - raise the pot if equity >= ``raise_equity``;
    - otherwise call if equity >= pot odds + ``call_margin``;
    - otherwise fold.
* Nobody has bet:
    - bet half the pot if equity >= ``bet_equity``;
    - otherwise check.

The thresholds were chosen by reasoning, not tuned on match results.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from hybridpoker.actions import ActionCategory
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.agents.strength import hand_strength
from hybridpoker.observation import Observation
from hybridpoker.pot_odds import pot_odds

C = ActionCategory


@dataclass(frozen=True)
class ProbabilityParameters:
    equity_samples: int = 500
    call_margin: float = 0.02
    raise_equity: float = 0.70
    bet_equity: float = 0.60


class ProbabilityAgent(Agent):
    """Decides by comparing equity against pot odds and fixed thresholds."""

    def __init__(
        self, name: str = "probability", parameters: ProbabilityParameters | None = None
    ) -> None:
        super().__init__(name)
        self.parameters = parameters or ProbabilityParameters()

    def config(self) -> dict[str, Any]:
        return {**super().config(), **asdict(self.parameters)}

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        p = self.parameters
        equity = hand_strength(observation, p.equity_samples, rng)

        if observation.to_call > 0:
            odds = pot_odds(observation.pot_size, observation.to_call)
            needed = odds + p.call_margin
            if observation.can_raise and equity >= p.raise_equity:
                return Decision(
                    C.RAISE_POT,
                    f"raise: equity {equity:.2f} >= raise threshold {p.raise_equity:.2f}",
                    equity,
                )
            if equity >= needed:
                return Decision(
                    C.CHECK_CALL,
                    f"call: equity {equity:.2f} >= pot odds {odds:.2f} + margin {p.call_margin:.2f}",
                    equity,
                )
            return Decision(
                C.FOLD,
                f"fold: equity {equity:.2f} < pot odds {odds:.2f} + margin {p.call_margin:.2f}",
                equity,
            )

        if observation.can_raise and equity >= p.bet_equity:
            return Decision(
                C.RAISE_HALF_POT,
                f"bet: equity {equity:.2f} >= bet threshold {p.bet_equity:.2f}",
                equity,
            )
        return Decision(
            C.CHECK_CALL, f"check: equity {equity:.2f} < bet threshold {p.bet_equity:.2f}", equity
        )
