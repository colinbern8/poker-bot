"""Opponent bots: a random baseline and configurable playing styles.

``StyleAgent`` plays according to three numbers between 0 and 1:

* **looseness**: how willing it is to call. It calls when its equity is at
  least ``pot_odds * 2 * (1 - looseness)``. At 0.5 that is exactly the
  break-even point; at 1.0 it never folds; at 0.1 it needs 1.8x the
  break-even equity.
* **aggression**: how often it bets or raises its strong hands, and how
  low its bar for "strong" is.
* **bluff_rate**: how often it bets or raises regardless of its hand.

Equity is estimated against a random hand with a small number of samples,
so the bots are deliberately imperfect and noisy. These settings are
private: they go into run metadata, never into model input features.
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


class RandomAgent(Agent):
    """Chooses uniformly among the legal action categories."""

    def __init__(self, name: str = "random") -> None:
        super().__init__(name)

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        return Decision(rng.choice(observation.legal_categories), reason="random choice")


@dataclass(frozen=True)
class StyleParameters:
    looseness: float
    aggression: float
    bluff_rate: float
    equity_samples: int = 200

    def __post_init__(self) -> None:
        for name in ("looseness", "aggression", "bluff_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.equity_samples < 1:
            raise ValueError("equity_samples must be at least 1.")

    @property
    def value_raise_equity(self) -> float:
        """Equity at which it raises for value when facing a bet."""
        return 0.80 - 0.25 * self.aggression

    @property
    def value_bet_equity(self) -> float:
        """Equity at which it bets for value when nobody has bet."""
        return 0.65 - 0.25 * self.aggression


STYLE_PRESETS: dict[str, StyleParameters] = {
    # Calls everything, never raises.
    "calling_station": StyleParameters(looseness=1.0, aggression=0.0, bluff_rate=0.0),
    # Raises and bluffs a lot, calls wide.
    "maniac": StyleParameters(looseness=0.75, aggression=0.85, bluff_rate=0.35),
    # Plays only strong hands, rarely raises, never bluffs.
    "nit": StyleParameters(looseness=0.1, aggression=0.3, bluff_rate=0.0),
    # Close to break-even calling, raises good hands, bluffs occasionally.
    "solid": StyleParameters(looseness=0.45, aggression=0.6, bluff_rate=0.08),
}


class StyleAgent(Agent):
    """A rule-based bot whose style is set by :class:`StyleParameters`."""

    def __init__(self, name: str, parameters: StyleParameters) -> None:
        super().__init__(name)
        self.parameters = parameters

    @classmethod
    def preset(cls, preset_name: str, equity_samples: int | None = None) -> StyleAgent:
        if preset_name not in STYLE_PRESETS:
            raise ValueError(
                f"Unknown style preset {preset_name!r}; choose from {sorted(STYLE_PRESETS)}."
            )
        parameters = STYLE_PRESETS[preset_name]
        if equity_samples is not None:
            parameters = StyleParameters(
                parameters.looseness, parameters.aggression, parameters.bluff_rate, equity_samples
            )
        return cls(preset_name, parameters)

    def config(self) -> dict[str, Any]:
        return {**super().config(), **asdict(self.parameters)}

    def _raise_size(self, equity: float, rng: random.Random) -> ActionCategory:
        p = self.parameters
        if equity >= 0.9 and p.aggression >= 0.5:
            return C.ALL_IN
        return C.RAISE_POT if rng.random() < p.aggression else C.RAISE_HALF_POT

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        p = self.parameters
        equity = hand_strength(observation, p.equity_samples, rng)

        if observation.to_call > 0:
            odds = pot_odds(observation.pot_size, observation.to_call)
            call_needed = odds * 2 * (1 - p.looseness)
            if observation.can_raise and equity >= p.value_raise_equity and rng.random() < p.aggression:
                return Decision(self._raise_size(equity, rng), "value raise", equity)
            if observation.can_raise and rng.random() < p.bluff_rate:
                return Decision(C.RAISE_HALF_POT, "bluff raise", equity)
            if equity >= call_needed:
                return Decision(C.CHECK_CALL, f"call: equity {equity:.2f} >= {call_needed:.2f}", equity)
            return Decision(C.FOLD, f"fold: equity {equity:.2f} < {call_needed:.2f}", equity)

        if observation.can_raise and equity >= p.value_bet_equity and rng.random() < p.aggression:
            return Decision(self._raise_size(equity, rng), "value bet", equity)
        if observation.can_raise and rng.random() < p.bluff_rate:
            return Decision(C.RAISE_HALF_POT, "bluff bet", equity)
        return Decision(C.CHECK_CALL, "check", equity)
