"""Poker agents and a factory for creating them by name."""

from __future__ import annotations

from hybridpoker.agents.base import Agent, Decision
from hybridpoker.agents.probability import ProbabilityAgent, ProbabilityParameters
from hybridpoker.agents.simple import STYLE_PRESETS, RandomAgent, StyleAgent, StyleParameters

AGENT_NAMES = ("probability", "random", *STYLE_PRESETS)


def make_agent(name: str, equity_samples: int | None = None) -> Agent:
    """Create an agent by name: "probability", "random", or a style preset.

    ``equity_samples`` overrides the agent's default Monte Carlo sample count.
    """
    if name == "probability":
        parameters = ProbabilityParameters()
        if equity_samples is not None:
            parameters = ProbabilityParameters(equity_samples=equity_samples)
        return ProbabilityAgent(parameters=parameters)
    if name == "random":
        return RandomAgent()
    if name in STYLE_PRESETS:
        return StyleAgent.preset(name, equity_samples)
    raise ValueError(f"Unknown agent {name!r}; choose from {list(AGENT_NAMES)}.")


__all__ = [
    "AGENT_NAMES",
    "STYLE_PRESETS",
    "Agent",
    "Decision",
    "ProbabilityAgent",
    "ProbabilityParameters",
    "RandomAgent",
    "StyleAgent",
    "StyleParameters",
    "make_agent",
]
