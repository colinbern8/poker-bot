"""Tests for the agents: random, style presets, and the probability baseline."""

import random
from collections import Counter

import pytest
from pokerkit import Card

from hybridpoker.actions import ActionCategory, LegalAction
from hybridpoker.agents import (
    AGENT_NAMES,
    STYLE_PRESETS,
    ProbabilityAgent,
    ProbabilityParameters,
    RandomAgent,
    StyleAgent,
    StyleParameters,
    make_agent,
)
from hybridpoker.agents.strength import hand_class, preflop_equity
from hybridpoker.observation import Observation
from hybridpoker.simulator import play_hand

C = ActionCategory
FAST_SAMPLES = 40  # keeps statistical agent tests quick


def river_observation(hole: str, board: str, pot_size: int, to_call: int) -> Observation:
    """A river decision with every action legal (amounts are illustrative)."""
    legal = []
    if to_call > 0:
        legal.append(LegalAction(C.FOLD, 0, 0, False))
    legal += [
        LegalAction(C.CHECK_CALL, to_call, to_call, False),
        LegalAction(C.RAISE_HALF_POT, 40, 40, False),
        LegalAction(C.RAISE_POT, 60, 60, False),
        LegalAction(C.ALL_IN, 150, 150, True),
    ]
    return Observation(
        hole_cards=tuple(Card.parse(hole)),
        board_cards=tuple(Card.parse(board)),
        street="river",
        position="button",
        pot_size=pot_size,
        to_call=to_call,
        own_stack=150,
        opponent_stack=140,
        effective_stack=140,
        own_bet=0,
        opponent_bet=to_call,
        big_blind=2,
        legal_actions=tuple(legal),
        history=(),
    )


def action_counts(agent, opponent_name: str = "random", hands: int = 300) -> Counter:
    """How often ``agent`` chose each category over many hands, in both seats."""
    counts: Counter = Counter()
    opponent = make_agent(opponent_name, FAST_SAMPLES)
    for i in range(hands):
        seats = [agent, opponent] if i % 2 == 0 else [opponent, agent]
        hand = play_hand(seats, deal_seed=i, agent_seeds=(i, i + 50_000))
        for record in hand.decisions:
            if record.agent_name == agent.name:
                counts[record.action.category] += 1
    return counts


def rate(counts: Counter, *categories: ActionCategory) -> float:
    return sum(counts[c] for c in categories) / sum(counts.values())


RAISES = (C.RAISE_HALF_POT, C.RAISE_POT, C.ALL_IN)

# --- Preflop strength cache ---------------------------------------------------------


@pytest.mark.parametrize(
    ("cards", "expected"),
    [("AsKs", "AKs"), ("KdAc", "AKo"), ("7h7c", "77"), ("2c9c", "92s"), ("Th9d", "T9o")],
)
def test_hand_class(cards: str, expected: str) -> None:
    assert hand_class(tuple(Card.parse(cards))) == expected


def test_preflop_equity_is_cached_and_ordered() -> None:
    first = preflop_equity("AA", 300)
    assert preflop_equity("AA", 300) == first  # same value every time
    assert first > preflop_equity("KK", 300) > preflop_equity("72o", 300)


# --- Random agent ---------------------------------------------------------------------


def test_random_agent_uses_every_category_and_only_legal_ones() -> None:
    counts = action_counts(RandomAgent(), opponent_name="calling_station", hands=200)
    assert set(counts) == set(C)  # simulator rejects anything illegal


# --- Style presets ----------------------------------------------------------------------


def test_presets_exist_and_are_valid() -> None:
    assert set(STYLE_PRESETS) == {"calling_station", "maniac", "nit", "solid"}
    agent = StyleAgent.preset("nit", equity_samples=77)
    assert agent.config() == {
        "class": "StyleAgent",
        "name": "nit",
        "looseness": 0.1,
        "aggression": 0.3,
        "bluff_rate": 0.0,
        "equity_samples": 77,
    }


@pytest.mark.parametrize("bad", [{"looseness": 1.5}, {"aggression": -0.1}, {"bluff_rate": 2.0}])
def test_style_parameters_are_validated(bad: dict) -> None:
    values = {"looseness": 0.5, "aggression": 0.5, "bluff_rate": 0.1} | bad
    with pytest.raises(ValueError):
        StyleParameters(**values)


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown style preset"):
        StyleAgent.preset("shark")


def test_calling_station_never_folds_or_raises() -> None:
    counts = action_counts(StyleAgent.preset("calling_station", FAST_SAMPLES), hands=200)
    assert set(counts) == {C.CHECK_CALL}


@pytest.fixture(scope="module")
def style_counts() -> dict[str, Counter]:
    return {
        name: action_counts(StyleAgent.preset(name, FAST_SAMPLES), hands=300)
        for name in ("maniac", "nit", "solid")
    }


def test_styles_differ_in_aggression(style_counts: dict[str, Counter]) -> None:
    raise_rate = {name: rate(counts, *RAISES) for name, counts in style_counts.items()}
    assert raise_rate["maniac"] > raise_rate["solid"] > raise_rate["nit"]


def test_styles_differ_in_folding(style_counts: dict[str, Counter]) -> None:
    fold_rate = {name: rate(counts, C.FOLD) for name, counts in style_counts.items()}
    assert fold_rate["nit"] > fold_rate["solid"] > fold_rate["maniac"]


def test_nit_never_bluffs() -> None:
    nit = StyleAgent.preset("nit", FAST_SAMPLES)
    # Hopeless river hand, nobody has bet: a nit only checks.
    observation = river_observation("7c2d", "AhKhQd9s4c", pot_size=30, to_call=0)
    choices = {nit.act(observation, random.Random(seed)).category for seed in range(100)}
    assert choices == {C.CHECK_CALL}


# --- Probability agent ----------------------------------------------------------------------


def probability_decision(hole: str, board: str, pot_size: int, to_call: int):
    agent = ProbabilityAgent()
    return agent.act(river_observation(hole, board, pot_size, to_call), random.Random(0))


def test_probability_agent_folds_below_pot_odds() -> None:
    # Exact equity 0.108 < pot odds 0.25 + margin 0.02.
    decision = probability_decision("7c2d", "AhKhQd9s4c", pot_size=30, to_call=10)
    assert decision.category is C.FOLD
    assert decision.reason.startswith("fold: equity 0.1")
    assert decision.equity == pytest.approx(0.108, abs=0.06)


def test_probability_agent_calls_above_pot_odds() -> None:
    # Exact equity 0.620: above 0.27, below the 0.70 raise threshold.
    decision = probability_decision("6c5c", "Ah9c6d4s2h", pot_size=30, to_call=10)
    assert decision.category is C.CHECK_CALL
    assert "pot odds 0.25" in decision.reason


def test_probability_agent_raises_with_the_nuts() -> None:
    decision = probability_decision("AhKh", "QhJhTh2c3d", pot_size=30, to_call=10)
    assert decision.category is C.RAISE_POT
    assert decision.equity == 1.0


def test_probability_agent_bets_strong_hands_and_checks_weak_ones() -> None:
    bet = probability_decision("AhKh", "QhJhTh2c3d", pot_size=30, to_call=0)
    check = probability_decision("7c2d", "AhKhQd9s4c", pot_size=30, to_call=0)
    assert bet.category is C.RAISE_HALF_POT
    assert check.category is C.CHECK_CALL and check.reason.startswith("check")


def test_probability_agent_calls_when_raise_is_impossible() -> None:
    observation = river_observation("AhKh", "QhJhTh2c3d", pot_size=300, to_call=150)
    observation = Observation(
        **{**observation.__dict__, "legal_actions": observation.legal_actions[:2]}
    )
    decision = ProbabilityAgent().act(observation, random.Random(0))
    assert decision.category is C.CHECK_CALL


def test_probability_agent_is_deterministic_for_a_seed() -> None:
    agent = ProbabilityAgent(parameters=ProbabilityParameters(equity_samples=100))
    observation = river_observation("6c5c", "Ah9c6d4s2h", pot_size=30, to_call=10)
    assert agent.act(observation, random.Random(5)) == agent.act(observation, random.Random(5))


# --- Factory ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_make_agent_creates_every_named_agent(name: str) -> None:
    agent = make_agent(name, equity_samples=50)
    assert agent.name == name
    assert agent.config()["name"] == name


def test_make_agent_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown agent"):
        make_agent("superbot")
