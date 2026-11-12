"""Tests for hybridpoker.opponent_configs (the 16 opponent configurations)."""

import pytest

from hybridpoker.agents.simple import STYLE_PRESETS, StyleAgent
from hybridpoker.opponent_configs import (
    FAMILY_ORDER,
    HELDOUT_POOL,
    MAX_SHIFT,
    PARAMETER_NAMES,
    TRAIN_POOL,
    OpponentConfig,
    build_configs,
    heldout_variants,
    variant_parameters,
)


@pytest.fixture(scope="module")
def configs():
    return build_configs(3000)


def test_sixteen_configs_four_per_family(configs) -> None:
    assert len(configs) == 16
    assert len({c.name for c in configs}) == 16
    for family in FAMILY_ORDER:
        members = [c for c in configs if c.family == family]
        assert [c.variant for c in members] == [0, 1, 2, 3]
        assert [c.name for c in members] == [family, f"{family}_v1", f"{family}_v2", f"{family}_v3"]


def test_presets_are_unchanged(configs) -> None:
    for c in configs:
        if c.variant == 0:
            preset = STYLE_PRESETS[c.family]
            assert (c.looseness, c.aggression, c.bluff_rate) == (
                preset.looseness, preset.aggression, preset.bluff_rate
            )
            assert c.pool == TRAIN_POOL


def test_variants_stay_within_shift_and_bounds(configs) -> None:
    for c in configs:
        preset = STYLE_PRESETS[c.family]
        for name in PARAMETER_NAMES:
            value = getattr(c, name)
            assert 0.0 <= value <= 1.0
            assert abs(value - getattr(preset, name)) <= MAX_SHIFT + 1e-9
            assert value == round(value, 3)
        assert c.equity_samples == 200


def test_exactly_one_heldout_variant_per_family(configs) -> None:
    heldout = [c for c in configs if c.pool == HELDOUT_POOL]
    assert len(heldout) == 4
    assert sorted(c.family for c in heldout) == sorted(FAMILY_ORDER)
    assert all(c.variant in (1, 2, 3) for c in heldout)
    assert sum(c.pool == TRAIN_POOL for c in configs) == 12


def test_configs_are_reproducible_and_seed_dependent() -> None:
    assert build_configs(3000) == build_configs(3000)
    assert variant_parameters("maniac", 2, 3000) != variant_parameters("maniac", 2, 3001)
    assert heldout_variants(3000) == heldout_variants(3000)


def test_config_builds_matching_agent_and_round_trips(configs) -> None:
    config = configs[5]
    agent = config.make_agent()
    assert isinstance(agent, StyleAgent)
    assert agent.name == config.name
    assert agent.parameters.looseness == config.looseness
    assert OpponentConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("family, variant", [("unknown", 1), ("nit", 0), ("nit", 4)])
def test_variant_parameters_validate_input(family, variant) -> None:
    with pytest.raises(ValueError):
        variant_parameters(family, variant)
