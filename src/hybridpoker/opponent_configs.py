"""The 16 opponent configurations used for the Phase 3 dataset.

Each of the four ``StyleAgent`` presets gets three seeded *variants* whose
looseness, aggression and bluff rate are shifted by up to +/-0.10. One
variant per preset is held out of training entirely, so the models can be
tested on opponents they have never seen.

Everything is derived from one master seed with ``derive_seed``, so the
same seed always produces the same configurations on every computer.

Configuration names and parameters are *private*: they go into the dataset
manifest and run metadata, never into model input features.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from hybridpoker.agents.simple import STYLE_PRESETS, StyleAgent, StyleParameters
from hybridpoker.seeding import derive_seed

MASTER_SEED = 3000
FAMILY_ORDER = ("calling_station", "maniac", "nit", "solid")
PARAMETER_NAMES = ("looseness", "aggression", "bluff_rate")
VARIANT_COUNT = 3
MAX_SHIFT = 0.10
EQUITY_SAMPLES = 200
TRAIN_POOL = "train"
HELDOUT_POOL = "heldout"


@dataclass(frozen=True)
class OpponentConfig:
    """One opponent configuration.

    Attributes:
        name: Unique name, e.g. ``maniac`` (preset) or ``maniac_v2``.
        family: The preset it is based on (its "style").
        variant: 0 for the preset itself, 1-3 for variants.
        pool: ``"train"`` (may appear in training data) or ``"heldout"``.
    """

    name: str
    family: str
    variant: int
    looseness: float
    aggression: float
    bluff_rate: float
    equity_samples: int
    pool: str

    def parameters(self) -> StyleParameters:
        return StyleParameters(
            looseness=self.looseness,
            aggression=self.aggression,
            bluff_rate=self.bluff_rate,
            equity_samples=self.equity_samples,
        )

    def make_agent(self) -> StyleAgent:
        return StyleAgent(self.name, self.parameters())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpponentConfig:
        return cls(**data)


def variant_parameters(
    family: str, variant: int, master_seed: int = MASTER_SEED
) -> tuple[float, float, float]:
    """Shifted (looseness, aggression, bluff_rate) for one variant.

    Offsets are drawn uniformly from [-MAX_SHIFT, +MAX_SHIFT] in the fixed
    order of ``PARAMETER_NAMES``, added to the preset, clipped to [0, 1]
    and rounded to 3 decimals.
    """
    if family not in STYLE_PRESETS:
        raise ValueError(f"Unknown preset {family!r}.")
    if not 1 <= variant <= VARIANT_COUNT:
        raise ValueError(f"variant must be 1..{VARIANT_COUNT}, got {variant}.")
    preset = STYLE_PRESETS[family]
    rng = random.Random(derive_seed(master_seed, "variant", family, variant))
    values = []
    for name in PARAMETER_NAMES:
        shifted = getattr(preset, name) + rng.uniform(-MAX_SHIFT, MAX_SHIFT)
        values.append(round(min(max(shifted, 0.0), 1.0), 3))
    return values[0], values[1], values[2]


def heldout_variants(master_seed: int = MASTER_SEED) -> dict[str, int]:
    """Which variant (1-3) of each preset is held out of training."""
    rng = random.Random(derive_seed(master_seed, "heldout"))
    return {family: rng.randint(1, VARIANT_COUNT) for family in FAMILY_ORDER}


def build_configs(master_seed: int = MASTER_SEED) -> tuple[OpponentConfig, ...]:
    """All 16 configurations: for each family, the preset then variants 1-3."""
    heldout = heldout_variants(master_seed)
    configs = []
    for family in FAMILY_ORDER:
        preset = STYLE_PRESETS[family]
        configs.append(
            OpponentConfig(
                name=family,
                family=family,
                variant=0,
                looseness=preset.looseness,
                aggression=preset.aggression,
                bluff_rate=preset.bluff_rate,
                equity_samples=EQUITY_SAMPLES,
                pool=TRAIN_POOL,
            )
        )
        for variant in range(1, VARIANT_COUNT + 1):
            looseness, aggression, bluff_rate = variant_parameters(family, variant, master_seed)
            configs.append(
                OpponentConfig(
                    name=f"{family}_v{variant}",
                    family=family,
                    variant=variant,
                    looseness=looseness,
                    aggression=aggression,
                    bluff_rate=bluff_rate,
                    equity_samples=EQUITY_SAMPLES,
                    pool=HELDOUT_POOL if heldout[family] == variant else TRAIN_POOL,
                )
            )
    return tuple(configs)
