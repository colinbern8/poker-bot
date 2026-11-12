"""Win-rate statistics: bb/100 with a 95% confidence interval.

**bb/100** is the average number of big blinds won per 100 hands, the
standard poker win-rate unit.

The confidence interval must be computed from *independent* units. In a
seat-swapped (duplicate) match, the two hands of one deal share the same
cards, so their results are correlated: if the hero wins big with the
good cards in one seat, the opponent tends to win big with them in the
other. Treating those two hands as independent would give a misleading
interval. So the unit is the **deal pair**:

1. For each deal, add up the hero's net chips over both seatings.
2. Convert each pair to bb/100: ``pair_net / 2 hands / big_blind * 100``.
3. bb/100 = the mean over pairs; standard error = sample standard
   deviation / sqrt(number of pairs).
4. 95% CI = mean +/- 1.96 x standard error (normal approximation, which
   is reasonable for many pairs; with fewer than about 30 units the
   interval is too narrow).

Without seat swapping, every hand is dealt from its own seed with fresh
stacks, so hands are independent and the unit is the single hand.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class WinRate:
    """A bb/100 estimate with its uncertainty.

    Attributes:
        bb_per_100: Mean big blinds won per 100 hands.
        standard_error: Standard error of ``bb_per_100``.
        ci_low, ci_high: Bounds of the confidence interval.
        confidence: Confidence level, e.g. 0.95.
        unit_count: Number of independent units (deal pairs or hands).
        hands_per_unit: 2 for seat-swapped pairs, 1 for single hands.
    """

    bb_per_100: float
    standard_error: float
    ci_low: float
    ci_high: float
    confidence: float
    unit_count: int
    hands_per_unit: int

    @property
    def hands(self) -> int:
        return self.unit_count * self.hands_per_unit

    def __str__(self) -> str:
        return (
            f"{self.bb_per_100:+.1f} bb/100 "
            f"[{self.ci_low:+.1f}, {self.ci_high:+.1f}] "
            f"({self.confidence:.0%} CI, {self.unit_count} units x {self.hands_per_unit} hands)"
        )


def win_rate(
    unit_net_chips: Sequence[float],
    hands_per_unit: int,
    big_blind: int,
    confidence: float = 0.95,
) -> WinRate:
    """Compute bb/100 and its confidence interval from independent units.

    Args:
        unit_net_chips: Net chips won by the hero in each independent unit
            (a deal pair, or a single hand).
        hands_per_unit: Hands in each unit (2 for a seat-swapped pair).
        big_blind: Big blind size in chips.
        confidence: Confidence level of the interval.
    """
    n = len(unit_net_chips)
    if n < 2:
        raise ValueError("At least 2 units are needed for a confidence interval.")
    if hands_per_unit < 1 or big_blind <= 0:
        raise ValueError("hands_per_unit and big_blind must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    values = [net / hands_per_unit / big_blind * 100 for net in unit_net_chips]
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    standard_error = math.sqrt(variance / n)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    return WinRate(
        bb_per_100=mean,
        standard_error=standard_error,
        ci_low=mean - z * standard_error,
        ci_high=mean + z * standard_error,
        confidence=confidence,
        unit_count=n,
        hands_per_unit=hands_per_unit,
    )
