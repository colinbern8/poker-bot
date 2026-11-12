"""Pot odds and simplified call expected value (EV).

Definitions used throughout the project:

* ``pot_size``: every chip currently in the middle, **including** the
  opponent's bet that the hero is facing, but **excluding** the chips the
  hero would add by calling.
* ``call_amount``: the additional chips the hero must put in to call.
  It can never exceed ``pot_size``, because the pot already contains the
  bet being called; violating this raises ``ValueError``.

Example: the pot was 20, the opponent bets 10, so ``pot_size`` is 30 and
``call_amount`` is 10. Calling makes the final pot 40.

Assumptions behind these formulas (they are deliberate simplifications):

1. The call ends the betting: no more chips go in after it (e.g. the
   opponent is all-in, or it is the last betting round). Future bets,
   raises, and bluffs are ignored.
2. ``equity`` is the hero's average share of the final pot at showdown,
   including fractional shares for split pots (see ``equity.py``).
3. The pot is contested heads-up: after the call, only the hero and the
   opponent who bet share the final pot of ``pot_size + call_amount``.
4. There is no rake.
5. EV is measured relative to folding. Chips the hero already put in the
   pot are sunk: they are lost by folding, so they do not count against
   calling.
6. ``call_amount <= pot_size`` (enforced). Under the definitions above the
   pot already holds the opponent's bet, and the hero never has to add
   more than that bet, so no real spot breaks this rule. A larger call
   almost always means the opponent's bet was left out of ``pot_size``,
   so it raises ``ValueError`` instead of returning a wrong answer.
   Consequence: pot odds are never above 0.5.
"""

from __future__ import annotations

import math


def _validate_amount(name: str, value: object) -> float:
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return float(value)


def _validate_pot_and_call(pot_size: object, call_amount: object) -> tuple[float, float]:
    pot = _validate_amount("pot_size", pot_size)
    call = _validate_amount("call_amount", call_amount)
    if pot <= 0:
        raise ValueError(f"pot_size must be positive, got {pot_size}.")
    if call < 0:
        raise ValueError(f"call_amount cannot be negative, got {call_amount}.")
    if call > pot:
        raise ValueError(
            f"call_amount ({call_amount}) cannot exceed pot_size ({pot_size}): "
            "pot_size must already include the bet being called."
        )
    return pot, call


def pot_odds(pot_size: float, call_amount: float) -> float:
    """Return the minimum equity needed for a call to break even.

    pot_odds = call_amount / (pot_size + call_amount)

    That is, the price of calling divided by the size of the final pot.
    Example: pot_size=30, call_amount=10 gives 10 / 40 = 0.25, so the hero
    must win at least 25% of the pot on average for calling to be
    profitable. A call_amount of 0 (a free check) gives 0.0.
    """
    pot, call = _validate_pot_and_call(pot_size, call_amount)
    return call / (pot + call)


def call_expected_value(equity: float, pot_size: float, call_amount: float) -> float:
    """Return the simplified expected chip gain of calling versus folding.

    EV(call) = equity * (pot_size + call_amount) - call_amount

    The hero pays ``call_amount`` and gets back, on average, ``equity``
    times the final pot. This equals ``equity * pot_size - (1 - equity) *
    call_amount``. EV is positive exactly when equity exceeds
    :func:`pot_odds`, and zero when they are equal.

    See the module docstring for the assumptions this relies on.
    """
    pot, call = _validate_pot_and_call(pot_size, call_amount)
    hero_equity = _validate_amount("equity", equity)
    if not 0.0 <= hero_equity <= 1.0:
        raise ValueError(f"equity must be between 0 and 1, got {equity}.")
    return hero_equity * (pot + call) - call
