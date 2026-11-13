"""Simplified expected-value (EV) equations for the Phase 4 EV agents.

Every EV is the **expected change in the hero's chips from now on,
compared with folding now**. Chips already in the pot are sunk: they are
lost if the hero folds and won back only as part of the pot.

Notation (all in chips, from the hero's ``Observation``):

* ``e``: the hero's equity against the opponent (Phase 1 estimate);
* ``P``: the pot, *including* every bet already made (Phase 1 convention);
* ``c``: the amount the hero must add to call (0 = checking is free);
* ``a``: chips the hero adds with a bet or raise (``LegalAction.chips_added``);
* ``r``: chips the opponent must add to call that bet;
* ``p_fold, p_call, p_raise``: the opponent's predicted responses
  (``p_raise`` = half-pot + pot + all-in raises), legal-masked;
* ``e_c``: the hero's equity **when the bet is called**, against the
  opponent's estimated continuing range (:mod:`hybridpoker.range_equity`;
  the strongest ``p_call + p_raise`` fraction of the opponent's possible
  hands). Without a range estimate, ``e_c = e``.

Equations:

=========================  ====================================================
Action                     EV
=========================  ====================================================
fold                       0
check (c = 0)              e * P
call (c > 0)               e * (P - x + c') - c'   with c' = min(c, stack),
                           x = c - c' returned to the opponent (short call)
bet / raise of a chips     p_fold * P
                           + p_call * [e_c * (P + a' + r') - a']
                           + p_raise * (-a)
                           with r' = min(r, opponent stack) and
                           a' = a - (r - r') (uncalled chips returned)
=========================  ====================================================

**Assumptions (documented simplifications)**

1. *No future betting.* After a check or call, and after a called bet, the
   hand is valued as if it went to showdown with no more chips added.
2. *A re-raise is a loss of the bet.* If the opponent re-raises, the hero is
   assumed to give up and lose the ``a`` chips just added. This is
   conservative: it ignores any chance of calling or winning.
3. *Equity when called.* A called bet is valued with ``e_c``, the hero's
   equity against the opponent's estimated continuing range (strongest-hand
   percentile approximation; see :mod:`hybridpoker.range_equity`). Checks
   and calls of the opponent's bet still use ``e`` (against a random hand),
   like the ProbabilityAgent: the opponent's own earlier bets do not narrow
   its range here.
4. *All-in.* If the hero's bet puts either player all-in, the opponent can
   only fold or call. The opponent's legal-action mask (in the hypothetical
   state) then gives ``p_raise = 0`` automatically.

Both EV agents (frequency prior and ML model) use exactly these equations,
so their difference measures only the value of the opponent predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybridpoker.action_probabilities import CLASS_ORDER
from hybridpoker.actions import RAISE_CATEGORIES, ActionCategory, LegalAction
from hybridpoker.observation import Observation

PROBABILITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ResponseProbabilities:
    """The opponent's predicted response to a hero bet or raise."""

    fold: float
    call: float
    raise_: float

    def __post_init__(self) -> None:
        values = (self.fold, self.call, self.raise_)
        if any(v < -PROBABILITY_TOLERANCE for v in values):
            raise ValueError(f"Response probabilities must be non-negative: {values}.")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"Response probabilities must sum to 1, got {sum(values)}.")

    @classmethod
    def from_action_probabilities(cls, probabilities) -> ResponseProbabilities:
        """Collapse legal-masked probabilities in ``CLASS_ORDER`` into fold / call / raise."""
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(CLASS_ORDER),):
            raise ValueError(f"Expected {len(CLASS_ORDER)} probabilities in CLASS_ORDER, got shape {p.shape}.")
        return cls(fold=float(p[0]), call=float(p[1]), raise_=float(p[2] + p[3] + p[4]))


def _check_equity(equity: float) -> None:
    if not 0.0 <= equity <= 1.0:
        raise ValueError(f"equity must be between 0 and 1, got {equity}.")


def ev_fold() -> float:
    return 0.0


def ev_check(equity: float, pot: float) -> float:
    """Checking when nothing is owed: showdown for the current pot."""
    _check_equity(equity)
    return equity * pot


def ev_call(equity: float, pot: float, to_call: float, own_stack: float) -> float:
    """Calling ``to_call`` (or all-in for less if the stack is short)."""
    _check_equity(equity)
    if to_call <= 0:
        raise ValueError("ev_call needs to_call > 0; use ev_check when checking is free.")
    paid = min(to_call, own_stack)
    returned = to_call - paid  # the opponent's uncalled chips go back to them
    return equity * (pot - returned + paid) - paid


def ev_bet(
    equity: float,
    pot: float,
    chips_added: float,
    own_bet: float,
    opponent_bet: float,
    opponent_stack: float,
    response: ResponseProbabilities,
    called_equity: float | None = None,
) -> float:
    """A bet or raise that adds ``chips_added`` chips; see the module table.

    ``called_equity`` is ``e_c``, the hero's equity when the bet is called
    (defaults to ``equity``, the equity against a random hand).
    """
    _check_equity(equity)
    called_equity = equity if called_equity is None else called_equity
    _check_equity(called_equity)
    if chips_added <= 0:
        raise ValueError("A bet or raise must add chips.")
    new_bet = own_bet + chips_added
    to_match = new_bet - opponent_bet
    if to_match <= 0:
        raise ValueError("A bet or raise must exceed the opponent's bet.")
    opponent_call = min(to_match, opponent_stack)
    committed = chips_added - (to_match - opponent_call)  # uncalled chips come back
    called = called_equity * (pot + committed + opponent_call) - committed
    return response.fold * pot + response.call * called + response.raise_ * (-chips_added)


@dataclass(frozen=True)
class ActionValue:
    """EV of one legal action, with the parts that explain it."""

    category: ActionCategory
    ev: float
    explanation: str


def action_value(
    observation: Observation,
    action: LegalAction,
    equity: float,
    response: ResponseProbabilities | None,
    called_equity: float | None = None,
) -> ActionValue:
    """EV of a legal hero action. ``response`` (and optionally ``called_equity``)
    apply to bets and raises only."""
    o = observation
    if action.category is ActionCategory.FOLD:
        return ActionValue(action.category, ev_fold(), "fold: 0")
    if action.category is ActionCategory.CHECK_CALL:
        if o.to_call == 0:
            ev = ev_check(equity, o.pot_size)
            return ActionValue(action.category, ev, f"check: {equity:.2f} x pot {o.pot_size} = {ev:+.2f}")
        ev = ev_call(equity, o.pot_size, o.to_call, o.own_stack)
        return ActionValue(action.category, ev,
                           f"call {o.to_call}: {equity:.2f} x {o.pot_size + o.to_call} - {o.to_call} = {ev:+.2f}")
    if action.category in RAISE_CATEGORIES:
        if response is None:
            raise ValueError("A bet or raise needs the opponent's predicted response.")
        ev = ev_bet(equity, o.pot_size, action.chips_added, o.own_bet, o.opponent_bet, o.opponent_stack, response,
                    called_equity)
        called = equity if called_equity is None else called_equity
        return ActionValue(
            action.category, ev,
            f"{action.category.value} to {action.bet_to}: P(fold) {response.fold:.2f}, P(call) {response.call:.2f}, "
            f"P(raise) {response.raise_:.2f}, equity if called {called:.2f} -> {ev:+.2f}",
        )
    raise ValueError(f"Unknown action {action.category}.")
