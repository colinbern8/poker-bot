"""Regression: Fold is offered exactly when the human owes chips (Phase 5E investigation).

A manual test reported that "Fold sometimes disappears". The investigation found
this is the documented action rule (``hybridpoker.actions.legal_actions``): Fold is
not offered when checking is free, and is always offered when facing a bet,
raise or all-in. These tests pin that rule in the post-flop spots that earlier
tests did not cover. The matching screen test is in ``test_demo_app.py``.
"""

from __future__ import annotations

from demo_support import SAMPLES, ScriptedAggressorEV
from hybridpoker.actions import RAISE_CATEGORIES
from hybridpoker.actions import ActionCategory as C
from hybridpoker.agents.ev_agent import MLResponseModel
from hybridpoker.observation import Observation
from hybridpoker.opponent_predictor import OpponentPredictor
from hybridpoker_demo.controller import HeadsUpController, HumanTurn


def spot(obs: Observation) -> str:
    """The kind of decision the human faces: "free" (checking is free), "all-in", "bet", "raise" or "blind"."""
    if obs.to_call == 0:
        return "free"
    if obs.history and obs.history[-1].is_all_in:
        return "all-in"
    aggressive = sum(a.street == obs.street and a.category in RAISE_CATEGORIES for a in obs.history)
    if obs.street == "preflop" and aggressive == 0:
        return "blind"
    return "bet" if aggressive == 1 else "raise"


def human_choice(obs: Observation, hand_index: int) -> C:
    """A fixed human script against ScriptedAggressorEV that reaches every post-flop spot."""
    kind = spot(obs)
    if obs.street == "flop" and kind in ("free", "bet"):
        return C.RAISE_HALF_POT  # bet when checked to; raise a bet (the AI then re-raises)
    if kind == "all-in":
        return C.FOLD if hand_index % 2 == 0 else C.CHECK_CALL
    return C.CHECK_CALL


def assert_fold_rule(obs: Observation) -> str:
    kind = spot(obs)
    legal = obs.legal_categories
    if kind == "free":
        assert C.FOLD not in legal and C.CHECK_CALL in legal, legal
    elif kind == "all-in":
        assert legal == (C.FOLD, C.CHECK_CALL), legal
    else:
        assert legal[0] is C.FOLD and C.CHECK_CALL in legal, legal
        assert obs.legal(C.FOLD).chips_added == 0
    return kind


def test_fold_is_offered_exactly_when_the_human_owes_chips(tiny_bundle) -> None:
    ai = ScriptedAggressorEV("ev_ml", MLResponseModel(OpponentPredictor(tiny_bundle)), SAMPLES)
    controller = HeadsUpController(ai, match_seed=11)
    seen = set()
    for hand_index in range(2):  # the button alternates, so the human plays both seats
        if hand_index:
            controller.next_hand()
        while isinstance(controller.view, HumanTurn):
            obs = controller.view.observation
            seen.add((assert_fold_rule(obs), obs.street != "preflop"))
            controller.submit(controller.view.ticket, human_choice(obs, hand_index))
    # Facing a post-flop bet, raise and all-in, and a free post-flop check, were all reached.
    assert {("bet", True), ("raise", True), ("all-in", True), ("free", True)} <= seen, seen
