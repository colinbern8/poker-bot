"""Failed or rejected operations leave the controller and the AI exactly as they were.

A snapshot covers the controller's committed state (phase, view, hand,
human actions, AI records, finished hands, full results, score) and the AI's
state (diagnostics list and opponent tracker rows and features). Rejected
inputs and injected failures must leave the snapshot unchanged, and a retry
must produce the same match as an uninterrupted reference match.
"""

from __future__ import annotations

import copy

import pytest

from demo_support import (
    assert_same_diagnostics,
    assert_same_hand,
    human_categories,
    ml_agent,
    reference_match,
    tracker_state,
    without_timing,
)
from hybridpoker.actions import ActionCategory
from hybridpoker.agents.ev_agent import EVAgent
from hybridpoker_demo.controller import (
    CompletedHand,
    HeadsUpController,
    HumanTurn,
    IllegalHumanAction,
    StaleTicket,
    Ticket,
    TrackerMismatch,
    WrongPhase,
)

SEED = 31
HANDS = 8


class InjectedFailure(RuntimeError):
    pass


class FlakyDecisionEV(EVAgent):
    """Fails after computing a decision (its diagnostics are already appended) while armed."""

    armed = False

    def act(self, observation, rng):
        decision = super().act(observation, rng)
        if self.armed:
            raise InjectedFailure("AI failed after deciding")
        return decision


class FlakyTrackerEV(EVAgent):
    """Fails after the tracker has already recorded the hand while armed."""

    armed = False

    def observe_hand_end(self, summary) -> None:
        super().observe_hand_end(summary)
        if self.armed:
            raise InjectedFailure("tracker update failed halfway")


class ForgetfulEV(EVAgent):
    """Never records the opponent: the controller's tracker check must catch it."""

    def observe_hand_end(self, summary) -> None:
        return None


def snapshot(controller: HeadsUpController, ai) -> str:
    state = {
        "phase": controller.phase, "view": controller.view, "hand": controller.hand_index,
        "actions": controller._human_actions, "records": controller._records,
        "completed": controller.completed_hands, "results": tuple(controller._results),
        "score": controller.scoreboard, "diagnostics": [without_timing(d) for d in ai.diagnostics],
        "tracker": tracker_state(ai),
    }
    return repr(copy.deepcopy(state))  # repr: exact, and NaN-safe


def drive(controller, ai, expected, fail_first_attempt: bool):
    """Play the reference actions; optionally arm a failure before every operation and retry once."""
    failures = 0

    def attempt(operation):
        nonlocal failures
        if fail_first_attempt:
            before = snapshot(controller, ai)
            ai.armed = True
            try:
                operation()
            except InjectedFailure:
                failures += 1
                assert snapshot(controller, ai) == before  # nothing changed
            else:
                return  # the operation needed no AI work of the armed kind
            finally:
                ai.armed = False
        operation()

    for h, ref in enumerate(expected):
        if h:
            attempt(controller.next_hand)
        actions = human_categories(ref["result"], ref["setup"])
        for category in actions:
            ticket = controller.view.ticket
            attempt(lambda: controller.submit(ticket, category))
        assert isinstance(controller.view, CompletedHand)
    return failures


@pytest.mark.parametrize("flaky", [FlakyDecisionEV, FlakyTrackerEV])
def test_injected_failures_change_nothing_and_retries_reproduce_the_reference(tiny_bundle, flaky) -> None:
    reference_ai = ml_agent(tiny_bundle)
    expected = reference_match(reference_ai, "maniac", SEED, HANDS)
    ai = ml_agent(tiny_bundle, flaky)
    controller = HeadsUpController(ai, match_seed=SEED)
    failures = drive(controller, ai, expected, fail_first_attempt=True)
    assert failures >= HANDS - 1  # a failure was injected into (nearly) every hand
    for h, ref in enumerate(expected):
        assert_same_hand(controller._results[h], ref["result"], ref["setup"].ai_seat)
    assert tracker_state(ai) == expected[-1]["tracker"]
    assert_same_diagnostics(ai.diagnostics, reference_ai.diagnostics)
    assert_same_diagnostics([r.diagnostics for h in controller.completed_hands for r in h.ai_decisions],
                            reference_ai.diagnostics)


def test_a_failure_while_the_ai_starts_a_hand_changes_nothing(tiny_bundle) -> None:
    """The AI acts first when it is on the button (odd hands); a failure there is also rolled back."""
    ai = ml_agent(tiny_bundle, FlakyDecisionEV)
    controller = HeadsUpController(ai, match_seed=SEED)
    while isinstance(controller.view, HumanTurn):
        controller.submit(controller.view.ticket, controller.view.legal_categories[0])
    before = snapshot(controller, ai)
    ai.armed = True
    with pytest.raises(InjectedFailure):
        controller.next_hand()
    ai.armed = False
    assert snapshot(controller, ai) == before
    assert controller.next_hand() is controller.view


def test_stale_duplicate_and_foreign_tickets_are_rejected(tiny_bundle) -> None:
    ai = ml_agent(tiny_bundle)
    controller = HeadsUpController(ai, match_seed=SEED)
    first = controller.view
    controller.submit(first.ticket, first.legal_categories[1])
    for bad in (first.ticket,                                            # duplicate / double click
                Ticket("another-match", controller.hand_index, 0),     # other match
                Ticket(controller.match_id, controller.hand_index + 1, 0)):  # wrong hand
        before = snapshot(controller, ai)
        with pytest.raises((StaleTicket, WrongPhase)):
            controller.submit(bad, ActionCategory.CHECK_CALL)
        assert snapshot(controller, ai) == before


def test_illegal_and_unknown_actions_are_rejected(tiny_bundle) -> None:
    ai = ml_agent(tiny_bundle)
    controller = HeadsUpController(ai, match_seed=SEED)
    checked_fold = False
    for _ in range(30):
        if not isinstance(controller.view, HumanTurn):
            controller.next_hand()
            continue
        view = controller.view
        for bad in ("bet_big", 3, None):
            before = snapshot(controller, ai)
            with pytest.raises(IllegalHumanAction):
                controller.submit(view.ticket, bad)
            assert snapshot(controller, ai) == before
        if view.observation.to_call == 0:  # folding is never offered when checking is free
            assert ActionCategory.FOLD not in view.legal_categories
            before = snapshot(controller, ai)
            with pytest.raises(IllegalHumanAction):
                controller.submit(view.ticket, ActionCategory.FOLD)
            assert snapshot(controller, ai) == before
            checked_fold = True
        controller.submit(view.ticket, "check_call")  # string values are accepted
    assert checked_fold


def test_operations_in_the_wrong_phase_are_rejected(tiny_bundle) -> None:
    ai = ml_agent(tiny_bundle)
    controller = HeadsUpController(ai, match_seed=SEED)
    before = snapshot(controller, ai)
    with pytest.raises(WrongPhase):
        controller.next_hand()  # the hand is still running
    assert snapshot(controller, ai) == before
    while isinstance(controller.view, HumanTurn):
        last = controller.view.ticket
        controller.submit(last, controller.view.legal_categories[-1])
    before = snapshot(controller, ai)
    with pytest.raises(WrongPhase):
        controller.submit(last, ActionCategory.CHECK_CALL)  # the hand is over
    assert snapshot(controller, ai) == before


def test_tracker_check_catches_an_ai_that_did_not_record_the_human(tiny_bundle) -> None:
    ai = ml_agent(tiny_bundle, ForgetfulEV)
    controller = HeadsUpController(ai, match_seed=SEED)
    with pytest.raises(TrackerMismatch):
        while isinstance(controller.view, HumanTurn):
            before = snapshot(controller, ai)
            controller.submit(controller.view.ticket, controller.view.legal_categories[-1])
    assert snapshot(controller, ai) == before
    assert controller.completed_hands == ()
