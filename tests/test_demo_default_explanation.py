"""The default (short) explanation of an AI decision: wording, comparison values, and warnings.

Most tests build a decision report by hand, so each case (fold, check, call, bet,
raise, all-in, ties, missing or inconsistent diagnostics) is exact. The last test
checks every decision of real played matches (small test model) against its
recorded diagnostics.
"""

from __future__ import annotations

import re

import pytest

from demo_support import human_categories, ml_agent, play_controller_hand, reference_match
from hybridpoker.actions import RAISE_CATEGORIES, LegalAction
from hybridpoker.actions import ActionCategory as C
from hybridpoker.agents.ev_agent import DecisionDiagnostics
from hybridpoker.opponent_tracker import decisions_from_history
from hybridpoker_demo import presentation as ui
from hybridpoker_demo.controller import AIDecisionReport, HeadsUpController

L = LegalAction
# Flop, the AI (big blind) faces a bet of 4 into a pot of 8, with 196 chips behind.
FACING = (L(C.FOLD, 0, 0, False), L(C.CHECK_CALL, 4, 4, False), L(C.RAISE_HALF_POT, 12, 12, False),
          L(C.RAISE_POT, 20, 20, False), L(C.ALL_IN, 196, 196, True))
# Flop, nothing to call, pot 8.
UNOPENED = (L(C.CHECK_CALL, 0, 0, False), L(C.RAISE_HALF_POT, 4, 4, False), L(C.RAISE_POT, 8, 8, False),
            L(C.ALL_IN, 196, 196, True))
# Preflop, the AI on the button facing the big blind.
PREFLOP = (L(C.FOLD, 1, 0, False), L(C.CHECK_CALL, 2, 1, False), L(C.RAISE_HALF_POT, 4, 3, False),
           L(C.RAISE_POT, 6, 5, False), L(C.ALL_IN, 200, 199, True))
# Facing a large bet: the pot-size raise is clamped to all-in, like the all-in itself.
CLAMPED = (L(C.FOLD, 0, 0, False), L(C.CHECK_CALL, 60, 60, False), L(C.RAISE_HALF_POT, 150, 150, False),
           L(C.RAISE_POT, 196, 196, True), L(C.ALL_IN, 196, 196, True))
# Facing an all-in: fold or call all-in.
SHOVED = (L(C.FOLD, 2, 0, False), L(C.CHECK_CALL, 200, 198, True))
ONLY = (L(C.CHECK_CALL, 0, 0, False),)  # a single legal action (a defensive case)

PROBS = [0.42, 0.40, 0.10, 0.05, 0.03]  # fold, call, raise ½ pot, raise pot, all-in
TO_CALL = {FACING: 4, UNOPENED: 0, PREFLOP: 1, CLAMPED: 60, SHOVED: 198, ONLY: 0}
STREET = {PREFLOP: "preflop", SHOVED: "preflop"}


def report(legal, category, evs, *, chosen=None, probs=True, low=(), diagnostics=True, drop_called=False):
    """A decision report as the controller builds it, with diagnostics like the agent records them."""
    street = STREET.get(legal, "flop")
    action = next(a for a in legal if a.category is category)
    d = None
    if diagnostics:
        d = DecisionDiagnostics(street=street, position="big_blind", equity=0.61, action_evs=dict(evs),
                                chosen=chosen or category.value, reason="recorded", total_seconds=0.3)
        for a in legal:
            if a.category in RAISE_CATEGORIES:
                name = a.category.value
                d.coverage[name] = {"key": "k", "training_rows": 3 if name in low else 900, "low": name in low}
                if probs:
                    d.opponent_probabilities[name] = list(PROBS)
                    d.continue_fraction[name] = 0.58
                    if not drop_called:
                        d.called_equity[name] = 0.47
    return AIDecisionReport(street, "big_blind", category, action.bet_to, action.chips_added, TO_CALL[legal],
                            legal, "recorded", 0.61, d)


def evs(**values: float) -> dict[str, float]:
    names = {"fold": "fold", "call": "check_call", "half": "raise_half_pot", "pot": "raise_pot", "allin": "all_in"}
    return {names[k]: v for k, v in values.items()}


ALL_RAISES_LOW = ("raise_half_pot", "raise_pot", "all_in")


# --- actions ----------------------------------------------------------------------------------------------


def test_call_names_the_chips_and_compares_with_the_next_best_option() -> None:
    e = ui.default_explanation(report(FACING, C.CHECK_CALL, evs(fold=0.0, call=1.3, half=-0.4, pot=-1.1, allin=-3.0),
                                      low=ALL_RAISES_LOW))
    assert e.headline == "AI calls 4 chips."
    assert e.reason == "Calling had the highest estimated value of the 5 available options."
    assert e.comparison == "Estimated value: Call +1.3 chips | Fold 0 chips"
    # Low coverage of the raises it did not choose is no reason to flag a call; no predictions for a call.
    assert e.warning is None and e.response is None
    assert (e.status, e.details) == ("ok", True)


def test_fold() -> None:
    e = ui.default_explanation(report(FACING, C.FOLD, evs(fold=0.0, call=-2.5, half=-3.0, pot=-4.0, allin=-6.0),
                                      low=ALL_RAISES_LOW))
    assert e.headline == "AI folds."
    assert e.reason == "Folding had the highest estimated value of the 5 available options."
    assert e.comparison == "Estimated value: Fold 0 chips | Call −2.5 chips"
    assert e.warning is None and e.response is None


def test_check_is_not_flagged_for_low_coverage_of_unchosen_bets() -> None:
    e = ui.default_explanation(report(UNOPENED, C.CHECK_CALL, evs(call=0.8, half=0.1, pot=-0.2, allin=-5.0),
                                      low=ALL_RAISES_LOW))
    assert e.headline == "AI checks."
    assert e.reason == "Checking had the highest estimated value of the 4 available options."
    assert e.comparison == "Estimated value: Check +0.8 chips | Bet ½ pot +0.1 chips"
    assert e.warning is None and e.response is None


def test_bet_with_prediction_and_low_coverage_warning_for_the_chosen_bet() -> None:
    e = ui.default_explanation(report(UNOPENED, C.RAISE_HALF_POT, evs(call=0.2, half=1.6, pot=1.1, allin=-2.0),
                                      low=("raise_half_pot",)))
    assert e.headline == "AI bets 4 chips."
    assert e.reason == "Betting half the pot had the highest estimated value of the 4 available options."
    assert e.response == "The model estimated you would fold about 42% of the time to this bet."
    assert e.comparison == "Estimated value: Bet ½ pot +1.6 chips | Bet pot +1.1 chips"
    assert e.warning == "Few similar training examples for this bet, so the predicted response is less certain."


def test_bet_without_low_coverage_has_no_warning() -> None:
    e = ui.default_explanation(report(UNOPENED, C.RAISE_HALF_POT, evs(call=0.2, half=1.6, pot=1.1, allin=-2.0),
                                      low=("raise_pot", "all_in")))
    assert e.warning is None and e.response is not None


def test_raise_and_all_in() -> None:
    e = ui.default_explanation(report(PREFLOP, C.RAISE_POT, evs(fold=0.0, call=0.4, half=0.9, pot=1.5, allin=-1.0)))
    assert e.headline == "AI raises to 6 chips."
    assert e.reason == "Raising the pot had the highest estimated value of the 5 available options."
    assert e.response == "The model estimated you would fold about 42% of the time to this raise."
    assert e.comparison == "Estimated value: Raise pot +1.5 chips | Raise ½ pot +0.9 chips"

    e = ui.default_explanation(report(FACING, C.ALL_IN, evs(fold=0.0, call=2.0, half=1.0, pot=1.0, allin=7.25),
                                      low=("all_in",)))
    assert e.headline == "AI goes all-in for 196 chips."
    assert e.reason == "Going all-in had the highest estimated value of the 5 available options."
    assert e.response == "The model estimated you would fold about 42% of the time to this all-in."
    assert e.comparison == "Estimated value: All-in +7.2 chips | Call +2.0 chips"
    assert e.warning == "Few similar training examples for this all-in, so the predicted response is less certain."


def test_call_all_in() -> None:
    e = ui.default_explanation(report(SHOVED, C.CHECK_CALL, evs(fold=0.0, call=12.0)))
    assert e.headline == "AI calls all-in for 198 chips."
    assert e.reason == "Calling had the highest estimated value of the 2 available options."
    assert e.comparison == "Estimated value: Call +12.0 chips | Fold 0 chips"


# --- ties, near ties and rounding -------------------------------------------------------------------------


def test_exact_tie_is_explained_by_the_tie_rule() -> None:
    e = ui.default_explanation(report(UNOPENED, C.CHECK_CALL, evs(call=0.5, half=0.5, pot=-0.2, allin=-5.0)))
    assert e.reason == ("Checking tied with betting half the pot for the highest estimated value; "
                        "on a tie the AI takes the option that adds fewer chips.")
    assert e.comparison == "Estimated value: Check +0.5 chips | Bet ½ pot +0.5 chips"

    # Within the agent's 1e-9 tolerance counts as the same tie.
    e = ui.default_explanation(report(UNOPENED, C.CHECK_CALL, evs(call=0.5, half=0.5 + 1e-10, pot=-0.2, allin=-5.0)))
    assert e.reason.startswith("Checking tied with betting half the pot")


def test_tie_between_equal_chip_amounts() -> None:
    e = ui.default_explanation(report(CLAMPED, C.RAISE_POT, evs(fold=0.0, call=1.0, half=2.0, pot=3.0, allin=3.0)))
    assert e.headline == "AI goes all-in for 196 chips."
    assert e.reason == ("Raising the pot (all-in) tied with going all-in for the highest estimated value; "
                        "the AI's fixed tie rule chose it.")
    assert e.comparison == "Estimated value: Raise pot (all-in) +3.0 chips | All-in +3.0 chips"


def test_near_tie_wording_uses_a_half_chip_threshold() -> None:
    e = ui.default_explanation(report(FACING, C.CHECK_CALL, evs(fold=0.0, call=0.3, half=-1.0, pot=-2.0, allin=-4.0)))
    assert e.reason == "Calling had the highest estimated value, only slightly ahead of folding."
    assert e.comparison == "Estimated value: Call +0.3 chips | Fold 0 chips"
    e = ui.default_explanation(report(FACING, C.CHECK_CALL, evs(fold=0.0, call=0.5, half=-1.0, pot=-2.0, allin=-4.0)))
    assert e.reason == "Calling had the highest estimated value of the 5 available options."


@pytest.mark.parametrize("values, comparison", [
    (evs(fold=0.0, call=0.04, half=-1.0, pot=-2.0, allin=-4.0), "Call +0.04 chips | Fold 0 chips"),
    (evs(fold=-1.0, call=1.23, half=1.21, pot=-2.0, allin=-4.0), "Call +1.23 chips | Raise ½ pot +1.21 chips"),
    (evs(fold=0.0, call=1.25, half=1.2, pot=-2.0, allin=-4.0), "Call +1.25 chips | Raise ½ pot +1.20 chips"),
    (evs(fold=-1.0, call=1.0000005, half=1.0, pot=-2.0, allin=-4.0),
     "Call +1.000 chips | Raise ½ pot +1.000 chips (a difference under 0.001 chips)"),
])
def test_rounding_never_makes_unequal_values_look_tied(values, comparison) -> None:
    e = ui.default_explanation(report(FACING, C.CHECK_CALL, values))
    assert e.comparison == "Estimated value: " + comparison
    assert "tied" not in e.reason and "only slightly ahead" in e.reason


def test_a_small_negative_value_is_not_shown_as_zero() -> None:
    e = ui.default_explanation(report(FACING, C.FOLD, evs(fold=0.0, call=-0.03, half=-1.0, pot=-2.0, allin=-4.0)))
    assert e.comparison == "Estimated value: Fold 0 chips | Call −0.03 chips"


# --- missing or inconsistent diagnostics ----------------------------------------------------------------------


def test_single_legal_action() -> None:
    e = ui.default_explanation(report(ONLY, C.CHECK_CALL, evs(call=0.7)))
    assert (e.headline, e.reason, e.comparison) == (
        "AI checks.", "This was the AI's only legal option.", "Estimated value: Check +0.7 chips")


def test_missing_diagnostics_and_values_give_a_notice_and_no_comparison() -> None:
    e = ui.default_explanation(report(FACING, C.CHECK_CALL, {}, diagnostics=False))
    assert (e.headline, e.status, e.details) == ("AI calls 4 chips.", "no_diagnostics", False)
    assert e.reason == "No recorded analysis is available for this decision."
    assert e.comparison is e.response is e.warning is None

    for values in ({}, evs(call=1.3), evs(fold=0.0, call=1.3, half=-0.4)):
        e = ui.default_explanation(report(FACING, C.CHECK_CALL, values))
        assert (e.status, e.details, e.comparison) == ("missing_values", False, None)
        assert e.reason.startswith("The AI's estimated values were not fully recorded")


def test_missing_prediction_for_the_chosen_bet_is_left_out() -> None:
    e = ui.default_explanation(report(UNOPENED, C.RAISE_HALF_POT, evs(call=0.2, half=1.6, pot=1.1, allin=-2.0),
                                      probs=False))
    assert e.response is None and e.warning is None and e.comparison is not None
    assert e.details is True


def test_incomplete_prediction_records_hide_the_detailed_analysis() -> None:
    e = ui.default_explanation(report(UNOPENED, C.CHECK_CALL, evs(call=0.8, half=0.1, pot=-0.2, allin=-5.0),
                                      drop_called=True))
    assert e.status == "ok" and e.details is False


@pytest.mark.parametrize("make", [
    lambda: report(FACING, C.CHECK_CALL, evs(fold=0.0, call=1.3, half=-0.4, pot=-1.1, allin=-3.0), chosen="fold"),
    lambda: report(SHOVED, C.CHECK_CALL, {**evs(fold=0.0, call=1.0), "raise_pot": 0.5}),
    lambda: report(FACING, C.CHECK_CALL, evs(fold=0.5, call=0.1, half=-0.4, pot=-1.1, allin=-3.0)),
], ids=["chosen-differs", "illegal-action-in-values", "chosen-not-best"])
def test_mismatched_diagnostics_are_not_explained(make) -> None:
    r = make()
    e = ui.default_explanation(r)
    assert e.headline == ui._headline(r)  # the action actually played
    assert (e.status, e.reason, e.details) == ("mismatch", ui.MISMATCH_REASON, False)
    assert e.comparison is e.response is e.warning is None


def test_default_text_avoids_jargon_and_card_reading_claims() -> None:
    cases = [
        report(FACING, C.CHECK_CALL, evs(fold=0.0, call=1.3, half=-0.4, pot=-1.1, allin=-3.0)),
        report(FACING, C.FOLD, evs(fold=0.0, call=-2.5, half=-3.0, pot=-4.0, allin=-6.0)),
        report(UNOPENED, C.RAISE_HALF_POT, evs(call=0.2, half=1.6, pot=1.1, allin=-2.0), low=ALL_RAISES_LOW),
        report(PREFLOP, C.RAISE_POT, evs(fold=0.0, call=0.4, half=0.9, pot=1.5, allin=-1.0)),
        report(FACING, C.ALL_IN, evs(fold=0.0, call=2.0, half=1.0, pot=1.0, allin=7.25)),
    ]
    for r in cases:
        e = ui.default_explanation(r)
        text = " ".join(t for t in (e.headline, e.reason, e.comparison, e.response, e.warning) if t)
        assert not re.search(r"\bEV\b|equity|bluff|trap|read|your cards|guarantee", text, re.IGNORECASE), text


# --- real played decisions ----------------------------------------------------------------------------------


SEEDS = (5, 77)
NUMBER = re.compile(r"([+−]?\d+(?:\.(\d+))?) chips")


@pytest.mark.parametrize("seed", SEEDS)
def test_real_decisions_are_explained_only_from_their_recorded_values(tiny_bundle, seed) -> None:
    controller = HeadsUpController(ml_agent(tiny_bundle), match_seed=seed)
    for h, ref in enumerate(reference_match(ml_agent(tiny_bundle), "maniac", seed, 6)):
        if h:
            controller.next_hand()
        play_controller_hand(controller, human_categories(ref["result"], ref["setup"]))
    checked = 0
    for hand in controller.completed_hands:
        ai_moves = [(a, dec) for a, dec in zip(hand.history, decisions_from_history(hand.history))
                    if a.position != hand.human_position]
        assert len(ai_moves) == len(hand.ai_decisions)
        for report_, (action, decision) in zip(hand.ai_decisions, ai_moves):
            d = report_.diagnostics
            e = ui.default_explanation(report_)
            checked += 1
            # The headline is the table announcement of the same public action.
            assert e.headline == ui.ai_action_announcement(action, decision.to_call) + "."
            assert (e.status, e.details) == ("ok", True)
            # Every number shown is a recorded value, rounded to the decimals shown.
            others = {n: v for n, v in d.action_evs.items() if n != d.chosen}
            expected = [d.action_evs[d.chosen]] + ([max(others.values())] if others else [])
            shown = NUMBER.findall(e.comparison)
            assert len(shown) == len(expected)
            for (text, decimals), value in zip(shown, expected):
                assert abs(float(text.replace("−", "-")) - value) <= 0.5 * 10 ** -len(decimals) + 1e-12
            # Predictions and warnings only for the chosen bet, and only as recorded.
            is_bet = report_.category in RAISE_CATEGORIES
            assert (e.response is not None) == (is_bet and d.chosen in d.opponent_probabilities)
            assert (e.warning is not None) == (is_bet and bool(d.coverage.get(d.chosen, {}).get("low")))
    assert checked > 0
