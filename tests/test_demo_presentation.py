"""Display helpers and the opponent profile: labels, results, and explanations that match the diagnostics."""

from __future__ import annotations

import math

import pytest
from pokerkit import Card

from demo_support import human_categories, ml_agent, play_controller_hand, reference_match, tracker_state
from hybridpoker.action_probabilities import CLASS_ORDER
from hybridpoker.actions import ActionCategory as C
from hybridpoker.observation import PublicAction
from hybridpoker_demo import presentation as ui
from hybridpoker_demo.controller import CompletedHand, HeadsUpController, HumanTurn

SEEDS = (5, 77, 4242)
HANDS = 8


def cards(text: str) -> list[Card]:
    return list(Card.parse(text))


def test_cards_use_suit_symbols_red_for_hearts_and_diamonds() -> None:
    assert [ui.card_text(c) for c in cards("Th2sAdKc")] == ["10♥", "2♠", "A♦", "K♣"]
    assert ui.card_markdown(cards("Qh")[0]) == ":red[**Q♥**]"
    assert ui.card_markdown(cards("Qs")[0]) == "**Q♠**"
    assert ui.cards_markdown([]) == "—"
    assert ui.hand_name(cards("AsAd"), cards("Ah7c2d9sKh")) == "Three of a kind"


@pytest.mark.parametrize("args, label", [
    ((C.FOLD, 2, 0, 1, "preflop"), "Fold"),
    ((C.CHECK_CALL, 2, 0, 0, "flop"), "Check"),
    ((C.CHECK_CALL, 6, 4, 4, "turn"), "Call 4"),
    ((C.CHECK_CALL, 200, 150, 150, "river", True), "Call 150 (all-in)"),
    ((C.RAISE_HALF_POT, 6, 6, 0, "flop"), "Bet 6 (½ pot)"),
    ((C.RAISE_POT, 6, 5, 1, "preflop"), "Raise to 6 (pot)"),
    ((C.RAISE_POT, 6, 4, 0, "preflop"), "Raise to 6 (pot)"),  # big blind raising after a limp
    ((C.RAISE_POT, 200, 190, 10, "turn", True), "Raise to 200 (pot) — all-in"),
    ((C.ALL_IN, 200, 198, 1, "preflop"), "All-in (200)"),
])
def test_action_labels_carry_the_engine_amounts(args, label) -> None:
    assert ui.action_text(*args) == label


def test_history_lines_name_players_streets_and_amounts() -> None:
    history = (PublicAction("preflop", "button", C.RAISE_POT, 6, 5, False),
               PublicAction("preflop", "big_blind", C.CHECK_CALL, 6, 4, False),
               PublicAction("flop", "big_blind", C.CHECK_CALL, 0, 0, False),
               PublicAction("flop", "button", C.RAISE_HALF_POT, 6, 6, False),
               PublicAction("flop", "big_blind", C.FOLD, 0, 0, False))
    assert ui.history_lines(history, "button") == [
        "Preflop · You (Button): Raise to 6 (pot)",
        "Preflop · AI (Big blind): Call 4",
        "Flop · AI (Big blind): Check",
        "Flop · You (Button): Bet 6 (½ pot)",
        "Flop · AI (Big blind): Fold",
    ]


def play_match(bundle, seed: int, hands: int = HANDS):
    ai = ml_agent(bundle)
    controller = HeadsUpController(ai, match_seed=seed)
    for h, ref in enumerate(reference_match(ml_agent(bundle), "maniac", seed, hands)):
        if h:
            controller.next_hand()
        play_controller_hand(controller, human_categories(ref["result"], ref["setup"]))
    return controller, ai


@pytest.mark.parametrize("seed", SEEDS)
def test_explanations_report_exactly_the_recorded_diagnostics(tiny_bundle, seed) -> None:
    controller, _ = play_match(tiny_bundle, seed)
    explained = 0
    for hand in controller.completed_hands:
        result = controller._results[hand.hand_index]
        ai_moves = [d for d in result.decisions if d.observation.position != hand.human_position]
        for report, move in zip(hand.ai_decisions, ai_moves):
            d = report.diagnostics
            info = ui.explain_decision(report)
            explained += 1
            # The chosen action and its label are the AI's real legal action.
            assert d.chosen == move.action.category.value
            assert info.action == ui.legal_action_label(move.action, move.observation.to_call, move.observation.street)
            assert info.equity == d.equity
            # Every legal action, in order, with its recorded EV; exactly one is marked chosen.
            assert [r.ev for r in info.ev_rows] == list(d.action_evs.values())
            assert [r.chosen for r in info.ev_rows] == [name == d.chosen for name in d.action_evs]
            labels = {a.category.value: ui.legal_action_label(a, move.observation.to_call, move.observation.street)
                      for a in move.observation.legal_actions}
            assert [r.action for r in info.ev_rows] == [labels[name] for name in d.action_evs]
            # Model predictions: masked probabilities collapsed to fold / call / raise.
            for row, (name, probs) in zip(info.responses, d.opponent_probabilities.items()):
                p = dict(zip(CLASS_ORDER, probs))
                assert row.action == labels[name]
                assert (row.fold, row.call) == (p["fold"], p["check_call"])
                assert math.isclose(row.raise_, p["raise_half_pot"] + p["raise_pot"] + p["all_in"])
                assert row.continue_fraction == d.continue_fraction[name]
                assert row.equity_if_called == d.called_equity[name]
                assert (row.training_rows, row.low_coverage) == (d.coverage[name]["training_rows"],
                                                                 d.coverage[name]["low"])
            assert bool(info.warnings and "this bet" in info.warnings[0]) == \
                bool(d.chosen in d.coverage and d.coverage[d.chosen]["low"])
            assert f"{d.equity:.0%}" in info.summary and f"{d.action_evs[d.chosen]:+.1f}" in info.summary
            assert info.action in info.summary
            assert info.history_hands == d.opponent_history_hands
    assert explained > 0


def test_explanation_requires_recorded_diagnostics(tiny_bundle) -> None:
    controller, _ = play_match(tiny_bundle, 5, 2)
    report = next(r for h in controller.completed_hands for r in h.ai_decisions)
    with pytest.raises(ValueError):
        ui.explain_decision(type(report)(**{**report.__dict__, "diagnostics": None}))


@pytest.mark.parametrize("seed", SEEDS)
def test_hand_results_and_folds(tiny_bundle, seed) -> None:
    controller, _ = play_match(tiny_bundle, seed)
    for hand in controller.completed_hands:
        text = ui.hand_result(hand)
        assert text.outcome == ("win" if hand.human_net > 0 else "loss" if hand.human_net < 0 else "split")
        if hand.went_to_showdown:
            assert ui.hand_name(hand.ai_hole_cards, hand.board_cards) in text.detail
        else:
            assert "folded" in text.detail and "hidden" in text.detail


@pytest.mark.parametrize("seed", SEEDS)
def test_opponent_profile_is_read_only_and_consistent_with_observed_actions(tiny_bundle, seed) -> None:
    controller, ai = play_match(tiny_bundle, seed)
    before = tracker_state(ai)
    profile = controller.opponent_profile()
    assert tracker_state(ai) == before  # reading changes nothing
    profile.rates["b_vpip"] = -1.0     # a copy: changing it does not reach the AI
    assert controller.opponent_profile().rates["b_vpip"] != -1.0
    profile = controller.opponent_profile()

    hands = controller.completed_hands
    human_actions = sum(a.position == h.human_position for h in hands for a in h.history)
    assert profile.hands_observed == len([h for h in hands if any(a.position == h.human_position for a in h.history)])
    assert profile.decisions_observed == human_actions == ai.tracker.observed_rows()

    # Observed counts and the model inputs agree: model = (count + alpha * population) / (n + alpha).
    observed = ui._observed_counts(hands)
    for key, feature in (("vpip", "b_vpip"), ("pfr", "b_pfr"), ("face_fold", "b_face_fold"),
                         ("face_call", "b_face_call"), ("open_check", "b_open_check")):
        count, n = observed[key]
        expected = (count + profile.alpha * profile.population[feature]) / (n + profile.alpha)
        assert math.isclose(profile.rates[feature], expected), key
    rows = ui.profile_rows(profile, hands)
    assert len(rows) == 7 and all(0.0 <= r.model_input <= 1.0 and 0.0 <= r.population <= 1.0 for r in rows)


def test_profile_is_none_for_an_ai_without_a_tracker() -> None:
    from demo_support import probability_agent

    assert HeadsUpController(probability_agent(), match_seed=1).opponent_profile() is None
