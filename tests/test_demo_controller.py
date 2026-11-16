"""The heads-up controller: rules, equivalence with uninterrupted matches, opponent tracking.

The human is played by the actions a stand-in bot chose in an uninterrupted
reference match on the same seed, so every controller hand can be compared
with the reference exactly.
"""

from __future__ import annotations

import pytest

from demo_support import (
    assert_same_diagnostics,
    assert_same_hand,
    human_categories,
    ml_agent,
    play_controller_hand,
    probability_agent,
    reference_match,
    tracker_state,
)
from hybridpoker.agents.ev_agent import EVAgent
from hybridpoker.game import BIG_BLIND, BIG_BLIND_SEAT, BUTTON_SEAT, SMALL_BLIND, STARTING_STACK
from hybridpoker.opponent_tracker import decisions_from_history, other_position
from hybridpoker_demo.controller import CompletedHand, HeadsUpController, HumanTurn, Phase
from hybridpoker_demo.seeds import MAX_SEED, check_seed, hand_setup

SEEDS = (5, 77, 4242)
HANDS = 8


class CountingEV(EVAgent):
    """The ML EV agent, counting end-of-hand notifications."""

    hand_end_calls = 0

    def observe_hand_end(self, summary) -> None:
        self.hand_end_calls += 1
        super().observe_hand_end(summary)


@pytest.mark.parametrize("seed", SEEDS)
def test_controller_match_equals_uninterrupted_reference_match(tiny_bundle, seed) -> None:
    style = ("maniac", "solid", "calling_station")[seed % 3]
    reference_ai = ml_agent(tiny_bundle)
    expected = reference_match(reference_ai, style, seed, HANDS)
    ai = ml_agent(tiny_bundle, CountingEV)
    controller = HeadsUpController(ai, match_seed=seed)
    reports = []
    for h, ref in enumerate(expected):
        if h:
            controller.next_hand()
        setup = ref["setup"]
        play_controller_hand(controller, human_categories(ref["result"], setup))
        done = controller.view
        assert isinstance(done, CompletedHand) and done.hand_index == h
        assert_same_hand(controller._results[h], ref["result"], setup.ai_seat)
        assert [r.rng_state_after for r in controller._records] == ref["rng_states"]
        assert tracker_state(ai) == ref["tracker"]       # opponent profile identical after every hand
        assert ai.hand_end_calls == h + 1                 # exactly one notification per completed hand
        reports.extend(done.ai_decisions)
    assert_same_diagnostics([r.diagnostics for r in reports], reference_ai.diagnostics)
    assert_same_diagnostics(ai.diagnostics, reference_ai.diagnostics)
    total = sum(ref["result"].payoffs[ref["setup"].human_seat] for ref in expected)
    assert controller.scoreboard.hands_completed == HANDS
    assert controller.scoreboard.human_net_chips == total == -controller.scoreboard.ai_net_chips


@pytest.mark.parametrize("seed", SEEDS)
def test_tracker_rows_equal_human_actions_counted_independently(tiny_bundle, seed) -> None:
    """Verify (not assume) the link between ``observed_rows()`` and the human's actions.

    Expected rows are counted two independent ways from the public history:
    the decisions whose position is the human's, and the decisions the
    tracker's own replay of the history attributes to the AI's opponent.
    """
    ai = ml_agent(tiny_bundle)
    controller = HeadsUpController(ai, match_seed=seed)
    expected = reference_match(ml_agent(tiny_bundle), "maniac", seed, HANDS)
    rows_before = 0
    for h, ref in enumerate(expected):
        if h:
            controller.next_hand()
        setup = ref["setup"]
        play_controller_hand(controller, human_categories(ref["result"], setup))
        done = controller.view
        by_position = sum(a.position == done.human_position for a in done.history)
        ai_position = "big_blind" if done.human_position == "button" else "button"
        by_tracker_rule = sum(d.position == other_position(ai_position) for d in decisions_from_history(done.history))
        assert ai.tracker.observed_rows() - rows_before == by_position == by_tracker_rule
        rows_before = ai.tracker.observed_rows()


def test_rules_fresh_200_chip_stacks_blinds_and_alternating_button(tiny_bundle) -> None:
    controller = HeadsUpController(ml_agent(tiny_bundle), match_seed=99)
    positions = []
    for h in range(HANDS):
        if h:
            controller.next_hand()
        setup = hand_setup(99, h)
        while isinstance(controller.view, HumanTurn):
            legal = controller.view.legal_categories
            controller.submit(controller.view.ticket, legal[-1 if h % 3 else 1])
        result = controller._results[h]
        first = result.decisions[0].observation
        assert first.own_stack + first.own_bet == STARTING_STACK == first.opponent_stack + first.opponent_bet
        assert first.pot_size == SMALL_BLIND + BIG_BLIND and first.street == "preflop"
        assert sum(result.payoffs) == 0
        assert controller.view.human_net == result.payoffs[setup.human_seat]
        positions.append(controller.view.human_position)
        assert setup.human_seat == (BUTTON_SEAT if h % 2 == 0 else BIG_BLIND_SEAT)
    assert positions == ["button", "big_blind"] * (HANDS // 2)
    board = controller.scoreboard
    assert board.human_net_chips == sum(h.human_net for h in controller.completed_hands)
    assert board.human_net_bb == board.human_net_chips / BIG_BLIND
    assert board.human_bb_per_100 == pytest.approx(100 * board.human_net_bb / HANDS)


def test_same_seed_and_actions_reproduce_the_match_and_seeds_are_checked(tiny_bundle) -> None:
    def play(seed):
        controller = HeadsUpController(ml_agent(tiny_bundle), match_seed=seed)
        for h in range(4):
            if h:
                controller.next_hand()
            while isinstance(controller.view, HumanTurn):
                controller.submit(controller.view.ticket, controller.view.legal_categories[0 if h % 2 else -2])
        return controller

    a, b = play(123), play(123)
    assert a._results == b._results
    assert [h.human_net for h in a.completed_hands] == [h.human_net for h in b.completed_hands]
    assert a.match_id != b.match_id  # tickets of one match never fit another
    c = play(124)
    assert [r.hole_cards for r in c._results] != [r.hole_cards for r in a._results]

    random_seeds = {HeadsUpController(probability_agent()).match_seed for _ in range(3)}
    assert len(random_seeds) == 3 and all(0 <= s <= MAX_SEED for s in random_seeds)
    for bad in (-1, MAX_SEED + 1, 1.5, True, "7"):
        with pytest.raises(ValueError):
            check_seed(bad)


def test_controller_refuses_an_agent_that_has_already_played(tiny_bundle) -> None:
    ai = ml_agent(tiny_bundle)
    reference_match(ai, "solid", 3, 1)
    with pytest.raises(ValueError):
        HeadsUpController(ai, match_seed=1)


def test_phase_and_views(tiny_bundle) -> None:
    controller = HeadsUpController(ml_agent(tiny_bundle), match_seed=8)
    while controller.phase is Phase.HUMAN_TO_ACT:
        view = controller.view
        assert view.ticket.decision_index == len(view.observation.history)
        assert view.ticket.hand_index == view.hand_index == controller.hand_index == 0
        controller.submit(view.ticket, view.legal_categories[-1])
    assert controller.phase is Phase.HAND_COMPLETE
    assert controller.completed_hands == (controller.view,)
