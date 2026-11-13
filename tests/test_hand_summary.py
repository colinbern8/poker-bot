"""Tests for the Phase 4 public end-of-hand notification (match/simulator integration)."""

import dataclasses

import pytest

from hybridpoker.agents import make_agent
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.hand_log import observation_columns
from hybridpoker.match import run_match
from hybridpoker.observation import PublicAction, PublicHandSummary
from hybridpoker.simulator import public_history


class Recorder(Agent):
    """Plays like a wrapped agent and records every end-of-hand summary."""

    def __init__(self, inner: Agent) -> None:
        super().__init__(inner.name)
        self.inner = inner
        self.summaries: list[PublicHandSummary] = []

    def act(self, observation, rng) -> Decision:
        return self.inner.act(observation, rng)

    def observe_hand_end(self, summary: PublicHandSummary) -> None:
        self.summaries.append(summary)

    def config(self):
        return self.inner.config()


@pytest.fixture(scope="module")
def recorded():
    hero = Recorder(make_agent("probability", 30))
    villain = Recorder(make_agent("maniac", 30))
    match = run_match(hero, villain, deal_count=6, seed=13)
    return match, hero, villain


def test_every_agent_gets_one_summary_per_hand_in_order(recorded) -> None:
    match, hero, villain = recorded
    assert len(hero.summaries) == len(villain.summaries) == len(match.hands) == 12
    for hand, h, v in zip(match.hands, hero.summaries, villain.summaries):
        assert (h.deal_index, h.seat_swap, h.hands_per_deal) == (hand.deal_index, hand.seat_swap, 2)
        assert (v.deal_index, v.seat_swap) == (hand.deal_index, hand.seat_swap)
        hero_position = "button" if hand.hero_seat == 1 else "big_blind"
        assert h.position == hero_position and v.position != hero_position


def test_summary_history_is_the_complete_public_history(recorded) -> None:
    match, hero, _ = recorded
    for hand, summary in zip(match.hands, hero.summaries):
        result = hand.result
        last = result.decisions[-1]
        # The last decision's observation saw everything except the final action.
        assert summary.history[:-1] == last.observation.history
        final = summary.history[-1]
        assert (final.category, final.bet_to) == (last.action.category, last.action.bet_to)
        assert summary.history == public_history(result)
        assert summary.board_cards == result.board_cards


def test_summary_contains_only_public_fields() -> None:
    fields = {f.name for f in dataclasses.fields(PublicHandSummary)}
    assert fields == {"position", "history", "board_cards", "deal_index", "seat_swap", "hands_per_deal"}
    action_fields = {f.name for f in dataclasses.fields(PublicAction)}
    assert action_fields == {"street", "position", "category", "bet_to", "chips_added", "is_all_in"}


def test_summary_never_reveals_hole_cards(recorded) -> None:
    match, hero, villain = recorded
    for hand, h, v in zip(match.hands, hero.summaries, villain.summaries):
        hidden = set(hand.result.hole_cards[0]) | set(hand.result.hole_cards[1])
        for summary in (h, v):
            assert not hidden & set(summary.board_cards)
            assert all(isinstance(a, PublicAction) for a in summary.history)


def test_default_hook_does_nothing() -> None:
    agent = make_agent("solid", 30)
    assert agent.observe_hand_end(None) is None  # type: ignore[arg-type]


def test_recording_does_not_change_play(recorded) -> None:
    match, _, _ = recorded
    plain = run_match(make_agent("probability", 30), make_agent("maniac", 30), deal_count=6, seed=13)
    assert [h.result.payoffs for h in plain.hands] == [h.result.payoffs for h in match.hands]
    assert [h.result.decisions[i].action for h in plain.hands for i in range(len(h.result.decisions))] == \
        [h.result.decisions[i].action for h in match.hands for i in range(len(h.result.decisions))]


def test_observation_columns_never_read_hole_cards(recorded) -> None:
    match, _, _ = recorded
    record = match.hands[0].result.decisions[0]
    columns = observation_columns(record.observation)
    swapped = dataclasses.replace(record.observation, hole_cards=match.hands[0].result.hole_cards[1 - record.seat])
    assert observation_columns(swapped) == columns
    assert all(key.startswith("obs_") for key in columns)
