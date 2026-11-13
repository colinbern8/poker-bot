"""Tests for hybridpoker.opponent_tracker: live Feature Set B equals the training features."""

import numpy as np
import pandas as pd
import pytest

from hybridpoker.actions import ActionCategory as C
from hybridpoker.agents import make_agent
from hybridpoker.agents.base import Agent
from hybridpoker.hand_log import decisions_frame
from hybridpoker.match import run_match
from hybridpoker.observation import PublicAction
from hybridpoker.opponent_history import B_FEATURES, history_features
from hybridpoker.opponent_tracker import OpponentTracker, decisions_from_history
from hybridpoker.simulator import public_hand_summary, public_history


class TrackingAgent(Agent):
    """Plays like ``inner``; before each own decision, snapshots its tracker's view of the opponent."""

    def __init__(self, inner: Agent, priors) -> None:
        super().__init__(inner.name)
        self.inner = inner
        self.tracker = OpponentTracker(priors)
        self.snapshots: list[tuple[int, pd.Series]] = []

    def act(self, observation, rng):
        self.snapshots.append((self.tracker.current_deal_index(), self.tracker.history_features()))
        return self.inner.act(observation, rng)

    def observe_hand_end(self, summary) -> None:
        self.tracker.observe_hand_end(summary)

    def config(self):
        return self.inner.config()


def test_to_call_reconstruction_hand_example() -> None:
    history = (
        PublicAction("preflop", "button", C.RAISE_POT, 6, 5, False),        # faced 1 (SB 1 vs BB 2)
        PublicAction("preflop", "big_blind", C.RAISE_POT, 18, 16, False),   # faced 4
        PublicAction("preflop", "button", C.CHECK_CALL, 18, 12, False),     # faced 12
        PublicAction("flop", "big_blind", C.CHECK_CALL, 0, 0, False),       # faced 0
        PublicAction("flop", "button", C.RAISE_HALF_POT, 18, 18, False),    # faced 0
        PublicAction("flop", "big_blind", C.FOLD, 0, 0, False),             # faced 18
    )
    assert [d.to_call for d in decisions_from_history(history)] == [1, 4, 12, 0, 0, 18]
    assert [d.position for d in decisions_from_history(history)][:2] == ["button", "big_blind"]


@pytest.mark.parametrize("hero, villain, seed", [("probability", "maniac", 3), ("nit", "solid", 8), ("random", "random", 1)])
def test_to_call_reconstruction_matches_logged_decisions(hero, villain, seed) -> None:
    match = run_match(make_agent(hero, 30), make_agent(villain, 30), deal_count=6, seed=seed)
    for hand in match.hands:
        rebuilt = decisions_from_history(public_history(hand.result))
        logged = hand.result.decisions
        assert len(rebuilt) == len(logged)
        for d, record in zip(rebuilt, logged):
            assert d.to_call == record.observation.to_call
            assert d.street == record.observation.street
            assert d.position == record.observation.position
            assert d.category == record.action.category.value


def test_current_deal_index_seat_swapped_and_not(tiny_dataset) -> None:
    for seat_swap in (True, False):
        hero = TrackingAgent(make_agent("solid", 30), tiny_dataset.priors)
        match = run_match(hero, make_agent("maniac", 30), deal_count=5, seed=4, seat_swap=seat_swap)
        hero_hands = [h for h in match.hands for r in h.result.decisions if r.seat == h.hero_seat]
        assert [deal for deal, _ in hero.snapshots] == [h.deal_index for h in hero_hands]


@pytest.mark.parametrize("hero_name, villain_name, seed", [("probability", "maniac", 11), ("solid", "nit", 12)])
def test_live_history_features_equal_offline_training_features(tiny_dataset, hero_name, villain_name, seed) -> None:
    """Each player's live view of its opponent must equal the training pipeline's features
    for that opponent in the same deal (computed offline from the full log)."""
    priors = tiny_dataset.priors
    hero = TrackingAgent(make_agent(hero_name, 30), priors)
    villain = TrackingAgent(make_agent(villain_name, 30), priors)
    match = run_match(hero, villain, deal_count=8, seed=seed)
    frame = decisions_frame(match)
    offline = history_features(frame, priors)
    role_of = {"hero": "villain", "villain": "hero"}  # whose features each tracker computes
    checked = 0
    for agent, role in ((hero, "hero"), (villain, "villain")):
        own_rows = frame[frame["label_actor_role"] == role]
        opponent_rows = frame[frame["label_actor_role"] == role_of[role]]
        assert len(agent.snapshots) == len(own_rows)
        for (deal, live), (_, row) in zip(agent.snapshots, own_rows.iterrows()):
            assert deal == row["id_deal_index"]
            same_deal = opponent_rows[opponent_rows["id_deal_index"] == deal]
            if same_deal.empty:
                continue  # opponent never acted in this deal; nothing offline to compare
            expected = offline.loc[same_deal.index[0], list(B_FEATURES)]
            np.testing.assert_allclose(live.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=0, atol=1e-12)
            checked += 1
    assert checked > 10


def test_mirror_hand_and_own_actions_are_not_counted(tiny_dataset) -> None:
    hero = TrackingAgent(make_agent("probability", 30), tiny_dataset.priors)
    match = run_match(hero, make_agent("maniac", 30), deal_count=4, seed=21)
    tracker = OpponentTracker(tiny_dataset.priors)
    summaries = []
    # Rebuild the hero's summaries and feed them one at a time.
    for hand in match.hands:
        summaries.append(public_hand_summary(hand.result, hand.hero_seat, hand.deal_index, hand.seat_swap, 2))
    before = tracker.history_features()
    tracker.observe_hand_end(summaries[0])  # deal 0, first seating
    assert tracker.current_deal_index() == 0
    pd.testing.assert_series_equal(tracker.history_features(), before)  # mirror hand of the current deal excluded
    tracker.observe_hand_end(summaries[1])  # deal 0, second seating
    assert tracker.current_deal_index() == 1
    after = tracker.history_features()
    assert after["b_log_prior_hands"] > 0  # deal 0 now counts
    villain_actions = sum(1 for s in summaries[:2] for d in decisions_from_history(s.history) if d.position != s.position)
    assert tracker.observed_rows() == villain_actions  # only the opponent's actions were recorded


def test_first_hand_features_equal_priors(tiny_dataset) -> None:
    features = OpponentTracker(tiny_dataset.priors).history_features()
    assert features["b_log_prior_hands"] == 0 and features["b_log_prior_decisions"] == 0
    assert features["b_vpip"] == pytest.approx(tiny_dataset.priors.vpip)
    assert features["b_face_fold"] == pytest.approx(tiny_dataset.priors.face_rates["fold"])
