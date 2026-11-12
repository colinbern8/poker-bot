"""Tests for hybridpoker.hand_log (CSV logs and run metadata)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from pokerkit import Card

from pokerkit import Deck

from hybridpoker.actions import ActionCategory
from hybridpoker.agents import make_agent
from hybridpoker.game import create_heads_up_state
from hybridpoker.hand_log import (
    COLUMN_GROUPS,
    decisions_frame,
    feature_columns,
    hand_rows,
    write_run,
)
from hybridpoker.match import MatchResult, run_match
from hybridpoker.observation import observe

FAST = 30


@pytest.fixture(scope="module")
def match_and_agents():
    hero = make_agent("probability", FAST)
    villain = make_agent("maniac", FAST)
    return run_match(hero, villain, deal_count=15, seed=21), hero, villain


@pytest.fixture(scope="module")
def frame(match_and_agents) -> pd.DataFrame:
    return decisions_frame(match_and_agents[0])


def parse(text: str) -> set:
    return set(Card.parse(text)) if isinstance(text, str) and text else set()


# --- Column separation ---------------------------------------------------------------


def test_every_column_belongs_to_a_documented_group(frame: pd.DataFrame) -> None:
    for column in frame.columns:
        assert any(column.startswith(prefix) for prefix in COLUMN_GROUPS), column


def test_observable_columns_never_contain_hole_cards(frame: pd.DataFrame) -> None:
    observable = [c for c in frame.columns if c.startswith("obs_")]
    assert not any("hole" in c or "equity" in c or "net" in c for c in observable)
    for _, row in frame.iterrows():
        hidden = parse(row["hidden_actor_hole_cards"]) | parse(row["hidden_opponent_hole_cards"])
        assert len(hidden) == 4
        assert not parse(row["obs_board"]) & hidden


def test_bot_parameters_are_not_in_decision_rows(frame: pd.DataFrame) -> None:
    for name in ("looseness", "aggression", "bluff_rate", "equity_samples", "raise_equity"):
        assert not any(name in column for column in frame.columns)


def test_labels_and_outcomes_are_kept_apart(frame: pd.DataFrame) -> None:
    assert set(frame["label_actor_agent"]) == {"probability", "maniac"}
    assert set(frame["label_actor_role"]) == {"hero", "villain"}
    assert {c for c in frame.columns if c.startswith("out_")} == {
        "out_actor_net_chips",
        "out_went_to_showdown",
        "out_final_board",
    }


# --- Game state before each decision -----------------------------------------------------


def test_first_decision_of_each_hand_is_button_facing_the_blind(frame: pd.DataFrame) -> None:
    first = frame[frame["id_decision_index"] == 0]
    assert (first["obs_position"] == "button").all()
    assert (first["obs_pot_size"] == 3).all()
    assert (first["obs_to_call"] == 1).all()
    assert (first["obs_history"].fillna("") == "").all()


def test_action_amounts_match_the_legal_menu(frame: pd.DataFrame) -> None:
    for _, row in frame.iterrows():
        category = row["act_category"]
        assert row[f"obs_legal_{category}"]
        assert row["act_bet_to"] == row[f"obs_bet_to_{category}"]
        assert row["act_chips_added"] <= row["obs_own_stack"]
        assert row["act_is_all_in"] == (row["act_chips_added"] == row["obs_own_stack"] > 0)


def test_legal_mask_matches_bet_amounts(frame: pd.DataFrame) -> None:
    for category in ActionCategory:
        legal = frame[f"obs_legal_{category.value}"]
        amounts = frame[f"obs_bet_to_{category.value}"]
        assert (amounts.notna() == legal).all()
    assert (frame.loc[frame["obs_to_call"] == 0, "obs_legal_fold"] == False).all()  # noqa: E712


def test_history_and_counts_are_consistent(frame: pd.DataFrame) -> None:
    for _, row in frame.iterrows():
        history = row["obs_history"] if isinstance(row["obs_history"], str) else ""
        entries = history.split(";") if history else []
        assert len(entries) == row["id_decision_index"]
        raises = [e for e in entries if e.split(":")[2] in ("raise_half_pot", "raise_pot", "all_in")]
        assert len(raises) == row["obs_raises_this_hand"]


def test_outcomes_match_hand_results(match_and_agents, frame: pd.DataFrame) -> None:
    match: MatchResult = match_and_agents[0]
    hands = pd.DataFrame(hand_rows(match))
    assert len(hands) == len(match.hands) == 30
    assert len(frame) == sum(len(h.result.decisions) for h in match.hands)
    assert (hands["net_seat0"] + hands["net_seat1"] == 0).all()
    merged = frame.merge(hands, left_on="id_hand_index", right_on="hand_index")
    hero_rows = merged[merged["label_actor_role"] == "hero"]
    assert (hero_rows["out_actor_net_chips"] == hero_rows["hero_net"]).all()


# --- Files on disk ------------------------------------------------------------------------


def test_write_run_creates_files_and_metadata(match_and_agents, tmp_path: Path) -> None:
    match, hero, villain = match_and_agents
    paths = write_run(match, hero, villain, tmp_path / "run")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["seed"] == 21 and metadata["deal_count"] == 15
    assert metadata["game"]["big_blind"] == 2 and metadata["game"]["starting_stack"] == 200
    assert metadata["agents"]["villain"]["aggression"] == 0.85  # private params live here
    assert metadata["results"]["ci_unit"] == "deal pair"
    assert metadata["column_groups"] == COLUMN_GROUPS

    decisions = pd.read_csv(paths["decisions"])
    assert len(decisions) == len(decisions_frame(match))
    text = paths["decisions"].read_text(encoding="utf-8")
    assert ".0," not in text.split("\n")[1]  # bet amounts stay whole numbers


def test_logs_are_byte_identical_for_the_same_seed(tmp_path: Path) -> None:
    contents = []
    for attempt in ("a", "b"):
        hero, villain = make_agent("probability", FAST), make_agent("solid", FAST)
        match = run_match(hero, villain, deal_count=6, seed=99)
        paths = write_run(match, hero, villain, tmp_path / attempt)
        contents.append((paths["decisions"].read_bytes(), paths["hands"].read_bytes()))
    assert contents[0] == contents[1]


# --- Leakage audit: observable features come from public information only ------------


def history_entries(row) -> list[list[str]]:
    history = row["obs_history"] if isinstance(row["obs_history"], str) else ""
    return [entry.split(":") for entry in history.split(";")] if history else []


def replay_from_public_information(row):
    """Rebuild the observation using only public data and FAKE hole cards.

    The logged board and betting history are replayed in a fresh PokerKit
    game whose hole cards are arbitrary. If the result matches every obs_
    value, those values cannot depend on any private information.
    """
    board = list(Card.parse(row["obs_board"])) if isinstance(row["obs_board"], str) else []
    fake_cards = iter([card for card in Deck.STANDARD if card not in set(board)])
    board_cards = iter(board)
    state = create_heads_up_state()

    def deal_pending_cards() -> None:
        while True:
            if state.can_deal_hole():
                state.deal_hole(next(fake_cards))
            elif state.can_burn_card():
                state.burn_card(next(fake_cards))
            elif state.can_deal_board():
                state.deal_board(next(board_cards))
            else:
                return

    for _street, _position, category, bet_to in history_entries(row):
        deal_pending_cards()
        if category == "fold":
            state.fold()
        elif category == "check_call":
            state.check_or_call()
        else:
            state.complete_bet_or_raise_to(int(bet_to))
    deal_pending_cards()
    return observe(state, history=())


def test_observable_features_are_reproducible_from_public_information(frame: pd.DataFrame) -> None:
    for _, row in frame.iterrows():
        public = replay_from_public_information(row)
        assert row["obs_street"] == public.street
        assert row["obs_position"] == public.position
        assert row["obs_pot_size"] == public.pot_size
        assert row["obs_to_call"] == public.to_call
        assert row["obs_own_stack"] == public.own_stack
        assert row["obs_opponent_stack"] == public.opponent_stack
        assert row["obs_effective_stack"] == public.effective_stack
        assert row["obs_own_bet"] == public.own_bet
        assert row["obs_opponent_bet"] == public.opponent_bet
        for category in ActionCategory:
            legal = public.legal(category)
            assert row[f"obs_legal_{category.value}"] == (legal is not None)
            expected = legal.bet_to if legal else None
            logged = row[f"obs_bet_to_{category.value}"]
            assert (pd.isna(logged) and expected is None) or logged == expected


def test_rows_describe_the_state_before_the_action(frame: pd.DataFrame) -> None:
    for _, hand in frame.groupby(["id_match", "id_hand_index"]):
        rows = hand.sort_values("id_decision_index").to_dict("records")
        for before, after in zip(rows, rows[1:]):
            # The next decision sees exactly this action added to the pot and history.
            assert after["obs_pot_size"] == before["obs_pot_size"] + before["act_chips_added"]
            last = history_entries(after)[-1]
            assert last[2] == before["act_category"]
            assert int(last[3]) == before["act_bet_to"]


def test_feature_columns_select_only_observable_features(frame: pd.DataFrame) -> None:
    features = feature_columns(frame)
    assert features == [c for c in frame.columns if c.startswith("obs_")]
    excluded = set(frame.columns) - set(features)

    # Every identifier, target, hidden, label and outcome column is excluded.
    for prefix in ("id_", "act_", "hidden_", "label_", "out_"):
        group = {c for c in frame.columns if c.startswith(prefix)}
        assert group and group <= excluded, prefix
    assert {
        "id_match", "id_deal_seed",  # identifiers (the deal seed determines the cards)
        "hidden_actor_hole_cards", "hidden_opponent_hole_cards",  # hidden cards
        "hidden_agent_equity", "hidden_agent_reason",  # agent internals
        "label_actor_agent", "label_actor_role",  # true style labels
        "out_actor_net_chips", "out_went_to_showdown", "out_final_board",  # future outcomes
    } <= excluded

    # No feature name refers to private or future information.
    forbidden = ("hole", "hidden", "label", "agent", "role", "net", "showdown",
                 "final", "seed", "match", "equity", "reason", "loose", "aggress", "bluff")
    for column in features:
        assert not any(word in column for word in forbidden), column

    # Every feature value can be computed at decision time by either player
    # (proven by test_observable_features_are_reproducible_from_public_information).


# --- Identifiers ---------------------------------------------------------------------------


def test_every_row_identifies_its_match_hand_and_decision(match_and_agents, frame: pd.DataFrame) -> None:
    match = match_and_agents[0]
    assert frame["id_match"].notna().all()
    assert set(frame["id_match"]) == {match.match_id}
    assert not frame.duplicated(["id_match", "id_hand_index", "id_decision_index"]).any()
    assert {row["match_id"] for row in hand_rows(match)} == {match.match_id}


def test_match_id_depends_on_configuration_and_seed() -> None:
    def match_id(villain_name: str, seed: int, samples: int = FAST) -> str:
        return run_match(make_agent("random"), make_agent(villain_name, samples), 2, seed).match_id

    base = match_id("nit", 1)
    assert base == match_id("nit", 1)  # deterministic
    assert base != match_id("nit", 2)  # different seed
    assert base != match_id("maniac", 1)  # different opponent
    assert base != match_id("nit", 1, samples=FAST + 1)  # different bot configuration
    assert base.startswith("m") and "nit" not in base
