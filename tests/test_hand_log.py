"""Tests for hybridpoker.hand_log (CSV logs and run metadata)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from pokerkit import Card

from hybridpoker.actions import ActionCategory
from hybridpoker.agents import make_agent
from hybridpoker.hand_log import COLUMN_GROUPS, decisions_frame, hand_rows, write_run
from hybridpoker.match import MatchResult, run_match

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
