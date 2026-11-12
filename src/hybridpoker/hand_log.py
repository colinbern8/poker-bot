"""Write match results to CSV files for analysis and the Phase 3 dataset.

A run directory contains:

* ``decisions.csv``: one row per decision, describing the game state just
  **before** the decision and the action chosen.
* ``hands.csv``: one row per hand (cards, payouts, showdown).
* ``metadata.json``: seeds, game settings, agent configurations (including
  private bot parameters), software versions, and summary results.

Decision columns are grouped by prefix so later phases can tell at a
glance what a model may use:

``id_``      Row identifiers. Not features.
``obs_``     Observable decision features: public information available
             to *both* players at decision time. Safe model inputs.
``act_``     The action taken: abstract category and actual chip amounts.
             The prediction target for opponent modeling.
``hidden_``  Private information: hole cards and the agent's internal
             equity estimate and reasoning. Never a model input.
``label_``   True identity of the acting agent. For evaluation only.
``out_``     Results known only after the hand ends (future information).
             Never a model input.

Bot parameters (looseness, aggression, ...) are written only to
``metadata.json``, never to the per-decision rows.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from pokerkit import Card

from hybridpoker import __version__
from hybridpoker.actions import RAISE_CATEGORIES, ActionCategory
from hybridpoker.agents.base import Agent
from hybridpoker.game import BIG_BLIND, SMALL_BLIND, STARTING_STACK
from hybridpoker.match import MatchResult
from hybridpoker.observation import Observation
from hybridpoker.pot_odds import pot_odds

SCHEMA_VERSION = 2
FEATURE_PREFIX = "obs_"

COLUMN_GROUPS = {
    "id_": "Row identifiers; not features.",
    "obs_": "Observable decision features: public information at decision time.",
    "act_": "Action taken: abstract category and actual chip amounts (target).",
    "hidden_": "Private hole cards and agent internals; never model inputs.",
    "label_": "True identity of the acting agent; evaluation only.",
    "out_": "Hand results known only afterwards (future information); never model inputs.",
}


def cards_text(cards: Iterable[Card]) -> str:
    """Cards as compact text, e.g. ``AsKh``."""
    return "".join(repr(card) for card in cards)


def observation_columns(obs: Observation) -> dict[str, Any]:
    """The public ``obs_`` columns for one observation, in log column order.

    Shared by the hand logger and by Phase 4 agents, so the features an
    agent computes during play are built exactly like the logged training
    data. It reads only public fields: never ``obs.hole_cards``.
    """
    this_street = [a for a in obs.history if a.street == obs.street]
    row: dict[str, Any] = {
        "obs_street": obs.street,
        "obs_position": obs.position,
        "obs_board": cards_text(obs.board_cards),
        "obs_pot_size": obs.pot_size,
        "obs_to_call": obs.to_call,
        "obs_pot_odds": pot_odds(obs.pot_size, obs.to_call),
        "obs_own_stack": obs.own_stack,
        "obs_opponent_stack": obs.opponent_stack,
        "obs_effective_stack": obs.effective_stack,
        "obs_own_bet": obs.own_bet,
        "obs_opponent_bet": obs.opponent_bet,
        "obs_big_blind": obs.big_blind,
        "obs_actions_this_street": len(this_street),
        "obs_raises_this_street": sum(a.category in RAISE_CATEGORIES for a in this_street),
        "obs_raises_this_hand": sum(a.category in RAISE_CATEGORIES for a in obs.history),
        "obs_history": ";".join(a.short() for a in obs.history),
    }
    for category in ActionCategory:
        legal = obs.legal(category)
        row[f"obs_legal_{category.value}"] = legal is not None
        row[f"obs_bet_to_{category.value}"] = legal.bet_to if legal else None
    return row


def decision_rows(match: MatchResult) -> list[dict[str, Any]]:
    """One dictionary per decision, with prefixed column names."""
    rows = []
    for hand in match.hands:
        result = hand.result
        for record in result.decisions:
            obs = record.observation
            seat = record.seat
            row: dict[str, Any] = {
                "id_match": match.match_id,
                "id_hand_index": hand.hand_index,
                "id_deal_index": hand.deal_index,
                "id_seat_swap": hand.seat_swap,
                "id_decision_index": record.decision_index,
                "id_deal_seed": result.deal_seed,
                **observation_columns(obs),
            }
            row.update(
                {
                    "act_category": record.action.category.value,
                    "act_bet_to": record.action.bet_to,
                    "act_chips_added": record.action.chips_added,
                    "act_is_all_in": record.action.is_all_in,
                    "hidden_actor_hole_cards": cards_text(result.hole_cards[seat]),
                    "hidden_opponent_hole_cards": cards_text(result.hole_cards[1 - seat]),
                    "hidden_agent_equity": record.decision.equity,
                    "hidden_agent_reason": record.decision.reason,
                    "label_actor_agent": record.agent_name,
                    "label_actor_role": "hero" if seat == hand.hero_seat else "villain",
                    "out_actor_net_chips": result.payoffs[seat],
                    "out_went_to_showdown": result.went_to_showdown,
                    "out_final_board": cards_text(result.board_cards),
                }
            )
            rows.append(row)
    return rows


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return only the observable ``obs_`` columns: the safe model inputs.

    Phase 3 should select model inputs with this function instead of
    dropping columns by hand, so hidden cards, outcomes, labels, ids (the
    deal seed determines the cards) and private bot parameters can never
    slip in.
    """
    return [column for column in frame.columns if column.startswith(FEATURE_PREFIX)]


def decisions_frame(match: MatchResult) -> pd.DataFrame:
    """Decision rows as a DataFrame; bet amounts stay whole numbers.

    ``obs_bet_to_*`` is empty where the action is illegal, so it uses
    pandas' nullable integer type instead of turning into decimals.
    """
    frame = pd.DataFrame(decision_rows(match))
    bet_columns = [f"obs_bet_to_{category.value}" for category in ActionCategory]
    return frame.astype({column: "Int64" for column in bet_columns})


def hand_rows(match: MatchResult) -> list[dict[str, Any]]:
    """One dictionary per hand."""
    rows = []
    for hand in match.hands:
        result = hand.result
        rows.append(
            {
                "match_id": match.match_id,
                "hand_index": hand.hand_index,
                "deal_index": hand.deal_index,
                "seat_swap": hand.seat_swap,
                "deal_seed": result.deal_seed,
                "hero_seat": hand.hero_seat,
                "agent_seat0": result.agent_names[0],
                "agent_seat1": result.agent_names[1],
                "hole_seat0": cards_text(result.hole_cards[0]),
                "hole_seat1": cards_text(result.hole_cards[1]),
                "board": cards_text(result.board_cards),
                "net_seat0": result.payoffs[0],
                "net_seat1": result.payoffs[1],
                "hero_net": hand.hero_net,
                "went_to_showdown": result.went_to_showdown,
                "decision_count": len(result.decisions),
            }
        )
    return rows


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(
    match: MatchResult, hero: Agent, villain: Agent, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run-level information, including private agent configurations."""
    paired = match.win_rate()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "match_id": match.match_id,
        "seed": match.seed,
        "deal_count": match.deal_count,
        "seat_swap": match.seat_swap,
        "hand_count": len(match.hands),
        "game": {
            "variant": "heads-up no-limit texas hold'em",
            "small_blind": SMALL_BLIND,
            "big_blind": BIG_BLIND,
            "starting_stack": STARTING_STACK,
            "antes": 0,
            "rake": 0,
            "stacks_reset_each_hand": True,
        },
        "agents": {"hero": hero.config(), "villain": villain.config()},
        "results": {
            "hero_bb_per_100": paired.bb_per_100,
            "standard_error": paired.standard_error,
            "ci_low": paired.ci_low,
            "ci_high": paired.ci_high,
            "confidence": paired.confidence,
            "ci_unit": "deal pair" if match.seat_swap else "hand",
            "unit_count": paired.unit_count,
        },
        "column_groups": COLUMN_GROUPS,
        "versions": {
            "hybridpoker": __version__,
            "pokerkit": version("pokerkit"),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
        },
        **({"extra": extra} if extra else {}),
    }


def write_run(
    match: MatchResult,
    hero: Agent,
    villain: Agent,
    out_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write decisions.csv, hands.csv, and metadata.json into ``out_dir``.

    Each file is written under a temporary name and renamed only when it is
    complete, and ``metadata.json`` is written last. A run directory is
    therefore complete if and only if it contains ``metadata.json``; an
    interrupted run never leaves files that look finished.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "decisions": out_dir / "decisions.csv",
        "hands": out_dir / "hands.csv",
        "metadata": out_dir / "metadata.json",
    }
    paths["metadata"].unlink(missing_ok=True)  # mark the directory incomplete first

    def replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        write(temporary)
        os.replace(temporary, path)

    replace_atomically(
        paths["decisions"],
        lambda p: decisions_frame(match).to_csv(p, index=False, lineterminator="\n"),
    )
    replace_atomically(
        paths["hands"],
        lambda p: pd.DataFrame(hand_rows(match)).to_csv(p, index=False, lineterminator="\n"),
    )
    metadata_text = json.dumps(run_metadata(match, hero, villain, extra), indent=2) + "\n"
    replace_atomically(paths["metadata"], lambda p: p.write_text(metadata_text, encoding="utf-8"))
    return paths
