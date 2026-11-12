"""Feature Set A: numbers describing the current hand, from public information only.

Every feature is computed from an explicit list of ``obs_`` columns
(``A_INPUT_COLUMNS``). The builder selects those columns first, so it
physically cannot read hole cards, labels, outcomes or identifiers.

Groups (all chip amounts are in big blinds):

* **Situation**: street, position, pot, amount to call, pot odds, stacks,
  stack-to-pot ratio, bets, whether the player faces a bet.
* **Legal actions**: which of the five actions are legal and their sizes.
* **Board texture** (from ``obs_board``): number of cards, highest rank,
  broadway cards, pairs and trips on board, largest suit count, straight
  connectivity and rank spread.
* **Betting history** (from ``obs_history``, actions earlier in this hand):
  raises by each player, the previous action, the preflop aggressor,
  whether the opponent is all-in, and streets with a raise.

Feature Set B (``opponent_history.py``) adds the acting player's earlier
behaviour in the match.
"""

from __future__ import annotations

import pandas as pd

from hybridpoker.actions import ActionCategory
from hybridpoker.game import BIG_BLIND, STREET_NAMES

CATEGORIES = tuple(c.value for c in ActionCategory)
RAISE_VALUES = frozenset(
    c.value for c in (ActionCategory.RAISE_HALF_POT, ActionCategory.RAISE_POT, ActionCategory.ALL_IN)
)
RANK_VALUES = {rank: value for value, rank in enumerate("23456789TJQKA", start=2)}
PREVIOUS_ACTIONS = ("none", "check_call", "raise_half_pot", "raise_pot", "all_in")

A_INPUT_COLUMNS = (
    "obs_street",
    "obs_position",
    "obs_board",
    "obs_pot_size",
    "obs_to_call",
    "obs_pot_odds",
    "obs_own_stack",
    "obs_opponent_stack",
    "obs_effective_stack",
    "obs_own_bet",
    "obs_opponent_bet",
    "obs_actions_this_street",
    "obs_raises_this_street",
    "obs_raises_this_hand",
    "obs_history",
    *(f"obs_legal_{c}" for c in CATEGORIES),
    *(f"obs_bet_to_{c}" for c in CATEGORIES),
)

SITUATION_FEATURES = (
    *(f"a_street_{s}" for s in STREET_NAMES),
    "a_position_button",
    "a_pot_bb",
    "a_to_call_bb",
    "a_pot_odds",
    "a_own_stack_bb",
    "a_opp_stack_bb",
    "a_eff_stack_bb",
    "a_spr",
    "a_own_bet_bb",
    "a_opp_bet_bb",
    "a_facing_bet",
)
LEGAL_FEATURES = (
    *(f"a_legal_{c}" for c in CATEGORIES),
    *(f"a_bet_to_{c}_bb" for c in CATEGORIES),
)
BOARD_FEATURES = (
    "a_board_cards",
    "a_board_high_rank",
    "a_board_broadway",
    "a_board_paired",
    "a_board_trips",
    "a_board_max_suit",
    "a_board_straightness",
    "a_board_rank_spread",
)
HISTORY_FEATURES = (
    "a_actions_this_street",
    "a_raises_this_street",
    "a_raises_this_hand",
    "a_actor_raises_hand",
    "a_opp_raises_hand",
    *(f"a_prev_{p}" for p in PREVIOUS_ACTIONS),
    "a_preflop_aggressor_actor",
    "a_preflop_aggressor_opp",
    "a_opp_all_in",
    "a_streets_with_raise",
)
A_FEATURES = SITUATION_FEATURES + LEGAL_FEATURES + BOARD_FEATURES + HISTORY_FEATURES
FEATURE_GROUPS_A = {
    "situation": SITUATION_FEATURES,
    "legal_actions": LEGAL_FEATURES,
    "board_texture": BOARD_FEATURES,
    "betting_history": HISTORY_FEATURES,
}


def parse_board(text: str) -> list[tuple[int, str]]:
    """``"KhJc5s"`` -> ``[(13, "h"), (11, "c"), (5, "s")]``."""
    if not text:
        return []
    if len(text) % 2:
        raise ValueError(f"Malformed board text {text!r}.")
    return [(RANK_VALUES[text[i]], text[i + 1]) for i in range(0, len(text), 2)]


def board_texture(text: str) -> dict[str, float]:
    """Public board-texture features for one board."""
    cards = parse_board(text)
    ranks = [rank for rank, _ in cards]
    distinct = set(ranks)
    rank_counts = [ranks.count(r) for r in distinct]
    suits = [suit for _, suit in cards]
    straight_ranks = distinct | ({1} if 14 in distinct else set())  # ace also plays low
    straightness = max(
        (len(straight_ranks & set(range(low, low + 5))) for low in range(1, 11)), default=0
    )
    return {
        "a_board_cards": len(cards),
        "a_board_high_rank": max(ranks, default=0),
        "a_board_broadway": sum(rank >= 10 for rank in ranks),
        "a_board_paired": int(any(count >= 2 for count in rank_counts)),
        "a_board_trips": int(any(count >= 3 for count in rank_counts)),
        "a_board_max_suit": max((suits.count(s) for s in set(suits)), default=0),
        "a_board_straightness": straightness if cards else 0,
        "a_board_rank_spread": (max(distinct) - min(distinct)) if len(distinct) >= 2 else 0,
    }


def parse_history(text: str) -> list[tuple[str, str, str, int]]:
    """``"preflop:button:raise_pot:6;..."`` -> ``[("preflop", "button", "raise_pot", 6), ...]``."""
    if not text:
        return []
    entries = []
    for item in text.split(";"):
        street, position, category, bet_to = item.split(":")
        if category not in CATEGORIES:
            raise ValueError(f"Unknown action category {category!r} in history.")
        entries.append((street, position, category, int(bet_to)))
    return entries


def history_summary(text: str, actor_position: str) -> dict[str, float]:
    """Betting-history features relative to the acting player."""
    entries = parse_history(text)
    raises = [e for e in entries if e[2] in RAISE_VALUES]
    previous = entries[-1][2] if entries else "none"
    if previous not in PREVIOUS_ACTIONS:  # a fold always ends the hand, so it can't precede a decision
        raise ValueError(f"Unexpected previous action {previous!r}.")
    preflop_raises = [e for e in raises if e[0] == "preflop"]
    aggressor = preflop_raises[-1][1] if preflop_raises else None
    features = {
        "a_actor_raises_hand": sum(e[1] == actor_position for e in raises),
        "a_opp_raises_hand": sum(e[1] != actor_position for e in raises),
        "a_preflop_aggressor_actor": int(aggressor == actor_position),
        "a_preflop_aggressor_opp": int(aggressor is not None and aggressor != actor_position),
        "a_streets_with_raise": len({e[0] for e in raises}),
    }
    for option in PREVIOUS_ACTIONS:
        features[f"a_prev_{option}"] = int(previous == option)
    return features


def feature_set_a(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature Set A for every row, in the fixed column order ``A_FEATURES``."""
    obs = frame.loc[:, list(A_INPUT_COLUMNS)]  # the only columns this function reads
    bb = float(BIG_BLIND)
    out = pd.DataFrame(index=obs.index)

    for street in STREET_NAMES:
        out[f"a_street_{street}"] = (obs["obs_street"] == street).astype(int)
    out["a_position_button"] = (obs["obs_position"] == "button").astype(int)
    out["a_pot_bb"] = obs["obs_pot_size"] / bb
    out["a_to_call_bb"] = obs["obs_to_call"] / bb
    out["a_pot_odds"] = obs["obs_pot_odds"].astype(float)
    out["a_own_stack_bb"] = obs["obs_own_stack"] / bb
    out["a_opp_stack_bb"] = obs["obs_opponent_stack"] / bb
    out["a_eff_stack_bb"] = obs["obs_effective_stack"] / bb
    out["a_spr"] = obs["obs_effective_stack"] / obs["obs_pot_size"]  # pot >= blinds, never 0
    out["a_own_bet_bb"] = obs["obs_own_bet"] / bb
    out["a_opp_bet_bb"] = obs["obs_opponent_bet"] / bb
    out["a_facing_bet"] = (obs["obs_to_call"] > 0).astype(int)

    for category in CATEGORIES:
        out[f"a_legal_{category}"] = obs[f"obs_legal_{category}"].astype(bool).astype(int)
    for category in CATEGORIES:
        sizes = pd.to_numeric(obs[f"obs_bet_to_{category}"], errors="raise").astype(float)
        out[f"a_bet_to_{category}_bb"] = sizes.fillna(0.0) / bb

    board = pd.DataFrame([board_texture(text) for text in obs["obs_board"]], index=obs.index)
    history = pd.DataFrame(
        [history_summary(text, pos) for text, pos in zip(obs["obs_history"], obs["obs_position"])],
        index=obs.index,
    )
    out = pd.concat([out, board, history], axis=1)
    out["a_actions_this_street"] = obs["obs_actions_this_street"]
    out["a_raises_this_street"] = obs["obs_raises_this_street"]
    out["a_raises_this_hand"] = obs["obs_raises_this_hand"]
    out["a_opp_all_in"] = (obs["obs_opponent_stack"] == 0).astype(int)
    return out.loc[:, list(A_FEATURES)].astype(float)
