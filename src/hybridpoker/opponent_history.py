"""Feature Set B: the acting player's earlier public behaviour in the match.

For a decision in deal *d*, Feature Set B summarizes how the **same player**
acted in deals 0 .. d-1 of the **same match**. Both seat-swapped hands of
deal *d* are excluded: they share cards and board, so using the mirror
hand would leak information a real player could never have.

**Who is "the same player"?** In a seat-swapped match the ``hero`` role
sits on the button when ``id_seat_swap == 0`` and in the big blind when
``id_seat_swap == 1``. The player role is therefore derived from public
seating information (:func:`player_role`), never from the private
``label_`` columns.

**Smoothing.** Early in a match there is little history. Each rate is
shrunk toward a population prior::

    rate = (count + alpha * prior) / (n + alpha),   alpha = 10

The priors are population rates fitted on training rows only
(:func:`fit_priors`) and then frozen. With no history every rate equals
its prior, and the observation-count features are 0.

Only actions are used. Earlier showdown cards are deliberately not used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from hybridpoker.features import RAISE_VALUES, feature_set_a

SMOOTHING_ALPHA = 10.0
HISTORY_KEYS = ("id_match", "id_deal_index", "id_seat_swap")
B_INPUT_COLUMNS = (*HISTORY_KEYS, "obs_position", "obs_street", "obs_to_call", "act_category")

# (feature suffix, action category) when nobody has bet / when facing a bet.
OPEN_ACTIONS = (("check", "check_call"), ("half", "raise_half_pot"), ("pot", "raise_pot"), ("allin", "all_in"))
FACE_ACTIONS = (
    ("fold", "fold"), ("call", "check_call"), ("half", "raise_half_pot"), ("pot", "raise_pot"), ("allin", "all_in")
)

B_FEATURES = (
    "b_log_prior_hands",
    "b_log_prior_decisions",
    *(f"b_open_{name}" for name, _ in OPEN_ACTIONS),
    *(f"b_face_{name}" for name, _ in FACE_ACTIONS),
    "b_vpip",
    "b_pfr",
)

_COUNT_COLUMNS = (
    "hands", "decisions", "preflop_hands", "vpip_hands", "pfr_hands", "n_open", "n_face",
    *(f"open_{name}" for name, _ in OPEN_ACTIONS),
    *(f"face_{name}" for name, _ in FACE_ACTIONS),
)


def player_role(frame: pd.DataFrame) -> pd.Series:
    """Match-level player role ("hero"/"villain") from public seating only."""
    on_button = frame["obs_position"] == "button"
    first_seating = frame["id_seat_swap"] == 0
    return pd.Series(np.where(on_button == first_seating, "hero", "villain"), index=frame.index)


@dataclass(frozen=True)
class HistoryPriors:
    """Population action rates used to smooth sparse history."""

    open_rates: dict[str, float]
    face_rates: dict[str, float]
    vpip: float
    pfr: float
    alpha: float = SMOOTHING_ALPHA
    fitted_rows: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> HistoryPriors:
        return cls(**data)


def _deal_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (match, role, deal): counts of the player's own decisions and hands."""
    view = frame.loc[:, list(B_INPUT_COLUMNS)].copy()
    view["role"] = player_role(view)
    category = view["act_category"]
    facing = view["obs_to_call"] > 0
    preflop = view["obs_street"] == "preflop"
    is_raise = category.isin(RAISE_VALUES)

    view["decisions"] = 1
    view["n_open"] = (~facing).astype(int)
    view["n_face"] = facing.astype(int)
    for name, value in OPEN_ACTIONS:
        view[f"open_{name}"] = ((~facing) & (category == value)).astype(int)
    for name, value in FACE_ACTIONS:
        view[f"face_{name}"] = (facing & (category == value)).astype(int)
    view["_preflop"] = preflop.astype(int)
    # Voluntarily put chips in preflop: any raise, or a call that cost chips.
    view["_vpip"] = (preflop & (is_raise | (facing & (category == "check_call")))).astype(int)
    view["_pfr"] = (preflop & is_raise).astype(int)

    group = ["id_match", "role", "id_deal_index"]
    per_hand = view.groupby([*group, "id_seat_swap"], sort=True).agg(
        _preflop=("_preflop", "max"), _vpip=("_vpip", "max"), _pfr=("_pfr", "max")
    )
    hands = per_hand.groupby(level=group).agg(
        hands=("_preflop", "size"),
        preflop_hands=("_preflop", "sum"),
        vpip_hands=("_vpip", "sum"),
        pfr_hands=("_pfr", "sum"),
    )
    decision_columns = [c for c in _COUNT_COLUMNS if c not in hands.columns]
    decisions = view.groupby(group, sort=True)[decision_columns].sum()
    return hands.join(decisions).loc[:, list(_COUNT_COLUMNS)]


def prior_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """For every (match, role, deal): the player's totals over **earlier** deals only."""
    counts = _deal_counts(frame)
    cumulative = counts.groupby(level=["id_match", "role"]).cumsum()
    return cumulative - counts  # exclusive: deals before this one


def fit_priors(train_frame: pd.DataFrame, alpha: float = SMOOTHING_ALPHA) -> HistoryPriors:
    """Population rates from training rows only."""
    totals = _deal_counts(train_frame).sum()
    if totals["n_open"] == 0 or totals["n_face"] == 0 or totals["preflop_hands"] == 0:
        raise ValueError("Training rows are too few to estimate history priors.")
    return HistoryPriors(
        open_rates={name: float(totals[f"open_{name}"] / totals["n_open"]) for name, _ in OPEN_ACTIONS},
        face_rates={name: float(totals[f"face_{name}"] / totals["n_face"]) for name, _ in FACE_ACTIONS},
        vpip=float(totals["vpip_hands"] / totals["preflop_hands"]),
        pfr=float(totals["pfr_hands"] / totals["preflop_hands"]),
        alpha=alpha,
        fitted_rows=len(train_frame),
    )


def history_features(frame: pd.DataFrame, priors: HistoryPriors) -> pd.DataFrame:
    """The ``b_`` features for every row, in the fixed order ``B_FEATURES``."""
    earlier = prior_counts(frame)
    keys = pd.DataFrame(
        {"id_match": frame["id_match"], "role": player_role(frame), "id_deal_index": frame["id_deal_index"]},
        index=frame.index,
    )
    counts = keys.join(earlier, on=["id_match", "role", "id_deal_index"]).loc[:, list(_COUNT_COLUMNS)]
    if counts.isna().any().any():
        raise RuntimeError("History counts are missing for some rows.")

    a = priors.alpha
    out = pd.DataFrame(index=frame.index)
    out["b_log_prior_hands"] = np.log1p(counts["hands"].astype(float))
    out["b_log_prior_decisions"] = np.log1p(counts["decisions"].astype(float))
    for name, _ in OPEN_ACTIONS:
        out[f"b_open_{name}"] = (counts[f"open_{name}"] + a * priors.open_rates[name]) / (counts["n_open"] + a)
    for name, _ in FACE_ACTIONS:
        out[f"b_face_{name}"] = (counts[f"face_{name}"] + a * priors.face_rates[name]) / (counts["n_face"] + a)
    out["b_vpip"] = (counts["vpip_hands"] + a * priors.vpip) / (counts["preflop_hands"] + a)
    out["b_pfr"] = (counts["pfr_hands"] + a * priors.pfr) / (counts["preflop_hands"] + a)
    return out.loc[:, list(B_FEATURES)].astype(float)


def feature_set_b(frame: pd.DataFrame, priors: HistoryPriors) -> pd.DataFrame:
    """Feature Set B: Feature Set A plus the history features."""
    return pd.concat([feature_set_a(frame), history_features(frame, priors)], axis=1)


def prior_hand_count(features_b: pd.DataFrame) -> pd.Series:
    """Number of earlier hands observed, recovered from ``b_log_prior_hands``."""
    return features_b["b_log_prior_hands"].map(lambda x: int(round(math.expm1(x))))
