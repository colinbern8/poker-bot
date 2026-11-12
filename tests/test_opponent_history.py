"""Tests for hybridpoker.opponent_history (Feature Set B).

The hand-built match below has known actions, so every count and smoothed
rate can be checked against a value computed by hand.
"""

import math

import numpy as np
import pandas as pd
import pytest

from hybridpoker.features import A_FEATURES
from hybridpoker.opponent_history import (
    B_FEATURES,
    HistoryPriors,
    feature_set_b,
    fit_priors,
    history_features,
    player_role,
    prior_hand_count,
)

PRIORS = HistoryPriors(
    open_rates={"check": 0.5, "half": 0.2, "pot": 0.2, "allin": 0.1},
    face_rates={"fold": 0.3, "call": 0.4, "half": 0.1, "pot": 0.1, "allin": 0.1},
    vpip=0.5,
    pfr=0.2,
    alpha=10.0,
)

# (deal, swap, position, street, to_call, action). With swap 0 the hero is on the button.
ROWS = [
    (0, 0, "button", "preflop", 1, "raise_pot"),         # r0 hero: raise (vpip, pfr)
    (0, 0, "big_blind", "preflop", 4, "fold"),           # r1 villain
    (0, 1, "button", "preflop", 1, "check_call"),        # r2 villain: limp (vpip)
    (0, 1, "big_blind", "preflop", 0, "check_call"),     # r3 hero: free check (not vpip)
    (0, 1, "big_blind", "flop", 0, "raise_half_pot"),    # r4 hero: bet
    (0, 1, "button", "flop", 3, "fold"),                 # r5 villain
    (1, 0, "button", "preflop", 1, "fold"),              # r6 hero
    (1, 1, "button", "preflop", 1, "raise_half_pot"),    # r7 villain (vpip, pfr)
    (1, 1, "big_blind", "preflop", 2, "check_call"),     # r8 hero: call (vpip)
    (2, 0, "button", "preflop", 1, "check_call"),        # r9 hero: limp (vpip)
]


def synthetic(match_id: str = "m1", rows=ROWS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id_match": match_id, "id_deal_index": d, "id_seat_swap": s, "obs_position": p,
             "obs_street": st, "obs_to_call": tc, "act_category": a}
            for d, s, p, st, tc, a in rows
        ]
    )


@pytest.fixture(scope="module")
def b() -> pd.DataFrame:
    return history_features(synthetic(), PRIORS)


def test_player_role_from_public_seating() -> None:
    roles = player_role(synthetic())
    assert list(roles) == ["hero", "villain", "villain", "hero", "hero", "villain",
                           "hero", "villain", "hero", "hero"]


def test_player_role_matches_true_role_on_simulated_matches(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    assert (player_role(frame) == frame["label_actor_role"]).all()


def test_first_deal_uses_priors_only(b) -> None:
    first = b.iloc[:6]
    assert (first["b_log_prior_hands"] == 0).all() and (first["b_log_prior_decisions"] == 0).all()
    assert np.allclose(first["b_open_check"], 0.5)
    assert np.allclose(first["b_face_fold"], 0.3)
    assert np.allclose(first["b_vpip"], 0.5) and np.allclose(first["b_pfr"], 0.2)


def test_hand_computed_hero_history_at_deal_1(b) -> None:
    r6 = b.iloc[6]
    # Hero in deal 0: 2 hands, 3 decisions; open: check + half-pot bet; facing: pot raise.
    assert r6["b_log_prior_hands"] == pytest.approx(math.log1p(2))
    assert r6["b_log_prior_decisions"] == pytest.approx(math.log1p(3))
    assert r6["b_open_check"] == pytest.approx((1 + 10 * 0.5) / (2 + 10))
    assert r6["b_open_half"] == pytest.approx((1 + 10 * 0.2) / (2 + 10))
    assert r6["b_face_pot"] == pytest.approx((1 + 10 * 0.1) / (1 + 10))
    assert r6["b_face_fold"] == pytest.approx((0 + 10 * 0.3) / (1 + 10))
    assert r6["b_vpip"] == pytest.approx((1 + 10 * 0.5) / (2 + 10))
    assert r6["b_pfr"] == pytest.approx((1 + 10 * 0.2) / (2 + 10))


def test_hand_computed_villain_history_at_deal_1(b) -> None:
    r7 = b.iloc[7]
    # Villain in deal 0: fold, limp, fold -- all facing a bet; never opened.
    assert r7["b_face_fold"] == pytest.approx((2 + 3) / (3 + 10))
    assert r7["b_face_call"] == pytest.approx((1 + 4) / (3 + 10))
    assert r7["b_open_check"] == pytest.approx(0.5)  # no open decisions: the prior
    assert r7["b_vpip"] == pytest.approx((1 + 5) / (2 + 10))
    assert r7["b_pfr"] == pytest.approx((0 + 2) / (2 + 10))


def test_hand_computed_hero_history_at_deal_2(b) -> None:
    r9 = b.iloc[9]
    assert r9["b_log_prior_hands"] == pytest.approx(math.log1p(4))
    assert r9["b_log_prior_decisions"] == pytest.approx(math.log1p(5))
    assert r9["b_face_fold"] == pytest.approx((1 + 3) / (3 + 10))
    assert r9["b_face_call"] == pytest.approx((1 + 4) / (3 + 10))
    assert r9["b_vpip"] == pytest.approx((2 + 5) / (4 + 10))
    assert r9["b_pfr"] == pytest.approx((1 + 2) / (4 + 10))


def test_both_seatings_of_a_deal_see_the_same_history(b) -> None:
    # r6 (deal 1, first seating) and r8 (deal 1, second seating) are both the hero.
    pd.testing.assert_series_equal(b.iloc[6], b.iloc[8], check_names=False)


def test_current_deal_actions_do_not_change_its_features() -> None:
    rows = list(ROWS)
    rows[6] = (1, 0, "button", "preflop", 1, "all_in")  # change the hero's deal-1 action
    changed = history_features(synthetic(rows=rows), PRIORS)
    original = history_features(synthetic(), PRIORS)
    pd.testing.assert_frame_equal(changed.iloc[:9], original.iloc[:9])  # deals 0 and 1 unchanged
    assert not changed.iloc[9].equals(original.iloc[9])  # deal 2 sees the change


def test_other_matches_do_not_affect_history() -> None:
    other = synthetic("m2", rows=[(d, s, p, st, tc, "fold" if tc else "check_call") for d, s, p, st, tc, _ in ROWS])
    together = history_features(pd.concat([synthetic(), other], ignore_index=True), PRIORS)
    alone = history_features(synthetic(), PRIORS)
    np.testing.assert_array_equal(together.iloc[:10].to_numpy(), alone.to_numpy())


def test_fit_priors_hand_computed() -> None:
    priors = fit_priors(synthetic())
    assert priors.open_rates == {"check": 0.5, "half": 0.5, "pot": 0.0, "allin": 0.0}
    assert priors.face_rates == {"fold": 3 / 8, "call": 3 / 8, "half": 1 / 8, "pot": 1 / 8, "allin": 0.0}
    assert priors.vpip == 5 / 8 and priors.pfr == 2 / 8
    assert priors.alpha == 10.0 and priors.fitted_rows == 10
    assert HistoryPriors.from_dict(priors.to_dict()) == priors


def test_fit_priors_needs_both_contexts() -> None:
    only_facing = [r for r in ROWS if r[4] > 0]
    with pytest.raises(ValueError):
        fit_priors(synthetic(rows=only_facing))


def test_feature_set_b_on_simulated_matches(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    features = feature_set_b(frame, tiny_dataset.priors)
    assert list(features.columns) == list(A_FEATURES) + list(B_FEATURES)
    assert not features.isna().any().any()
    rates = features[[c for c in B_FEATURES if not c.startswith("b_log")]]
    assert ((rates >= 0) & (rates <= 1)).all().all()

    hands = prior_hand_count(features)
    first_deal = frame["id_deal_index"] == 0
    assert (hands[first_deal] == 0).all()
    ordered = pd.DataFrame({"m": frame["id_match"], "role": player_role(frame),
                            "deal": frame["id_deal_index"], "hands": hands}).sort_values(["m", "role", "deal"])
    assert (ordered.groupby(["m", "role"])["hands"].diff().dropna() >= 0).all()  # never decreases
