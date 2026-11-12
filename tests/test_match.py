"""Tests for hybridpoker.match (seat-swapped matches with paired seeds)."""

import pytest

from hybridpoker.agents import make_agent
from hybridpoker.game import BIG_BLIND_SEAT, BUTTON_SEAT
from hybridpoker.match import run_match

FAST = 30  # equity samples, keeps tests quick


def test_each_deal_is_played_in_both_seats_with_the_same_cards() -> None:
    match = run_match(make_agent("random"), make_agent("nit", FAST), deal_count=10, seed=4)

    assert len(match.hands) == 20
    for first, second in zip(match.hands[0::2], match.hands[1::2]):
        assert first.deal_index == second.deal_index
        assert (first.seat_swap, second.seat_swap) == (0, 1)
        assert (first.hero_seat, second.hero_seat) == (BUTTON_SEAT, BIG_BLIND_SEAT)
        assert first.result.deal_seed == second.result.deal_seed
        assert first.result.hole_cards == second.result.hole_cards
        # So the hero holds each seat's cards exactly once.
        hero_cards = {
            first.result.hole_cards[first.hero_seat],
            second.result.hole_cards[second.hero_seat],
        }
        assert hero_cards == set(first.result.hole_cards)


def test_identical_strategies_break_even_on_every_pair() -> None:
    # Both agents always check or call, so every hand goes to showdown and
    # the hero wins in one seat exactly what it loses in the other.
    match = run_match(
        make_agent("calling_station", FAST), make_agent("calling_station", FAST), 25, seed=8
    )
    assert match.hero_net_by_deal() == [0] * 25
    paired = match.win_rate()
    assert paired.bb_per_100 == 0.0 and paired.standard_error == 0.0
    # Treating the hands as independent wrongly reports uncertainty.
    assert match.naive_win_rate().standard_error > 0


def test_same_seed_reproduces_the_match() -> None:
    first = run_match(make_agent("probability", FAST), make_agent("maniac", FAST), 8, seed=11)
    second = run_match(make_agent("probability", FAST), make_agent("maniac", FAST), 8, seed=11)
    assert first.hands == second.hands


def test_different_seeds_give_different_deals() -> None:
    a = run_match(make_agent("random"), make_agent("random"), 5, seed=1)
    b = run_match(make_agent("random"), make_agent("random"), 5, seed=2)
    assert [h.result.hole_cards for h in a.hands] != [h.result.hole_cards for h in b.hands]


def test_without_seat_swap_every_hand_is_a_unit() -> None:
    match = run_match(make_agent("random"), make_agent("random"), 12, seed=3, seat_swap=False)
    assert len(match.hands) == 12
    assert all(h.hero_seat == BUTTON_SEAT for h in match.hands)
    assert match.win_rate().hands_per_unit == 1
    assert match.win_rate().unit_count == 12


def test_hero_net_by_deal_sums_both_hands() -> None:
    match = run_match(make_agent("random"), make_agent("random"), 6, seed=5)
    by_deal = match.hero_net_by_deal()
    for deal in range(6):
        assert by_deal[deal] == sum(h.hero_net for h in match.hands if h.deal_index == deal)
    assert match.win_rate().bb_per_100 == pytest.approx(sum(by_deal) / 12 / 2 * 100)


def test_progress_callback_and_validation() -> None:
    calls = []
    run_match(make_agent("random"), make_agent("random"), 3, seed=0, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 3), (2, 3), (3, 3)]
    with pytest.raises(ValueError):
        run_match(make_agent("random"), make_agent("random"), 0, seed=0)
