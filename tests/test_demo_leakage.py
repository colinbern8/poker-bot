"""No hidden cards leak in either direction, and AI explanations appear only after the hand.

* The human's view during a hand holds only their own cards and the board.
* After a hand, the AI's cards are shown only if the hand reached a showdown;
  its explanations (equity, EVs, predicted responses) are shown either way.
* The AI never receives the human's hole cards.

Cards are found by walking every field of the views (dataclasses, tuples,
lists, dicts); card text is also searched for in every string.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from pokerkit import Card

from demo_support import ml_agent
from hybridpoker.hand_log import cards_text
from hybridpoker_demo.controller import CompletedHand, HeadsUpController, HumanTurn, Ticket

SEEDS = (3, 19, 101)
HANDS = 10


def walk(obj, seen=None) -> Iterator:
    """Every value reachable from ``obj`` through dataclass fields and containers.

    A :class:`Ticket` is not entered: its ``match_id`` is a random hex string
    that can contain card-like text (e.g. ``...548d...``) by chance.
    """
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, Ticket):
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, (type, Card)):
        for field in dataclasses.fields(obj):
            yield from walk(getattr(obj, field.name), seen)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk(key, seen)
            yield from walk(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            yield from walk(value, seen)


def cards_in(obj) -> set[Card]:
    return {v for v in walk(obj) if isinstance(v, Card)}


def strings_in(obj) -> list[str]:
    return [v for v in walk(obj) if isinstance(v, str)]


def assert_hidden(obj, cards) -> None:
    assert not cards_in(obj) & set(cards)
    texts = [cards_text([c]) for c in cards]
    assert not any(t in s for s in strings_in(obj) for t in texts)


def play(tiny_bundle, seed):
    """Play a match with varied actions, yielding (view, full result or None) as it goes."""
    controller = HeadsUpController(ml_agent(tiny_bundle), match_seed=seed)
    for h in range(HANDS):
        if h:
            controller.next_hand()
        while isinstance(controller.view, HumanTurn):
            yield controller, controller.view
            legal = controller.view.legal_categories
            controller.submit(controller.view.ticket, legal[(h + len(controller.view.observation.history)) % len(legal)])
        yield controller, controller.view


def test_no_hidden_cards_leak_and_explanations_wait_for_the_end_of_the_hand(tiny_bundle) -> None:
    showdowns = folds = turns = 0
    for seed in SEEDS:
        for controller, view in play(tiny_bundle, seed):
            if isinstance(view, HumanTurn):
                turns += 1
                assert {f.name for f in dataclasses.fields(view)} == {"ticket", "hand_index", "observation"}
                own = set(view.observation.hole_cards)
                assert len(own) == 2
                assert cards_in(view) <= own | set(view.observation.board_cards)
                # The AI's cards for this hand (known only to the test, from a full replay at the end)
                # are checked below against the stored result.
                continue
            result = controller._results[view.hand_index]
            human_seat = 1 if view.human_position == "button" else 0
            ai_cards, human_cards = result.hole_cards[1 - human_seat], result.hole_cards[human_seat]
            assert view.human_hole_cards == human_cards
            if view.went_to_showdown:
                showdowns += 1
                assert view.ai_hole_cards == ai_cards
            else:
                folds += 1
                assert view.ai_hole_cards is None
                assert_hidden(view, ai_cards)
            # Explanations are present after every hand in which the AI acted.
            ai_moves = [d for d in result.decisions if d.seat != human_seat]
            assert len(view.ai_decisions) == len(ai_moves)
            assert all(r.diagnostics is not None for r in view.ai_decisions)
            assert_hidden([r.diagnostics for r in view.ai_decisions], human_cards)
            # The AI never saw the human's cards.
            for move in ai_moves:
                assert move.observation.hole_cards == ai_cards
                assert not cards_in(move.observation) & set(human_cards)
    assert turns and showdowns and folds  # both kinds of hand were exercised


def test_human_turns_never_contain_the_ai_cards(tiny_bundle) -> None:
    for seed in SEEDS:
        pending = []
        for controller, view in play(tiny_bundle, seed):
            if isinstance(view, HumanTurn):
                pending.append(view)
                continue
            result = controller._results[view.hand_index]
            human_seat = 1 if view.human_position == "button" else 0
            for turn in pending:
                assert_hidden(turn, result.hole_cards[1 - human_seat])
            pending = []
