"""The Phase 5D poker table (table_view.py): layout, colours, hidden information, no animations."""

from __future__ import annotations

import re

import pytest

from demo_support import ml_agent
from hybridpoker.game import STARTING_STACK
from hybridpoker.simulator import shuffled_deck
from hybridpoker_demo import table_view as tv
from hybridpoker_demo.controller import HeadsUpController, HumanTurn
from hybridpoker_demo.presentation import card_text
from hybridpoker_demo.seeds import hand_setup

AI = "Hybrid ML Agent"
BACK = 'class="hp-card hp-back"'


def test_palette_is_the_approved_earth_tones() -> None:
    assert tv.PALETTE["forest"] == "#354B3E" and tv.PALETTE["cream"] == "#EEE9DF" and tv.PALETTE["card"] == "#F8F5ED"
    assert {"sage", "walnut", "charcoal", "brick"} <= set(tv.PALETTE)


@pytest.mark.parametrize("foreground, background", tv.TEXT_PAIRS)
def test_text_colours_meet_wcag_aa_contrast(foreground, background) -> None:
    assert tv.contrast_ratio(tv.PALETTE[foreground], tv.PALETTE[background]) >= 4.5


def test_contrast_ratio_matches_known_values() -> None:
    assert tv.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert tv.contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


def test_no_animations_transitions_or_gradients() -> None:
    css = tv.TABLE_CSS.lower()
    for forbidden in ("animation", "transition", "@keyframes", "gradient", "box-shadow"):
        assert forbidden not in css, forbidden


def turns_and_hands(bundle, seed: int, hands: int):
    """Play a short match, yielding every human turn and finished hand with the AI's real cards."""
    controller = HeadsUpController(ml_agent(bundle), match_seed=seed)
    for h in range(hands):
        if h:
            controller.next_hand()
        setup = hand_setup(seed, h)
        deck = shuffled_deck(setup.deal_seed)
        ai_cards = [card_text(deck[setup.ai_seat]), card_text(deck[setup.ai_seat + 2])]
        while isinstance(controller.view, HumanTurn):
            yield controller.view, ai_cards
            legal = controller.view.legal_categories
            controller.submit(controller.view.ticket, legal[(h + len(controller.view.observation.history)) % len(legal)])
        yield controller.view, ai_cards


def test_table_layout_and_hidden_information(tiny_bundle) -> None:
    turns = showdowns = folds = 0
    for seed in (3, 19):
        for view, ai_cards in turns_and_hands(tiny_bundle, seed, 8):
            if isinstance(view, HumanTurn):
                turns += 1
                obs = view.observation
                html = tv.table_during_hand(view, AI)
                # Layout order: AI seat at the top, then the pot and board, then the human seat.
                assert html.index(AI) < html.index("Pot") < html.index("You ·")
                assert html.count(BACK) == 2 and not any(card in html for card in ai_cards)
                assert all(card_text(c) in html for c in obs.hole_cards)
                assert all(card_text(c) in html for c in obs.board_cards)
                assert html.count("hp-slot") == 5 - len(obs.board_cards)
                assert f"Stack <b>{obs.opponent_stack}</b> · Bet <b>{obs.opponent_bet}</b>" in html
                assert f"Stack <b>{obs.own_stack}</b> · Bet <b>{obs.own_bet}</b>" in html
                assert f"Pot <b>{obs.pot_size}</b>" in html
                continue
            html = tv.table_after_hand(view, AI, STARTING_STACK)
            assert f"Stack <b>{STARTING_STACK + view.human_net}</b>" in html
            assert f"Stack <b>{STARTING_STACK + view.ai_net}</b>" in html
            if view.went_to_showdown:
                showdowns += 1
                assert html.count(BACK) == 0 and all(card in html for card in ai_cards) and "Showdown" in html
            else:
                folds += 1
                assert html.count(BACK) == 2 and not any(card in html for card in ai_cards) and "Hand over" in html
    assert turns and showdowns and folds


def test_red_suits_use_the_brick_class_and_text_is_escaped(tiny_bundle) -> None:
    view, _ = next(turns_and_hands(tiny_bundle, 3, 1))
    html = tv.table_during_hand(view, "<script>alert(1)</script>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    for card in view.observation.hole_cards:
        face = re.search(rf'<div class="hp-card hp-face( hp-red)?">{re.escape(card_text(card))}</div>', html)
        assert face is not None and bool(face.group(1)) == (card.suit.value in "hd")
