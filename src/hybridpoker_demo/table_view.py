"""The heads-up poker table as HTML (Phase 5D): plain functions, no Streamlit.

Approved layout and colours:

* a large rounded forest-green table (#354B3E) with a thin walnut rail on a
  warm cream page (#EEE9DF);
* the AI centred at the top with two cards (face-down during the hand);
* the pot and five community-card positions in the middle;
* the human's two cards centred at the bottom;
* "You" and "Hybrid AI" as seat names, the position as a small tag;
* stacks and bets at each seat (after the hand: "This hand: +4 chips");
* the AI's latest recorded action, announced between its seat and the pot.

Cards are warm white (#F8F5ED); hearts and diamonds are a muted brick red,
spades and clubs charcoal. There are no animations, transitions or
gradients (a test guards this).

**Hidden information.** The table is built only from the controller's public
views. During a hand (:class:`HumanTurn`) the AI's cards are always drawn as
backs; after the hand (:class:`CompletedHand`) its faces appear only if the
hand reached a showdown, exactly as ``CompletedHand.ai_hole_cards`` allows.
All text is HTML-escaped.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

from pokerkit import Card

from hybridpoker_demo.controller import CompletedHand, HumanTurn
from hybridpoker_demo.presentation import POSITION_NAMES, STREET_NAMES, card_text, latest_ai_action, signed

PALETTE = {
    "forest": "#354B3E",   # table
    "cream": "#EEE9DF",    # page background, text on the table
    "card": "#F8F5ED",     # card faces
    "sage": "#A3AE96",     # borders, empty slots, secondary accents
    "walnut": "#6B4F3A",   # table rail, card backs, main button
    "charcoal": "#2F2F2D", # text, spades and clubs
    "brick": "#9A3B30",    # hearts and diamonds
}

# Text/background pairs used on screen; each must reach WCAG AA (4.5:1) contrast.
TEXT_PAIRS = (
    ("charcoal", "cream"),   # page text
    ("cream", "forest"),     # names, stacks, pot on the table
    ("charcoal", "card"),    # spades and clubs
    ("brick", "card"),       # hearts and diamonds
    ("cream", "walnut"),     # main button label
    ("charcoal", "card"),    # AI action announcement
    ("walnut", "card"),      # street tag in the announcement
)

BOARD_SLOTS = 5
_RED_SUITS = frozenset("hd")

TABLE_CSS = f"""
<style>
/* Keep the table, the buttons beneath it and the explanations in one centred column on wide screens. */
[data-testid="stMainBlockContainer"] {{ max-width: 1000px; margin: 0 auto; }}
.hp-table {{
  max-width: 820px; margin: 0.25rem auto 0.5rem auto; padding: 20px 32px;
  background: {PALETTE['forest']}; border: 9px solid {PALETTE['walnut']}; border-radius: 190px;
  color: {PALETTE['cream']}; display: flex; flex-direction: column; align-items: center; gap: 10px;
  font-family: inherit;
}}
.hp-seat {{ display: flex; flex-direction: column; align-items: center; gap: 6px; }}
.hp-name {{ font-weight: 600; font-size: 1rem; letter-spacing: 0.01em; display: flex; align-items: center; gap: 8px; }}
.hp-pos {{ font-weight: 500; font-size: 0.72rem; padding: 1px 7px; border: 1px solid {PALETTE['sage']}; border-radius: 999px; }}
.hp-stats {{ font-size: 0.9rem; }}
.hp-stats b {{ font-weight: 600; }}
.hp-cards {{ display: flex; gap: 8px; justify-content: center; }}
.hp-card {{
  width: 58px; height: 82px; border-radius: 6px; box-sizing: border-box;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.45rem; font-weight: 600; line-height: 1;
}}
.hp-face {{ background: {PALETTE['card']}; border: 1px solid #D9D2C3; color: {PALETTE['charcoal']}; }}
.hp-face.hp-red {{ color: {PALETTE['brick']}; }}
.hp-back {{ background: {PALETTE['walnut']}; border: 1px solid {PALETTE['card']}; padding: 5px; }}
.hp-back span {{ display: block; width: 100%; height: 100%; border: 1px solid {PALETTE['sage']}; border-radius: 3px; }}
.hp-slot {{ border: 1.5px dashed rgba(238, 233, 223, 0.35); }}
.hp-middle {{ display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 4px 0; }}
.hp-pot {{ font-size: 0.95rem; }}
.hp-pot b {{ font-weight: 600; font-size: 1.05rem; }}
/* The AI's latest action: a calm, persistent label in the flow of the table (it never covers cards). */
.hp-notice {{
  min-height: 2.3rem; box-sizing: border-box; max-width: 100%; padding: 5px 16px; border-radius: 999px;
  background: {PALETTE['card']}; border: 2px solid {PALETTE['walnut']}; color: {PALETTE['charcoal']};
  font-size: 1.05rem; font-weight: 600; text-align: center; line-height: 1.3;
}}
.hp-notice-street {{ color: {PALETTE['walnut']}; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; margin-right: 8px; }}
.hp-notice-empty {{ visibility: hidden; }}
/* Phones: a less rounded table and slightly smaller cards, so no text meets the rail. */
@media (max-width: 640px) {{
  .hp-table {{ border-radius: 56px; padding: 18px 12px; border-width: 7px; }}
  .hp-card {{ width: 48px; height: 68px; font-size: 1.2rem; }}
  .hp-notice {{ font-size: 1rem; padding: 5px 12px; }}
}}
</style>
"""


# --- Contrast (WCAG 2.x) ------------------------------------------------------------------------


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


# --- Pieces ---------------------------------------------------------------------------------------


def card_face_html(card: Card) -> str:
    red = " hp-red" if card.suit.value in _RED_SUITS else ""
    return f'<div class="hp-card hp-face{red}">{html.escape(card_text(card))}</div>'


def card_back_html() -> str:
    return '<div class="hp-card hp-back" title="face-down card"><span></span></div>'


def empty_slot_html() -> str:
    return '<div class="hp-card hp-slot"></div>'


def _cards(cards: Sequence[Card] | None, hidden_count: int = 2) -> str:
    inner = "".join(card_face_html(c) for c in cards) if cards is not None else card_back_html() * hidden_count
    return f'<div class="hp-cards">{inner}</div>'


def _board(cards: Sequence[Card]) -> str:
    faces = "".join(card_face_html(c) for c in cards)
    return f'<div class="hp-cards">{faces}{empty_slot_html() * (BOARD_SLOTS - len(cards))}</div>'


def _seat(name: str, position: str, stats: str, cards_html: str, cards_first: bool) -> str:
    label = (f'<div class="hp-name">{html.escape(name)}'
             f'<span class="hp-pos">{html.escape(POSITION_NAMES[position])}</span></div>')
    stats_html = f'<div class="hp-stats">{stats}</div>'
    parts = (cards_html, label, stats_html) if cards_first else (label, stats_html, cards_html)
    return f'<div class="hp-seat">{"".join(parts)}</div>'


def _middle(label: str, value: str, board: Sequence[Card]) -> str:
    return (f'<div class="hp-middle"><div class="hp-pot">{html.escape(label)} <b>{html.escape(value)}</b></div>'
            f"{_board(board)}</div>")


def ai_notice_html(history, ai_position: str) -> str:
    """The AI's latest recorded action; before it has acted, an invisible placeholder keeps the layout still."""
    latest = latest_ai_action(history, ai_position)
    if latest is None:
        return '<div class="hp-notice hp-notice-empty" aria-hidden="true"></div>'
    street, text = latest
    return (f'<div class="hp-notice" role="status"><span class="hp-notice-street">{html.escape(street)}</span>'
            f"{html.escape(text)}</div>")


def _other(position: str) -> str:
    return "big_blind" if position == "button" else "button"


# --- Tables ---------------------------------------------------------------------------------------


def table_during_hand(turn: HumanTurn, ai_name: str) -> str:
    """The table while the human is to act. The AI's cards are always backs here."""
    obs = turn.observation
    ai_position = _other(obs.position)
    ai = _seat(ai_name, ai_position,
               f"Stack <b>{obs.opponent_stack}</b> · Bet <b>{obs.opponent_bet}</b>", _cards(None), False)
    notice = ai_notice_html(obs.history, ai_position)
    middle = _middle(f"{STREET_NAMES[obs.street]} · Pot", str(obs.pot_size), obs.board_cards)
    you = _seat("You", obs.position, f"Stack <b>{obs.own_stack}</b> · Bet <b>{obs.own_bet}</b>",
                _cards(obs.hole_cards), True)
    return f'<div class="hp-table">{ai}{notice}{middle}{you}</div>'


def _hand_result_stats(starting_stack: int, net: int) -> str:
    return f"Stack <b>{starting_stack + net}</b> · This hand: <b>{signed(net)}</b> chips"


def table_after_hand(hand: CompletedHand, ai_name: str, starting_stack: int) -> str:
    """The finished hand. The AI's cards are faces only if the hand reached a showdown."""
    ai_position = _other(hand.human_position)
    ai = _seat(ai_name, ai_position, _hand_result_stats(starting_stack, hand.ai_net), _cards(hand.ai_hole_cards), False)
    notice = ai_notice_html(hand.history, ai_position)
    middle = _middle("Showdown" if hand.went_to_showdown else "Hand over", "", hand.board_cards)
    you = _seat("You", hand.human_position, _hand_result_stats(starting_stack, hand.human_net),
                _cards(hand.human_hole_cards), True)
    return f'<div class="hp-table">{ai}{notice}{middle}{you}</div>'
