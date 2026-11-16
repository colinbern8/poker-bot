"""Illustrative decisions of the two EV agents on hand-picked spots (Stage 4B).

    python scripts/ev_decision_examples.py

Not a tournament and not evidence of strength: a few fixed spots, each
decided by the Frequency EV Agent and the ML EV Agent with the same
random seed (identical equity), plus the ML agent after it has watched a
few hands of two different opponents. Writes results/phase4/stage4b_examples.md.
Requires the frozen model file (kept outside the repository).
"""

from __future__ import annotations

import random
from pathlib import Path

from pokerkit import Card, Deck

from hybridpoker.actions import ActionCategory as C
from hybridpoker.actions import apply_action
from hybridpoker.agents import make_agent
from hybridpoker.agents.ev_agent import make_frequency_ev_agent, make_ml_ev_agent
from hybridpoker.game import SEAT_POSITIONS, STREET_NAMES, create_heads_up_state
from hybridpoker.match import run_match
from hybridpoker.observation import PublicAction, observe
from hybridpoker.simulator import public_hand_summary

OUT = Path("results/phase4/stage4b_examples.md")


def spot(holes: str, board: str, actions: list[C]):
    """Observation of the next actor after ``actions`` (seat 0 = big blind gets cards 1 and 3)."""
    state = create_heads_up_state()
    used = set(Card.parse(holes + board))
    burns = iter([c for c in Deck.STANDARD if c not in used])
    board_cards = iter(Card.parse(board))
    for card in Card.parse(holes):
        state.deal_hole(card)
    history: list[PublicAction] = []

    def deal():
        while True:
            if state.can_burn_card():
                state.burn_card(next(burns))
            elif state.can_deal_board():
                state.deal_board(next(board_cards))
            else:
                return

    for category in actions:
        deal()
        obs = observe(state, tuple(history))
        action = obs.legal(category)
        seat, street = state.actor_index, STREET_NAMES[state.street_index]
        apply_action(state, action)
        history.append(PublicAction(street, SEAT_POSITIONS[seat], category, action.bet_to, action.chips_added,
                                    action.is_all_in))
    deal()
    return observe(state, tuple(history))


# (title, holes [bb1 btn1 bb2 btn2], board, actions)
SPOTS = [
    ("Button, ace-king suited, preflop open", "2c" "As" "7d" "Ks", "", []),
    ("Button, seven-two offsuit, preflop open", "Ah" "7c" "Kd" "2s", "", []),
    ("Big blind, pair of queens, flop first to act", "Qs" "9h" "Qd" "8c", "Kc7h2d4s3c", [C.CHECK_CALL, C.CHECK_CALL]),
    ("Big blind, king-high flop with a weak hand, facing a half-pot bet", "6s" "Ah" "4d" "Kh",
     "Kc9h2d4s3c", [C.CHECK_CALL, C.CHECK_CALL, C.CHECK_CALL, C.RAISE_HALF_POT]),
]


def describe(agent, obs, seed: int) -> list[str]:
    agent.act(obs, random.Random(seed))
    d = agent.diagnostics[-1]
    lines = [f"- **{agent.name}** chose **{d.chosen}** (equity vs a random hand {d.equity:.3f})",
             "  - EVs: " + ", ".join(f"{k} {v:+.2f}" for k, v in d.action_evs.items())]
    for name, p in d.opponent_probabilities.items():
        cov = d.coverage[name]
        lines.append(f"  - if {name}: opponent fold {p[0]:.2f} / call {p[1]:.2f} / raise {sum(p[2:]):.2f} "
                     f"-> continues {d.continue_fraction[name]:.2f}, equity if called {d.called_equity[name]:.3f}; "
                     f"training coverage `{cov['key']}` = {cov['training_rows']} rows{' (LOW)' if cov['low'] else ''}")
    return lines


def main() -> None:
    md = ["# Stage 4B: example decisions (illustrative only)", "",
          "Hand-picked spots, each decided by both EV agents with the same seed (identical equity). "
          "Bets are valued with the hero's equity against the opponent's estimated continuing range "
          "(strongest P(call)+P(re-raise) fraction of its possible hands). "
          "The ML agent has seen no hands of this opponent here (history = training priors). "
          "Not a tournament; not evidence of playing strength.", ""]
    for title, holes, board, actions in SPOTS:
        obs = spot(holes, board, actions)
        md += [f"## {title}", "", f"Pot {obs.pot_size}, to call {obs.to_call}, board `{''.join(map(repr, obs.board_cards)) or '-'}`.", ""]
        for make in (make_frequency_ev_agent, make_ml_ev_agent):
            md += describe(make(), obs, seed=11)
        md.append("")

    # The same spot after the ML agent has watched 8 deals of two very different opponents.
    title, holes, board, actions = SPOTS[3]
    obs = spot(holes, board, actions)
    md += ["## History matters: the facing-a-bet spot after watching 8 deals of an opponent", "",
           "The ML agent first watches a short match between a solid bot and the named opponent "
           "(public end-of-hand summaries only), then decides the same spot. Only the opponent-history "
           "features differ.", ""]
    for opponent in ("nit", "maniac"):
        agent = make_ml_ev_agent()
        watched = run_match(make_agent("solid"), make_agent(opponent), deal_count=8, seed=99)
        for hand in watched.hands:
            agent.observe_hand_end(public_hand_summary(hand.result, hand.hero_seat, hand.deal_index, hand.seat_swap, 2))
        md.append(f"**After watching a {opponent}** ({agent.tracker.observed_rows()} of its actions):")
        md += describe(agent, obs, seed=11)
        md.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
