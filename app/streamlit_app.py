"""HybridPoker: heads-up No-Limit Hold'em against the frozen Hybrid ML Agent (Phase 5B).

Run from the repository folder:

    streamlit run app/streamlit_app.py

How this screen stays safe:

* **Shared vs per-session.** Only the read-only model (loaded once, SHA-256
  checked) and the warmed equity tables are shared (``st.cache_resource``).
  Each browser session has its own controller, and each match a new AI agent
  with its own opponent tracker.
* **One action per click.** Buttons only *queue* an action together with the
  ticket of the turn they answer. The queued action is then applied once,
  under the "AI is thinking" spinner. A repeated or late click carries an old
  ticket and is ignored by the controller, which also leaves its state
  unchanged if anything fails.
* **Hidden information.** During a hand the screen draws only the human's own
  view: the AI's cards are face-down, its reasoning is not shown, and no seed
  is displayed. The explanation appears after the hand; the AI's cards only
  at a showdown.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from hybridpoker.game import BIG_BLIND, SMALL_BLIND, STARTING_STACK
from hybridpoker.opponent_predictor import ModelHashMismatch
from hybridpoker_demo import presentation as ui
from hybridpoker_demo import resources
from hybridpoker_demo.controller import (
    CompletedHand,
    HeadsUpController,
    HumanTurn,
    IllegalHumanAction,
    StaleTicket,
    WrongPhase,
)
from hybridpoker_demo.seeds import check_seed

AI_NAME = "Hybrid ML Agent"
MODEL_NAME = "Random Forest B #22"

# --- Shared, read-only resources ---------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def shared_model():
    """The frozen model and warmed equity tables, loaded once per server process.

    Safe to share: the model is never changed by play. Errors are not cached,
    so restoring the model file and reloading the page retries.
    """
    started = time.perf_counter()
    model = resources.load_frozen_ml_model()
    loaded = time.perf_counter() - started
    warmed = resources.warm_caches()
    return model, {"load_seconds": loaded, "warm_seconds": warmed}


def load_model_or_stop():
    try:
        with st.spinner("Loading the frozen Hybrid ML model and warming up (first start only; up to a minute)…"):
            return shared_model()
    except resources.ModelUnavailable:
        st.error("**The frozen model file is missing, so the demonstration cannot start.** "
                 "No other agent is used in its place.")
        st.markdown(f"Expected file: `{resources.model_path()}`. Restore the verified copy, then reload this page:")
        st.code(resources.RESTORE_HINT, language="bash")
    except ModelHashMismatch:
        st.error("**The model file does not match the frozen SHA-256 hash, so it was not loaded.** "
                 "No other agent is used in its place.")
        st.markdown("Replace it with the verified copy, then reload this page:")
        st.code(resources.RESTORE_HINT, language="bash")
    except Exception as error:  # noqa: BLE001 - shown to the presenter, never swallowed
        st.error(f"**The model could not be loaded:** {type(error).__name__}: {error}")
    st.stop()


# --- Per-session state and queued actions -----------------------------------------------------------

state = st.session_state  # this browser session's state (read only inside a Streamlit run)


def queue(operation: tuple) -> None:
    """Button callback: remember what was clicked; it is applied once, below."""
    state.pending = operation


def queue_new_match() -> None:
    """Form callback: start a new match with a random deal or a rehearsal seed."""
    text = str(state.get("seed_text", "")).strip()
    state.seed_text = ""  # never leave a seed visible on screen during play
    if state.get("seed_mode") == "Rehearsal seed":
        try:
            seed = check_seed(int(text))
        except ValueError:
            state.notice = ("warning", "Enter the rehearsal seed as a whole number from 0 to 9223372036854775807.")
            return
        queue(("new", seed))
    else:
        queue(("new", None))


def apply(operation: tuple, model) -> None:
    kind = operation[0]
    game: HeadsUpController | None = state.game
    if kind == "new":
        state.game = HeadsUpController(resources.build_ml_agent(model), match_seed=operation[1])
    elif kind == "act":
        _, ticket, category = operation
        if game is None or ticket.match_id != game.match_id:
            raise StaleTicket("This click belongs to an earlier match.")
        game.submit(ticket, category)
    elif kind == "next":
        _, match_id, hand_index = operation
        if game is None or match_id != game.match_id or hand_index != game.hand_index:
            raise StaleTicket("This click belongs to an earlier hand.")
        game.next_hand()


def run_pending(model, area) -> None:
    """Apply the queued action once. The controller is unchanged if this fails."""
    if state.pending is None:
        return
    with area, st.spinner("The AI is thinking…"):
        operation, state.pending = state.pending, None  # taken before acting: a rerun cannot repeat it
        try:
            apply(operation, model)
        except (StaleTicket, WrongPhase):
            state.notice = ("info", "That click was already handled, so it was ignored.")
        except IllegalHumanAction as error:
            state.notice = ("warning", f"That action is not allowed now: {error}")
        except Exception as error:  # noqa: BLE001 - recover without losing the game
            state.notice = ("error", f"Something went wrong ({type(error).__name__}: {error}). The game was not "
                                     "changed: try the same action again, or start a new match.")


# --- Drawing -------------------------------------------------------------------------------------------


def sidebar(timings: dict) -> None:
    with st.sidebar:
        st.title("♠ HybridPoker")
        st.caption("Heads-up No-Limit Hold'em · blinds 1/2 · 200 chips each, every hand")
        with st.form("new_match", clear_on_submit=True, border=True):
            st.markdown("**New match**")
            st.radio("Deal", ["Random deal", "Rehearsal seed"], key="seed_mode", horizontal=True)
            st.text_input("Rehearsal seed (only for 'Rehearsal seed')", key="seed_text", placeholder="e.g. 2026")
            st.form_submit_button("Start new match", on_click=queue_new_match, width="stretch")
        st.markdown(f"**Opponent:** {AI_NAME}  \n{MODEL_NAME}, SHA-256 verified")
        st.caption(f"Model load {timings['load_seconds']:.1f} s · warm-up {timings['warm_seconds']:.1f} s")
        st.caption("Demonstration games are not research evidence: the evaluation used seat-swapped matches "
                   "against scripted bots (see results/phase4/).")


def show_notice() -> None:
    if state.notice:
        kind, text = state.notice
        {"info": st.info, "warning": st.warning, "error": st.error}[kind](text)
        state.notice = None


def scoreboard(game: HeadsUpController) -> None:
    board = game.scoreboard
    hand_number = game.hand_index + 1
    a, b, c = st.columns(3)
    a.metric("Hand", f"{hand_number}" if game.phase.value == "human_to_act" else f"{hand_number} (finished)")
    b.metric("Your total", f"{board.human_net_chips:+d} chips")
    c.metric("In big blinds", f"{board.human_net_bb:+g} bb")


def seat(title: str, cards_md: str, stack: int, bet: int | None, note: str = "") -> None:
    with st.container(border=True):
        left, middle, right = st.columns([2, 3, 2])
        left.markdown(f"**{title}**")
        middle.subheader(cards_md, anchor=False)
        lines = [f"Stack **{stack}**"] + ([f"Bet **{bet}**"] if bet is not None else []) + ([note] if note else [])
        right.markdown("  \n".join(lines))


def table_during_hand(turn: HumanTurn) -> None:
    obs = turn.observation
    ai_position = "big_blind" if obs.position == "button" else "button"
    seat(f"{AI_NAME} · {ui.POSITION_NAMES[ai_position]}", ui.HIDDEN_CARDS, obs.opponent_stack, obs.opponent_bet)
    with st.container(border=True):
        left, right = st.columns([5, 2])
        left.markdown(f"**{ui.STREET_NAMES[obs.street]}**")
        left.subheader(ui.cards_markdown(obs.board_cards), anchor=False)
        right.markdown("**Pot**")
        right.subheader(str(obs.pot_size), anchor=False)
    seat(f"You · {ui.POSITION_NAMES[obs.position]}", ui.cards_markdown(obs.hole_cards), obs.own_stack, obs.own_bet)


def actions(game: HeadsUpController, turn: HumanTurn) -> None:
    obs = turn.observation
    st.markdown(f"**Your move.** To call: **{obs.to_call}**")
    columns = st.columns(len(obs.legal_actions))
    for column, action in zip(columns, obs.legal_actions):
        label = ui.legal_action_label(action, obs.to_call, obs.street)
        key = f"act-{turn.hand_index}-{turn.ticket.decision_index}-{action.category.value}"
        column.button(label, key=key, on_click=queue, args=(("act", turn.ticket, action.category),),
                      width="stretch", type="primary" if action.category.value == "check_call" else
                      "secondary")


def action_log(history, human_position: str, expanded: bool) -> None:
    with st.expander("Actions this hand", expanded=expanded):
        lines = ui.history_lines(history, human_position)
        st.markdown("\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1)) if lines else "No actions yet.")


def table_after_hand(hand: CompletedHand) -> None:
    ai_position = "big_blind" if hand.human_position == "button" else "button"
    ai_cards = ui.cards_markdown(hand.ai_hole_cards) if hand.ai_hole_cards is not None else ui.HIDDEN_CARDS
    seat(f"{AI_NAME} · {ui.POSITION_NAMES[ai_position]}", ai_cards, STARTING_STACK + hand.ai_net, None,
         ui.chips_and_bb(hand.ai_net))
    with st.container(border=True):
        st.markdown("**Board**")
        st.subheader(ui.cards_markdown(hand.board_cards), anchor=False)
    seat(f"You · {ui.POSITION_NAMES[hand.human_position]}", ui.cards_markdown(hand.human_hole_cards),
         STARTING_STACK + hand.human_net, None, ui.chips_and_bb(hand.human_net))


def result_banner(game: HeadsUpController, hand: CompletedHand) -> None:
    result = ui.hand_result(hand)
    show = {"win": st.success, "loss": st.warning, "split": st.info}[result.outcome]
    show(f"**{result.headline}.** {result.detail}")
    st.button("Next hand ▶", key="next-hand", type="primary", on_click=queue,
              args=(("next", game.match_id, game.hand_index),))


def explanation(game: HeadsUpController, hand: CompletedHand) -> None:
    st.subheader("Why did the AI play that way?")
    st.caption("Shown only after the hand. Every number below is recorded from the AI's own decision "
               "(equity, expected values and the opponent model's predictions); nothing is estimated afterwards.")
    reports = [r for r in hand.ai_decisions if r.diagnostics is not None]
    if not reports:
        st.info("The AI did not have to make a decision in this hand.")
    for i, report in enumerate(reports):
        info = ui.explain_decision(report)
        with st.expander(f"{ui.STREET_NAMES[info.street]}: the AI chose **{info.action}**", expanded=i == len(reports) - 1):
            st.markdown(info.summary)
            for warning in info.warnings:
                st.warning(warning)
            st.markdown("**Expected value of each legal action** (chips, one-step estimate)")
            st.table(pd.DataFrame([{"Action": r.action, "Expected value": f"{r.ev:+.2f}",
                                    "Chosen": "✓" if r.chosen else ""} for r in info.ev_rows]), hide_index=True)
            if info.responses:
                st.markdown(f"**Model prediction of your response** ({MODEL_NAME}; masked to legal actions)")
                st.table(pd.DataFrame([{
                    "If the AI played": r.action, "You fold": f"{r.fold:.0%}", "You call": f"{r.call:.0%}",
                    "You raise": f"{r.raise_:.0%}", "AI equity if you continue": f"{r.equity_if_called:.0%}",
                    "Training examples": f"{r.training_rows}" + (" ⚠" if r.low_coverage else ""),
                } for r in info.responses]), hide_index=True)
            st.caption(f"Equity {info.equity:.0%} against a random hand · based on {info.history_hands} earlier "
                       f"hands of your play · decision time {info.seconds:.2f} s")


def profile(game: HeadsUpController) -> None:
    current = game.opponent_profile()
    if current is None:
        return
    st.subheader("What the AI has learned about you")
    st.caption(f"Observed = your actual actions in {len(game.completed_hands)} finished hands. AI's estimate = "
               f"the opponent model's input: your rates blended with the population average (weight "
               f"{current.alpha:g} decisions), so it moves slowly at first.")
    rows = ui.profile_rows(current, game.completed_hands)
    st.table(pd.DataFrame([{"Behaviour": r.behaviour, "Observed (you)": r.observed,
                            "AI's estimate (model input)": f"{r.model_input:.0%}",
                            "Population average": f"{r.population:.0%}"} for r in rows]), hide_index=True)


def welcome() -> None:
    st.header("Heads-up poker against the Hybrid ML Agent")
    st.markdown(
        "The AI combines **probability** (its chance of winning, from simulation) with a **machine-learning "
        "model** that predicts how *you* will respond to each bet, learning from your play as the match goes on. "
        "It picks the action with the highest expected value.\n\n"
        f"Each hand starts with {STARTING_STACK} chips per player, blinds {SMALL_BLIND}/{BIG_BLIND}, and the "
        "button alternates. After every hand you can see why the AI acted as it did.")
    st.button("Start match ▶", key="start-match", type="primary", on_click=queue, args=(("new", None),))


# --- Page ------------------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="HybridPoker", page_icon="♠", layout="wide")
    state.setdefault("game", None)
    state.setdefault("pending", None)
    state.setdefault("notice", None)

    model, timings = load_model_or_stop()
    sidebar(timings)
    status_area = st.container()
    run_pending(model, status_area)
    show_notice()

    game: HeadsUpController | None = state.game
    if game is None:
        welcome()
        return
    st.header(f"You vs the {AI_NAME}")
    scoreboard(game)
    view = game.view
    if isinstance(view, HumanTurn):
        table_during_hand(view)
        actions(game, view)
        action_log(view.observation.history, view.observation.position, expanded=False)
    else:
        result_banner(game, view)
        table_after_hand(view)
        action_log(view.history, view.human_position, expanded=True)
        explanation(game, view)
        profile(game)


# Streamlit runs this file as "__main__". Anything else that runs it - for example a
# multiprocessing worker re-running the parent's main file on Windows ("__mp_main__") -
# must not draw a page.
if __name__ == "__main__":
    main()
