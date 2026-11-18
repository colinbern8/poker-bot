"""Streamlit AppTest integration tests for the demonstration screen (app/streamlit_app.py).

**Test-only setup, isolated from production.** The ``app`` fixture uses
pytest's ``monkeypatch`` to replace three functions of
``hybridpoker_demo.resources`` for the duration of one test:

* ``load_frozen_ml_model`` returns the small ``tiny_bundle`` model (CI has no
  frozen model file);
* ``warm_caches`` does nothing (it only removes a delay);
* ``build_ml_agent`` builds the same ``EVAgent`` with few samples, so tests
  run quickly.

The app itself has no test hooks. The real model is exercised only by the
opt-in acceptance test in ``tests/acceptance/``.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from demo_support import SAMPLES, AllLowCoverageEV, ScriptedAggressorEV, ShoveEV
from hybridpoker.actions import RAISE_CATEGORIES, ActionCategory
from hybridpoker.agents.ev_agent import EVAgent, MLResponseModel
from hybridpoker.opponent_predictor import ModelHashMismatch, OpponentPredictor
from hybridpoker.simulator import shuffled_deck
from hybridpoker.state_coverage import COVERAGE_PATH, load_coverage
from hybridpoker_demo import presentation as ui
from hybridpoker_demo import resources
from hybridpoker_demo.controller import CompletedHand, HumanTurn
from hybridpoker_demo.seeds import hand_setup

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
TIMEOUT = 120
REHEARSAL_SEED = 918273645  # distinctive, so a leak of it into the page would be noticed


class FailOnceEV(EVAgent):
    """Raises once after deciding (to test recovery), then behaves normally."""

    fail = True

    def act(self, observation, rng):
        decision = super().act(observation, rng)
        if FailOnceEV.fail:
            FailOnceEV.fail = False
            raise RuntimeError("injected failure")
        return decision


@pytest.fixture
def app(monkeypatch, tiny_bundle):
    """Factory for AppTests that use the small test model (see the module docstring).

    AppTest replaces ``sys.modules["__main__"]`` with the app script and does
    not restore it. Later tests that start worker processes with "spawn" (the
    Windows default) would then re-run the app in each worker, so the
    original ``__main__`` is put back afterwards.
    """
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    model = MLResponseModel(OpponentPredictor(tiny_bundle))
    agents: list[EVAgent] = []

    def build(shared, root=None, cls=EVAgent):
        """Like ``build_ml_agent`` (same class and coverage, the shared model), with few samples."""
        agent = cls("ev_ml", shared, SAMPLES, coverage=load_coverage(COVERAGE_PATH), range_samples=SAMPLES)
        agents.append(agent)
        return agent

    st.cache_resource.clear()
    monkeypatch.setattr(resources, "load_frozen_ml_model", lambda root=None: model)
    monkeypatch.setattr(resources, "warm_caches", lambda: 0.0)
    monkeypatch.setattr(resources, "build_ml_agent", build)

    def make() -> AppTest:
        return AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()

    make.model, make.agents, make.monkeypatch, make.build = model, agents, monkeypatch, build
    yield make
    st.cache_resource.clear()


# --- helpers -----------------------------------------------------------------------------------------


def game(at: AppTest):
    return at.session_state["game"]


def texts(at: AppTest) -> list[str]:
    """Every piece of text the page shows (main area and sidebar)."""
    out: list[str] = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "text", "info", "success", "warning",
                 "error", "code"):
        out += [str(e.value) for e in getattr(at, kind)]
    out += [f"{m.label} {m.value}" for m in at.metric]
    out += [b.label for b in at.button] + [e.label for e in at.expander]
    out += [e.proto.body for e in at.get("html")]  # the poker table (Phase 5D) is one HTML block
    for table in at.table:
        out += [str(v) for v in table.value.astype(str).to_numpy().ravel().tolist()] + list(table.value.columns)
    return out


def page(at: AppTest) -> str:
    return "\n".join(texts(at))


def table_html(at: AppTest) -> str:
    """The rendered poker table (the HTML block containing the table)."""
    tables = [e.proto.body for e in at.get("html") if 'class="hp-table"' in e.proto.body]
    assert len(tables) == 1, len(tables)
    return tables[0]


def action_buttons(at: AppTest) -> dict[str, str]:
    return {b.key: b.label for b in at.button if b.key and b.key.startswith("act-")}


def click_action(at: AppTest, category: ActionCategory) -> None:
    turn = game(at).view
    at.button(key=f"act-{turn.hand_index}-{turn.ticket.decision_index}-{category.value}").click().run()
    assert not at.exception, [e.value for e in at.exception]


def choose(turn: HumanTurn, policy: int) -> ActionCategory:
    legal = turn.legal_categories
    return legal[policy % len(legal)] if policy >= 0 else (
        ActionCategory.CHECK_CALL if ActionCategory.CHECK_CALL in legal else legal[0])


def play_hand(at: AppTest, policy: int = -1) -> CompletedHand:
    while isinstance(game(at).view, HumanTurn):
        click_action(at, choose(game(at).view, policy))
    return game(at).view


def start(at: AppTest, seed: int | None = None) -> None:
    """Start a match: the welcome button, or the sidebar form (random deal or rehearsal seed)."""
    if seed is None and "start-match" in [b.key for b in at.button]:
        at.button(key="start-match").click().run()
    else:
        at.radio(key="seed_mode").set_value("Random deal" if seed is None else "Rehearsal seed")
        if seed is not None:
            at.text_input(key="seed_text").input(str(seed))
        at.button[[b.label for b in at.button].index("Start new match")].click().run()
    assert not at.exception, [e.value for e in at.exception]


def incident(caplog, reference: str) -> logging.LogRecord:
    """The developer-log record written for an on-screen reference code."""
    records = [r for r in caplog.records if r.name == "hybridpoker.demo" and reference in r.getMessage()]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    return records[0]


def ai_hole_cards(match_seed: int, hand_index: int):
    """The AI's cards of a hand, computed from the seed exactly as the simulator deals them."""
    setup = hand_setup(match_seed, hand_index)
    deck = shuffled_deck(setup.deal_seed)
    return deck[setup.ai_seat], deck[setup.ai_seat + 2]


EXPLANATION_BLOCK = re.compile(r"\*\*(Preflop|Flop|Turn|River) — AI ")


def explanation_blocks(at: AppTest) -> list[str]:
    """The short default explanation of each AI decision, in order."""
    return [m.value for m in at.markdown if EXPLANATION_BLOCK.match(m.value)]


def default_block(report) -> str:
    """The markdown the page should show for one decision (see ``explanation`` in the app)."""
    brief = ui.default_explanation(report)
    lines = [f"**{ui.STREET_NAMES[report.street]} — {brief.headline}**",
             " ".join(t for t in (brief.reason, brief.response) if t)]
    lines += [brief.comparison] if brief.comparison else []
    lines += [f"⚠ *{brief.warning}*"] if brief.warning else []
    return "  \n".join(lines)


# --- tests ---------------------------------------------------------------------------------------------


def test_startup_shows_the_welcome_page_and_model_status(app) -> None:
    at = app()
    assert not at.exception
    assert "start-match" in [b.key for b in at.button]
    assert game(at) is None
    assert "SHA-256 verified" in page(at)


def test_play_complete_hands_with_score_updates_and_hand_transitions(app) -> None:
    at = app()
    start(at)
    total = 0
    for h in range(4):
        assert game(at).hand_index == h
        hand = play_hand(at, policy=h)
        total += hand.human_net
        result = ui.hand_result(hand)
        assert any(result.headline in t for t in texts(at))
        text = page(at)
        assert (f"Hand {h + 1} (finished) · Session profit: **{ui.signed(total)} chips** "
                f"({ui.signed(total / 2)} bb)") in text
        assert f"This hand: <b>{ui.signed(hand.human_net)}</b> chips" in table_html(at)
        assert "Your total" not in text
        assert "bb/100" not in text  # no bb/100 on the screen
        assert hand.human_position == ("button" if h % 2 == 0 else "big_blind")  # button alternates
        at.button(key="next-hand").click().run()
    assert game(at).scoreboard.hands_completed == 4


def test_action_buttons_are_exactly_the_legal_actions_with_engine_amounts(app) -> None:
    at = app()
    start(at)
    seen_call = False
    for h in range(3):
        while isinstance(game(at).view, HumanTurn):
            turn = game(at).view
            obs = turn.observation
            buttons = action_buttons(at)
            expected = {f"act-{turn.hand_index}-{turn.ticket.decision_index}-{a.category.value}":
                        ui.legal_action_label(a, obs.to_call, obs.street) for a in obs.legal_actions}
            assert buttons == expected
            if obs.to_call:
                seen_call = True
                call = obs.legal(ActionCategory.CHECK_CALL)
                assert (f"Call all-in ({obs.to_call})" if call.is_all_in else f"Call {obs.to_call}") in buttons.values()
            text = page(at)
            table = table_html(at)
            assert f"Stack <b>{obs.own_stack}</b> · Bet <b>{obs.own_bet}</b>" in table
            assert f"Stack <b>{obs.opponent_stack}</b> · Bet <b>{obs.opponent_bet}</b>" in table
            assert f"Pot <b>{obs.pot_size}</b>" in table
            assert table.count('class="hp-card hp-back"') == 2  # the AI's two face-down cards
            assert table.count('hp-slot') + len(obs.board_cards) == 5  # five community-card positions
            click_action(at, choose(turn, h + 1))
        at.button(key="next-hand").click().run()
    assert seen_call


def test_facing_an_ai_preflop_shove_shows_fold_and_call_all_in(app) -> None:
    """Regression (manual Phase 5D test): after the AI shoves preflop from the big blind, the
    screen offers exactly two buttons, "Fold" and "Call all-in (198)", and the table shows
    the AI all-in with its cards face-down until the showdown."""
    app.monkeypatch.setattr(resources, "build_ml_agent", lambda model, root=None: app.build(model, cls=ShoveEV))
    at = app()
    start(at)
    turn = game(at).view
    assert turn.observation.position == "button" and turn.observation.street == "preflop"
    click_action(at, ActionCategory.CHECK_CALL)  # the human completes the small blind (Call 1)

    turn = game(at).view
    assert isinstance(turn, HumanTurn) and turn.observation.opponent_stack == 0
    prefix = f"act-{turn.hand_index}-{turn.ticket.decision_index}-"
    assert action_buttons(at) == {f"{prefix}fold": "Fold", f"{prefix}check_call": "Call all-in (198)"}
    table = table_html(at)
    assert "Stack <b>0</b> · Bet <b>200</b>" in table      # the AI, all-in
    assert ">AI goes all-in for 200 chips</div>" in table   # announced at the table
    assert "Stack <b>198</b> · Bet <b>2</b>" in table      # the human
    assert table.count('class="hp-card hp-back"') == 2      # the AI's cards stay hidden

    click_action(at, ActionCategory.CHECK_CALL)
    hand = game(at).view
    assert isinstance(hand, CompletedHand) and hand.went_to_showdown and len(hand.board_cards) == 5
    assert table_html(at).count('class="hp-card hp-back"') == 0  # both hands shown at the showdown
    text = page(at)
    assert "Preflop · AI (Big blind): All-in (200)" in text
    assert "Preflop · You (Button): Call all-in (198)" in text


def test_every_ai_action_is_announced_at_the_table_from_the_recorded_history(app) -> None:
    """The announcement on screen is always the AI's latest action in the recorded history, with the
    seat names "You" and "Hybrid AI" and the positions as small tags."""
    at = app()
    start(at, REHEARSAL_SEED)
    seen = set()
    for h in range(4):
        while True:
            view = game(at).view
            history = view.observation.history if isinstance(view, HumanTurn) else view.history
            mine = view.observation.position if isinstance(view, HumanTurn) else view.human_position
            ai_position = "big_blind" if mine == "button" else "button"
            table = table_html(at)
            assert '<div class="hp-name">You<span class="hp-pos">' in table
            assert '<div class="hp-name">Hybrid AI<span class="hp-pos">' in table
            latest = ui.latest_ai_action(history, ai_position)
            if latest is None:
                assert "hp-notice-empty" in table
            else:
                assert f'<span class="hp-notice-street">{latest[0]}</span>{latest[1]}</div>' in table
                seen.add(latest[1].split()[1])  # checks, calls, bets, raises, goes, folds
                ai_lines = [line for line in ui.history_lines(history, mine) if " · AI (" in line]
                assert ai_lines and ai_lines[-1].startswith(latest[0])  # the same action ends the history
            if not isinstance(view, HumanTurn):
                break
            click_action(at, choose(view, h + len(history)))
        at.button(key="next-hand").click().run()
    assert len(seen) >= 2, seen


def test_hidden_cards_seeds_and_diagnostics_never_shown_during_a_hand(app) -> None:
    at = app()
    start(at, REHEARSAL_SEED)
    assert at.text_input(key="seed_text").value == ""  # the form cleared the typed seed
    showdowns = folds = 0
    for h in range(8):
        ai_cards = [ui.card_text(c) for c in ai_hole_cards(REHEARSAL_SEED, h)]
        while isinstance(game(at).view, HumanTurn):
            text = page(at)
            assert not any(card in text for card in ai_cards)
            assert str(REHEARSAL_SEED) not in text and str(game(at).match_seed) not in text
            assert table_html(at).count('class="hp-card hp-back"') == 2  # AI cards face-down on the table
            assert "Why did the AI play that way?" not in text and not at.table  # no diagnostics yet
            assert "equity" not in text.lower()
            click_action(at, choose(game(at).view, h))
        hand = game(at).view
        text = page(at)
        assert str(REHEARSAL_SEED) not in text
        backs = table_html(at).count('class="hp-card hp-back"')
        if hand.went_to_showdown:
            showdowns += 1
            assert all(card in text for card in ai_cards) and backs == 0
        else:
            folds += 1
            assert not any(card in text for card in ai_cards) and backs == 2
        at.button(key="next-hand").click().run()
    assert showdowns and folds


def test_explanations_show_the_recorded_values(app) -> None:
    at = app()
    start(at, 77)
    for _ in range(6):
        hand = play_hand(at, policy=2)
        reports = [r for r in hand.ai_decisions if r.diagnostics is not None]
        if reports:
            break
        at.button(key="next-hand").click().run()
    text = page(at)
    tables = [t.value for t in at.table]
    assert explanation_blocks(at) == [default_block(r) for r in reports]  # the short default explanations
    labels = [e.label for e in at.expander if e.label.startswith("Detailed analysis")]
    assert len(labels) == len(reports) and "Technical details" not in text
    for report in reports:
        d = report.diagnostics
        info = ui.explain_decision(report)
        assert info.summary in text  # the full summary, now inside "Detailed analysis"
        ev_table = next(t for t in tables if list(t.columns) == ["Action", "Expected value", "Chosen"]
                        and list(t["Action"]) == [r.action for r in info.ev_rows])
        assert list(ev_table["Expected value"]) == [f"{ev:+.2f}" for ev in d.action_evs.values()]
        assert list(ev_table["Chosen"]).count("✓") == 1
        if d.opponent_probabilities:
            responses = next(t for t in tables if "You fold" in t.columns
                             and list(t["If the AI played"]) == [r.action for r in info.responses])
            assert list(responses["You fold"]) == [f"{p[0]:.0%}" for p in d.opponent_probabilities.values()]
    profile_table = next(t for t in tables if "AI's estimate (model input)" in t.columns)
    profile = game(at).opponent_profile()
    assert list(profile_table["AI's estimate (model input)"])[0] == f"{profile.rates['b_vpip']:.0%}"
    assert "Observed (you)" in profile_table.columns and "Population average" in profile_table.columns


def test_default_warning_concerns_only_the_action_the_ai_chose(app) -> None:
    """Regression (Phase 5E): a check, call or fold is not flagged because an unchosen bet has
    low training coverage. Every bet option is marked low here; only chosen bets get the flag,
    and the warning about the other options stays under "Detailed analysis"."""
    app.monkeypatch.setattr(resources, "build_ml_agent",
                            lambda model, root=None: app.build(model, cls=AllLowCoverageEV))
    at = app()
    start(at, 77)
    passive = 0
    for _ in range(8):
        hand = play_hand(at)  # the human checks or calls, so the AI meets spots where it may too
        reports = [r for r in hand.ai_decisions if r.diagnostics is not None]
        blocks = explanation_blocks(at)
        assert len(blocks) == len(reports)
        for report, block in zip(reports, blocks):
            chose_bet = report.category in RAISE_CATEGORIES
            assert ("⚠" in block) == chose_bet, block
            if not chose_bet and report.diagnostics.coverage:
                passive += 1
                assert ui.explain_decision(report).warnings  # the old screen flagged on any warning
        if any(not (r.category in RAISE_CATEGORIES) and r.diagnostics.coverage for r in reports):
            details = [w.value for w in at.warning]
            assert any(w.startswith("Predictions for these other options also rest on fewer than") for w in details)
        if passive >= 3:
            break
        at.button(key="next-hand").click().run()
    assert passive >= 3


def test_fold_button_is_shown_exactly_when_the_human_owes_chips(app) -> None:
    """Regression (Phase 5E Fold investigation): post-flop, facing a bet, a raise or an all-in the
    screen shows a Fold button; when checking is free it shows Check and no Fold button."""
    from test_demo_fold_availability import human_choice, spot

    app.monkeypatch.setattr(resources, "build_ml_agent",
                            lambda model, root=None: app.build(model, cls=ScriptedAggressorEV))
    at = app()
    start(at)
    seen = set()
    for hand_index in range(2):
        while isinstance(game(at).view, HumanTurn):
            turn = game(at).view
            obs = turn.observation
            kind = spot(obs)
            prefix = f"act-{turn.hand_index}-{turn.ticket.decision_index}-"
            buttons = action_buttons(at)
            if kind == "free":
                assert f"{prefix}fold" not in buttons and buttons[f"{prefix}check_call"] == "Check"
            else:
                assert buttons[f"{prefix}fold"] == "Fold"
                assert list(buttons)[0] == f"{prefix}fold"  # first in the row, as always
            if kind == "all-in":
                assert set(buttons) == {f"{prefix}fold", f"{prefix}check_call"}
            seen.add((kind, obs.street != "preflop"))
            click_action(at, human_choice(obs, hand_index))
        at.button(key="next-hand").click().run()
    assert {("bet", True), ("raise", True), ("all-in", True), ("free", True)} <= seen, seen


def test_duplicate_and_stale_clicks_do_not_act_twice(app) -> None:
    at = app()
    start(at)
    current = game(at)
    turn = current.view
    category = choose(turn, -1)
    click_action(at, category)
    snapshot = (current.hand_index, current._human_actions, len(current._records), len(app.agents[-1].diagnostics),
                current.view)
    # A second click on the same (now old) button: its callback queues the old ticket again.
    at.session_state["pending"] = ("act", turn.ticket, category)
    at.run()
    assert "already handled" in page(at)
    assert (current.hand_index, current._human_actions, len(current._records), len(app.agents[-1].diagnostics),
            current.view) == snapshot
    # A late "Next hand" for an earlier hand is ignored too.
    play_hand(at)
    at.button(key="next-hand").click().run()
    hand_index = game(at).hand_index
    at.session_state["pending"] = ("next", current.match_id, hand_index - 1)
    at.run()
    assert game(at).hand_index == hand_index and "already handled" in page(at)
    # An illegal action (folding when checking is free) is refused with a message.
    while isinstance(game(at).view, HumanTurn) and game(at).view.observation.to_call:
        click_action(at, choose(game(at).view, -1))
    if isinstance(game(at).view, HumanTurn):
        before = game(at)._human_actions
        at.session_state["pending"] = ("act", game(at).view.ticket, ActionCategory.FOLD)
        at.run()
        assert "not allowed" in page(at) and game(at)._human_actions == before
        assert "ActionCategory" not in page(at) and "legal:" not in page(at)  # no internal detail on screen


def test_new_match_resets_and_rehearsal_seeds_reproduce(app) -> None:
    first, second = app(), app()
    start(first, 2026)
    start(second, 2026)
    for _ in range(3):
        a, b = play_hand(first, policy=1), play_hand(second, policy=1)
        assert (a.human_hole_cards, a.board_cards, a.human_net) == (b.human_hole_cards, b.board_cards, b.human_net)
        first.button(key="next-hand").click().run()
        second.button(key="next-hand").click().run()
    assert game(first)._results == game(second)._results
    old = game(first)
    start(first)  # a new random match from the sidebar
    new = game(first)
    assert new is not old and new.match_id != old.match_id
    assert new.scoreboard.hands_completed == 0 and new.hand_index == 0
    assert "Session profit: **+0 chips** (+0 bb)" in page(first)
    assert app.agents[-1] is new._ai and app.agents[-1].tracker.observed_rows() == 0  # fresh AI and tracker


def test_sessions_are_isolated_and_only_the_model_is_shared(app) -> None:
    one, two = app(), app()
    start(one)
    start(two)
    g1, g2 = game(one), game(two)
    assert g1 is not g2 and g1._ai is not g2._ai and g1._ai.tracker is not g2._ai.tracker
    assert g1._ai.response_model is app.model and g2._ai.response_model is app.model  # one shared, read-only model
    play_hand(one)
    assert g2.scoreboard.hands_completed == 0 and g2._ai.tracker.observed_rows() == 0


def test_unexpected_error_is_reported_safely_and_the_game_recovers(app, caplog) -> None:
    """The screen shows a plain message and a reference code; the log has the full error."""
    caplog.set_level(logging.INFO, logger="hybridpoker.demo")
    app.monkeypatch.setattr(resources, "build_ml_agent", lambda model, root=None: app.build(model, cls=FailOnceEV))
    FailOnceEV.fail = True
    at = app()
    start(at)
    failed = False
    for _ in range(40):
        current = game(at)
        if not isinstance(current.view, HumanTurn):
            at.button(key="next-hand").click().run()
            continue
        before = (current.view, current._human_actions, len(current._ai.diagnostics))
        click_action(at, choose(current.view, -1))
        if "Something went wrong" in page(at):
            failed = True
            text = page(at)
            assert "injected failure" not in text and "RuntimeError" not in text and "Traceback" not in text
            reference = re.search(r"reference ([0-9A-F]{6})", text).group(1)
            record = incident(caplog, reference)
            assert record.levelno == logging.ERROR and "injected failure" in record.getMessage()
            assert record.exc_info is not None  # the traceback is logged, down to the line that failed
            assert 'raise RuntimeError("injected failure")' in caplog.text
            assert (current.view, current._human_actions, len(current._ai.diagnostics)) == before
            click_action(at, choose(current.view, -1))  # the same button works again
            assert "Something went wrong" not in page(at)
            break
    assert failed


def test_spawned_worker_processes_do_not_run_the_page(app) -> None:
    """Regression (Windows CI): with "spawn", workers re-run the parent's __main__ file.

    Right after an AppTest, ``__main__`` is the app script. A spawned worker must
    still start: the page code runs only when ``__name__ == "__main__"``.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    app()
    assert sys.modules["__main__"].__file__.endswith("streamlit_app.py")
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert pool.submit(abs, -7).result(timeout=120) == 7


def test_unexpected_model_loading_error_shows_no_internal_details(app, caplog) -> None:
    caplog.set_level(logging.INFO, logger="hybridpoker.demo")

    def fail(root=None):
        raise RuntimeError("internal detail at /secret/path")

    app.monkeypatch.setattr(resources, "load_frozen_ml_model", fail)
    at = app()
    assert not at.exception
    text = page(at)
    assert any("could not be loaded" in e.value for e in at.error)
    assert "internal detail" not in text and "/secret/path" not in text and "RuntimeError" not in text
    reference = re.search(r"reference \*\*([0-9A-F]{6})\*\*", text).group(1)
    record = incident(caplog, reference)
    assert "internal detail at /secret/path" in record.getMessage() and record.exc_info is not None
    assert len(at.button) == 0 and at.session_state["game"] is None and not app.agents  # no fallback


@pytest.mark.parametrize("error, message", [
    (resources.ModelUnavailable("missing"), "frozen model file is missing"),
    (ModelHashMismatch("bad hash"), "does not match the frozen SHA-256"),
])
def test_missing_or_invalid_model_stops_with_instructions_and_no_fallback(app, error, message) -> None:
    def fail(root=None):
        raise error

    app.monkeypatch.setattr(resources, "load_frozen_ml_model", fail)
    at = app()
    assert not at.exception
    assert any(message in e.value for e in at.error)
    assert any("restore_phase4_model.py" in c.value for c in at.code)
    assert len(at.button) == 0           # no way to start a game
    assert at.session_state["game"] is None
    assert not app.agents                # no agent of any kind was created (no silent fallback)
