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

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from demo_support import SAMPLES
from hybridpoker.actions import ActionCategory
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
    """Factory for AppTests that use the small test model (see the module docstring)."""
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
    for table in at.table:
        out += [str(v) for v in table.value.astype(str).to_numpy().ravel().tolist()] + list(table.value.columns)
    return out


def page(at: AppTest) -> str:
    return "\n".join(texts(at))


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


def ai_hole_cards(match_seed: int, hand_index: int):
    """The AI's cards of a hand, computed from the seed exactly as the simulator deals them."""
    setup = hand_setup(match_seed, hand_index)
    deck = shuffled_deck(setup.deal_seed)
    return deck[setup.ai_seat], deck[setup.ai_seat + 2]


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
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Your total"] == f"{total:+d} chips"
        assert metrics["In big blinds"] == f"{total / 2:+g} bb"
        assert "Hand" in metrics and not any("bb/100" in t for t in texts(at))  # no bb/100 on the screen
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
                assert any(label.startswith(f"Call {obs.to_call}") for label in buttons.values())
            text = page(at)
            assert f"Stack **{obs.own_stack}**" in text and f"Stack **{obs.opponent_stack}**" in text
            assert f"Bet **{obs.own_bet}**" in text and f"Bet **{obs.opponent_bet}**" in text
            assert str(obs.pot_size) in [h.value for h in at.subheader]  # the pot
            click_action(at, choose(turn, h + 1))
        at.button(key="next-hand").click().run()
    assert seen_call


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
            assert "Why did the AI play that way?" not in text and not at.table  # no diagnostics yet
            assert "equity" not in text.lower()
            click_action(at, choose(game(at).view, h))
        hand = game(at).view
        text = page(at)
        assert str(REHEARSAL_SEED) not in text
        if hand.went_to_showdown:
            showdowns += 1
            assert all(card in text for card in ai_cards)
        else:
            folds += 1
            assert not any(card in text for card in ai_cards)
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
    for report in reports:
        d = report.diagnostics
        info = ui.explain_decision(report)
        assert info.summary in text
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
    assert {m.label: m.value for m in first.metric}["Your total"] == "+0 chips"
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


def test_unexpected_error_is_reported_and_the_game_recovers(app) -> None:
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
            assert (current.view, current._human_actions, len(current._ai.diagnostics)) == before
            click_action(at, choose(current.view, -1))  # the same button works again
            assert "Something went wrong" not in page(at)
            break
    assert failed


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
