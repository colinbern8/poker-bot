"""Opt-in acceptance test of the Streamlit screen with the real frozen model (never runs in CI).

    HYBRIDPOKER_REAL_MODEL=1 python -m pytest tests/acceptance -v -s

Nothing is substituted: the unmodified app loads the SHA-256-verified model,
warms the caches and plays a few rehearsal hands through real button
clicks. When opted in it fails (rather than skips) if the model is missing or
wrong. Startup time and AI decision times (median, p95, max) are printed with
``-s`` and compared with the targets; they are reported, never asserted.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from demo_support import latency_report, startup_report
from hybridpoker.agents.ev_agent import FROZEN_MODEL_SHA256
from hybridpoker.phase4_final import MANIFEST_PATH, load_manifest
from hybridpoker_demo.controller import HumanTurn

pytestmark = pytest.mark.skipif(os.environ.get("HYBRIDPOKER_REAL_MODEL") != "1",
                                reason="opt-in: set HYBRIDPOKER_REAL_MODEL=1 with the frozen model in place")

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
SEED = 2026
# More hands give a more reliable p95, e.g. HYBRIDPOKER_ACCEPTANCE_HANDS=20 (about 50+ AI decisions).
HANDS = int(os.environ.get("HYBRIDPOKER_ACCEPTANCE_HANDS", "4"))


def test_unmodified_app_plays_with_the_real_frozen_model(monkeypatch) -> None:
    # AppTest replaces sys.modules["__main__"]; restore it afterwards (see tests/test_demo_app.py).
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    st.cache_resource.clear()
    started = time.perf_counter()
    at = AppTest.from_file(str(APP), default_timeout=600).run()
    startup = time.perf_counter() - started
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]  # the model loaded and passed its hash check

    at.radio(key="seed_mode").set_value("Rehearsal seed")
    at.text_input(key="seed_text").input(str(SEED))
    at.button[[b.label for b in at.button].index("Start new match")].click().run()
    game = at.session_state["game"]
    assert game.match_seed == SEED
    assert game._ai.config() == load_manifest(MANIFEST_PATH)["hero_configs"]["ev_ml"]
    assert game._ai.response_model.model_sha256 == FROZEN_MODEL_SHA256

    for h in range(HANDS):
        while isinstance(game.view, HumanTurn):
            turn = game.view
            legal = turn.legal_categories
            choice = legal[(h + turn.ticket.decision_index) % len(legal)]
            at.button(key=f"act-{turn.hand_index}-{turn.ticket.decision_index}-{choice.value}").click().run()
            assert not at.exception, [e.value for e in at.exception]
            assert "Something went wrong" not in "".join(e.value for e in at.error)
        assert "Why did the AI play that way?" in [s.value for s in at.subheader]
        at.button(key="next-hand").click().run()
    assert game.scoreboard.hands_completed == HANDS

    print("\n" + startup_report(startup, "startup (model load + warm-up + first page)"))
    print(latency_report([d.total_seconds for d in game._ai.diagnostics], f"app, {HANDS} hands"))
