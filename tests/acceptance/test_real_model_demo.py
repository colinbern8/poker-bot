"""Opt-in acceptance test with the real frozen model (never runs in CI).

The model file is kept outside git, so this test runs only when asked:

    HYBRIDPOKER_REAL_MODEL=1 python -m pytest tests/acceptance -v -s

When opted in it **fails** (rather than skips) if the model is missing, has
the wrong SHA-256, or the environment differs from the frozen manifest. It
checks, with the real agent at its real sample counts:

1. the live agent's configuration equals the evaluated ``ev_ml`` hero;
2. interrupted controller matches equal uninterrupted reference matches
   (cards, actions, payoffs, the AI's generator states, diagnostics apart
   from timings, and the opponent tracker after every hand);
3. startup time and AI decision latency: median, 95th percentile and max,
   compared with the targets (p95 < 2 s, startup < 90 s). Printed with
   ``-s``; targets are reported, never asserted.
"""

from __future__ import annotations

import os
import time

import pytest

from demo_support import (
    assert_same_diagnostics,
    assert_same_hand,
    human_categories,
    latency_report,
    play_controller_hand,
    reference_match,
    startup_report,
    tracker_state,
)
from hybridpoker.phase4_final import MANIFEST_PATH, check_environment, load_manifest
from hybridpoker_demo.controller import HeadsUpController
from hybridpoker_demo.resources import build_ml_agent, load_frozen_ml_model, warm_caches

pytestmark = pytest.mark.skipif(os.environ.get("HYBRIDPOKER_REAL_MODEL") != "1",
                                reason="opt-in: set HYBRIDPOKER_REAL_MODEL=1 with the frozen model in place")

SEEDS = (1201, 1202)
HANDS = 6


@pytest.fixture(scope="module")
def model():
    started = time.perf_counter()
    loaded = load_frozen_ml_model()  # raises if missing or the hash differs (checked first: fails fast)
    loading = time.perf_counter() - started
    warming = warm_caches()
    print(f"\nmodel load + SHA-256 check {loading:.1f} s; cache warm-up {warming:.1f} s")
    print(startup_report(loading + warming, "startup (load + check + warm-up)"))
    return loaded


def test_environment_and_configuration_match_the_frozen_evaluation(model) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    assert check_environment(manifest) == []
    assert build_ml_agent(model).config() == manifest["hero_configs"]["ev_ml"]


@pytest.mark.parametrize("seed", SEEDS)
def test_real_agent_controller_matches_uninterrupted_reference(model, seed) -> None:
    reference_ai = build_ml_agent(model)
    expected = reference_match(reference_ai, "solid", seed, HANDS)
    ai = build_ml_agent(model)
    controller = HeadsUpController(ai, match_seed=seed)
    for h, ref in enumerate(expected):
        if h:
            controller.next_hand()
        play_controller_hand(controller, human_categories(ref["result"], ref["setup"]))
        assert_same_hand(controller._results[h], ref["result"], ref["setup"].ai_seat)
        assert [r.rng_state_after for r in controller._records] == ref["rng_states"]
        assert tracker_state(ai) == ref["tracker"]
    assert_same_diagnostics(ai.diagnostics, reference_ai.diagnostics)

    print("\n" + latency_report([d.total_seconds for d in ai.diagnostics], f"seed {seed}"))
