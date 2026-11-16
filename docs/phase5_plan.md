# Phase 5: Playable Heads-Up Demonstration

Goal: a reliable, explainable human-versus-AI heads-up game for the
December 7–9, 2026 presentation, built **without changing** the frozen
Phase 4 model, decision logic or evaluation results.

| Stage | Content | Status |
|---|---|---|
| 5A | Game engine: interrupt-and-replay adapter, heads-up controller, tests | merged (PR #9) |
| **5B** | Streamlit screen, explanations, AppTest tests, presentation checklist | this PR |

## Decisions (approved by the project owner)

- Human versus the **Hybrid ML Agent** (the frozen `ev_ml` hero) only.
- The **five abstract actions** (fold, check/call, half-pot, pot, all-in);
  only legal ones are accepted.
- **Independent hands:** both players start every hand with **200 chips**,
  blinds **1/2**. A running score is kept separately.
  - Carried-over stacks are not possible: the agent rebuilds hypothetical
    states from a fresh 200-chip hand (`counterfactual.replay_public_state`).
- The **button alternates** every hand (the human is on the button in hands
  0, 2, 4, ...).
- **Detailed AI explanations only after the hand.** The AI's hole cards are
  shown only at a showdown; mucked cards stay hidden.
- Runs **locally** with the verified frozen model (SHA-256 checked).
- **Random match seeds** for normal play; **fixed seeds** for rehearsals.

## Why the game is "interrupt and replay"

The frozen simulator `play_hand` plays a whole hand in one call and asks each
agent for a decision when it is that agent's turn. A person cannot answer
inside that call, and the frozen code must not change. So:

1. The human's seat is a **scripted agent**
   (`hybridpoker_demo.human.ScriptedHumanAgent`). It plays back the actions the
   person has already chosen. When it runs out, it raises
   `HumanInputRequired` carrying the human's own observation. `play_hand` does
   not catch it, so the hand simply stops.
2. When the person chooses an action, the hand is **replayed from the start**
   with one more scripted action.
3. The AI is wrapped in a **recording agent**
   (`hybridpoker_demo.recording.RecordingAgent`):
   - a decision it already made is returned from the record (not
     recomputed), and its random generator is set back to exactly the state it
     had after that decision;
   - a new decision is computed by the real agent and recorded.

`play_hand` builds a fresh generator from the same seed on every replay, and
all of the AI's randomness comes from that generator. A replayed hand is
therefore **identical** to a hand played straight through. The tests prove it
by comparing with the unchanged simulator, including the generator state after
every AI decision.

A replay costs about 1 ms (measured: median 0.8 ms, p95 2.3 ms per human
action, AI thinking excluded); the AI's own thinking dominates.

## Architecture (`src/hybridpoker_demo/`)

| Module | Role |
|---|---|
| `human.py` | `ScriptedHumanAgent`, `HumanInputRequired` |
| `recording.py` | `RecordingAgent`, `RecordedDecision`, `ReplayDivergence` |
| `seeds.py` | match seeds (random or fixed), per-hand deck and agent seeds, seating |
| `resources.py` | repository root, frozen model loading with SHA-256 check, `build_ml_agent`, cache warm-up |
| `controller.py` | `HeadsUpController`: the match, input checks, all-or-nothing updates, post-hand reports |

The package does not import Streamlit. It sits outside `src/hybridpoker/`,
because the frozen manifest reports any new module inside the frozen package.

### Controller guarantees

- **Tickets.** Every human turn has a ticket (match id, hand, decision
  index). An input with a stale, duplicate or foreign ticket is rejected, so a
  double click cannot act twice.
- **Legal input only.** An unknown or illegal category (e.g. folding when
  checking is free) is rejected.
- **All-or-nothing updates.** Each operation computes the new state on the
  side and commits it only at the end. On any failure (an AI error, a tracker
  error) the AI's diagnostics list is truncated back, its opponent tracker is
  restored, and the controller's state is unchanged, so the input can be
  retried.
- **Opponent tracking exactly once.** The AI is told about a finished hand
  once, in the same step that finishes the hand. The controller checks that
  the tracker recorded one row per human action of that hand. The tests verify
  this relationship independently.
- **Information.** During a hand the human sees only their own observation.
  After the hand, the AI's decisions are reported with their diagnostics:
  equity, EV of every action, predicted responses, the opponent-model
  coverage flags and timings.

## Tests (Phase 5A)

| File | What it shows |
|---|---|
| `tests/test_demo_replay.py` | Go/no-go gate. Interrupted hands equal uninterrupted ones for three AI types (ML with a small model, situation-frequency EV, probability agent): cards, actions, payoffs, AI decisions, generator states, diagnostics, opponent tracking. Negative control without `setstate`; tampered and incomplete replays are rejected |
| `tests/test_demo_controller.py` | Controller matches equal uninterrupted reference matches, including tracker and diagnostics; one notification per hand; tracker rows equal human actions counted two independent ways; rules (200 chips, blinds, alternating button, score); seeds |
| `tests/test_demo_state_safety.py` | Injected AI and tracker failures leave a full snapshot unchanged and retries reproduce the reference; stale, duplicate, foreign, illegal and wrong-phase inputs are rejected without change |
| `tests/test_demo_leakage.py` | No AI cards in the human's view or in non-showdown results; explanations only after the hand; the AI never sees the human's cards |
| `tests/test_demo_resources.py` | Missing or wrong model is refused; the live agent's configuration equals the evaluated `ev_ml` hero |
| `tests/test_phase4_freeze_guard.py` | The frozen `check_manifest` passes (46 files, manifest hash, model hash); `results/phase4/` is unchanged |
| `tests/acceptance/test_real_model_demo.py` | **Opt-in**, with the real frozen model: environment and configuration, replay equivalence and latency |

## Stage 5B: the Streamlit screen

Run with `streamlit run app/streamlit_app.py` and open http://localhost:8501.
Presentation steps: [`phase5_demo_checklist.md`](phase5_demo_checklist.md).

| Part | File | Role |
|---|---|---|
| Screen | `app/streamlit_app.py` | Table, action buttons, result, explanation, opponent profile, sidebar (new match, rehearsal seed, model status) |
| Display logic | `src/hybridpoker_demo/presentation.py` | Plain functions: card text, action labels with chip amounts, action log, results, explanations built only from recorded diagnostics, observed-vs-model profile |
| Controller additions | `controller.py` | Read-only `opponent_profile()`; each `AIDecisionReport` carries the AI's `to_call` and legal actions, so explanation labels use the engine's exact amounts |
| Settings | `.streamlit/config.toml` | Localhost only, no first-run prompt, no usage statistics (offline) |

**Shared versus per-session state.**
- Only the read-only model and the warmed equity tables are cached for the
  whole server (`st.cache_resource`).
- Each browser session has its own controller in `st.session_state`.
- Each match gets a new agent, and with it a new opponent tracker.

**One action per click.**
- Buttons only queue an action together with its ticket.
- The script applies the queued action once, under the "AI is thinking"
  spinner. It takes the action out of the queue before acting, so a rerun
  cannot repeat it.
- A repeated or late click carries an old ticket and is ignored with a short
  notice.
- Failures leave the game unchanged (5A guarantees) and show a recovery
  message.

**Hidden information on screen.**
- During a hand the screen draws only the human's view: face-down AI cards,
  no reasoning, no seed.
- The typed rehearsal seed is cleared from its box when the match starts.
- After the hand:
  - the explanation appears;
  - the AI's cards appear only at a showdown.

**Explanations from recorded values only.** Each AI decision shows:
- its action and equity against a random hand;
- the expected value of every legal action;
- the model's predicted fold / call / raise response to each bet size, with
  the AI's equity if the human continues;
- training-coverage warnings;
- a summary sentence assembled from those same numbers.

The opponent profile keeps three columns apart:
- the human's **observed** actions;
- the **model's inputs** (rates smoothed toward the population);
- the **population** averages.

**Not shown on the main screen:** bb/100, which is unreliable over a few
hands.

### Tests added in 5B

| File | What it shows |
|---|---|
| `tests/test_demo_presentation.py` | Labels and amounts; explanation values equal the recorded diagnostics for every AI decision; the opponent profile is read-only, and its model inputs equal the smoothing formula applied to the observed counts |
| `tests/test_demo_app.py` | AppTest, with the small model substituted by a test-only fixture. Covers: startup, complete hands, legal buttons with engine amounts, score and transitions, rehearsal-seed reproducibility, new match, ignored duplicate, stale and illegal clicks, no hidden cards, seeds or diagnostics during a hand, explanation values on screen, error recovery, per-session isolation with one shared model, and missing or wrong model with no fallback |
| `tests/acceptance/test_real_model_app.py` | **Opt-in**: the unmodified app with the real model, played through button clicks |

## Known limitations and risks

- **The real-model acceptance test has not been run in this environment.** The
  frozen model file is not in the repository, and rebuilding it would mean
  retraining. It must be run once on the presentation machine:
  ```bash
  HYBRIDPOKER_REAL_MODEL=1 python -m pytest tests/acceptance -v -s
  ```
- **Warm-up.** Filling the preflop equity tables took 39 s in the
  development container. In Phase 4 it took up to 17 s per worker. The
  screen must do it once at start-up.
- **Decision time.** In the Phase 4 evaluation it was 0.65 s median and
  1.4 s at the 95th percentile, in a Codespace. It is not yet measured on
  the presentation machine.
- **Not research evidence.** The demonstration differs from the
  evaluation:
  - one hand per deal instead of seat-swapped pairs;
  - a human opponent, who is unlike the training bots.

  Results of demo games say nothing about the research question.
- **The agent shoves often** (23% of decisions in the evaluation). Audiences
  should expect many all-ins.
- **Refreshing the browser** starts a new match: state lives in the browser
  session. This is acceptable for a demo.
- **AppTest does not run a real browser.** In the development container the
  screen was also checked in Chromium (screenshots), using the small model;
  not with the real one.
- **The real-model acceptance tests have not been run**, so the presentation
  machine must run them (checklist step 3).
