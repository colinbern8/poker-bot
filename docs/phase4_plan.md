# Phase 4 Plan: Hybrid Agent and Profitability Experiment

Status: **approved 2026-11-12, with refinements (section 8). Stage 4A
(decision equations, opponent tracking, simulator integration)
implemented; awaiting review before Stage 4B.**
Competition presentation: **December 7–9, 2026**.

## 1. Research question

> Can supervised machine-learning-based opponent modeling improve
> decisions made by a probability-based poker agent?

Phase 3 showed that the opponent model predicts actions better than a
baseline. Phase 4 tests whether **using** those predictions makes an agent
win more chips against the same opponents.

## 2. The opponent model (frozen input from Phase 3)

| Item | Value |
|---|---|
| Model | Random Forest, Feature Set B, grid #22: 300 trees, `max_depth=12`, `min_samples_leaf=20`, `max_features=0.5`, no class weighting |
| Selection | Lowest validation log-loss of legal-masked probabilities (decided after inspecting Stage 3; see `docs/phase3_plan.md` section 13a) |
| Probabilities | Always legal-masked and renormalized, in the fixed order fold, check_call, raise_half_pot, raise_pot, all_in |
| Calibration | Top-label ECE 0.014 on test and held-out (`results/phase3/probability_model_report.md`) |
| Speed | About 15 ms per batched prediction call |

The model file is frozen by SHA-256 hash in the evaluation manifest.

## 3. Agents compared (three heroes)

| Agent | Decision rule | Opponent model |
|---|---|---|
| **ProbabilityAgent** (existing baseline, unchanged) | Equity vs pot-odds thresholds (Phase 2) | none |
| **Frequency EV Agent** (control) | The new expected-value rule (section 4) | Training action frequencies, legal-masked (the Dummy) |
| **ML EV Agent** (treatment) | The same expected-value rule | Frozen Random Forest B |

**Primary comparison: ML EV Agent minus Frequency EV Agent.** It isolates
the benefit of learned opponent modeling, because the decision rule is
identical.

**Key secondary comparison: ML EV Agent minus ProbabilityAgent.** It shows
the overall improvement over the mathematical baseline.

Both comparisons use paired deals and are reported in bb/100 with
confidence intervals.

## 4. Hybrid decision rule (explainable, one step ahead)

The equations are implemented in `src/hybridpoker/ev.py` and unit-tested
on hand-computed spots in `tests/test_ev.py`. The tests cover:

- checking;
- calling, including a short-stack call;
- betting with the opponent always folding, always calling, always
  re-raising, and a mix;
- a preflop raise;
- facing a raise, with sunk chips excluded;
- all-in, with uncalled chips returned.

Every EV is the expected change in the hero's chips compared with folding
now.

The equity calculation is the Phase 1 one, unchanged: equity against one
random hand, 500 Monte Carlo samples. For each legal hero action the agent
estimates an expected chip gain, measured from the current point of the
hand:

- **Fold:** 0.
- **Check (no bet faced):** `equity × pot`.
- **Call:** `equity × (pot + to_call) − to_call`. This is the Phase 1
  call EV.
- **Bet or raise adding `a` chips (the opponent must add `r` to call):**
  1. The agent builds the opponent's hypothetical state after the action
     (`counterfactual.py`).
  2. The opponent model gives legal-masked response probabilities:
     P(fold), P(call) and P(raise), where P(raise) covers half-pot, pot
     and all-in.
  3. It combines them:

     `EV = P(fold) × pot + P(call) × [equity × (pot + a' + r') − a'] − P(raise) × a`

     Here `r' = min(r, opponent stack)` and `a' = a − (r − r')`, because
     uncalled chips are returned.
- **Facing a raise:** the same fold, call and re-raise equations apply.
  Chips the hero already put in are sunk.
- **All-in:** if the hero's bet puts either player all-in, the
  opponent's legal mask only allows fold or call, so P(raise) = 0. A
  short call returns the uncalled excess.
- **Choice:** the action with the highest EV. Ties go to the more passive
  action.
- **Logging:** every decision records a reason, for example
  `bet half-pot: P(fold)=0.42, P(call)=0.51 -> EV +3.1 > check +2.4`.

Known simplifications, stated in the report:

- **No future betting:** after a check, call or called bet, the hand is
  valued at showdown with no more chips.
- **Re-raises lose the bet:** the hero gives up the chips just added.
  This is conservative.
- **Fixed equity:** equity is against a random hand in every branch,
  with no range narrowing.
- **Five abstract bet sizes only.**
- **Counterfactual distribution shift.** The model was trained on states
  that occurred, but the agents query states created by their own
  candidate actions. Some of these are rare in training. For example,
  only 11 of 23,603 training rows (all from maniac and solid players, all
  calls) show a player facing a preflop all-in. For a preflop shove the
  model therefore predicts about the training-wide facing-all-in call
  rate (0.84) for any opponent. This is an experimental limitation.
  Stage 4B will log how often decisions depend on such poorly covered
  states.

These apply equally to the Frequency EV and ML EV Agents, so their difference
measures only the value of the opponent model.

## 5. Implementation (files)

| File | Change | Phase 2/3 behaviour |
|---|---|---|
| `hand_log.py` | Extract the existing `obs_` column construction into a pure `observation_columns(obs)` function, used by the logger and by the agent | Logs must stay byte-identical (existing test plus a new one) |
| `agents/base.py`, `simulator.py` | Add an optional end-of-hand notification, `Agent.observe_hand_end(public_summary)`, which defaults to doing nothing. It carries public actions, board, deal index and seating only: **no hole cards, no showdown cards** | Existing agents are unaffected; Phase 2 logs byte-identical |
| `opponent_tracker.py` (new) | Match-level memory of the opponent's public actions per completed deal. It builds Feature Set B with the **same functions** as training (`history_features`) and excludes the current deal and its mirror hand | Train/serve parity test: on logged matches, live features equal offline features |
| `opponent_predictor.py` (new) | Loads the frozen model bundle, checks its hash, returns legal-masked probabilities | |
| `agents/hybrid.py` (new) | HybridAgent with a pluggable predictor (ML or prior) | Unit tests on hand-computed EV spots |
| `phase4_manifest.py`, `scripts/run_phase4_eval.py` (new) | Seeds, opponents, pairing, freeze record, parallel runner | |

## 6. Experimental protocol

**Development (any change allowed):**

- Development seeds come from **master seed 5000**: `derive_seed(5000, "dev", i)`.
- They are used for debugging, sanity matches, measuring speed, and
  estimating variance for the sample size.

**Freeze (before any final match is played):**

- Tag the commit `phase4-freeze`.
- Write `data/manifests/phase4_eval_manifest.json` with:
  - the code commit;
  - the model file's SHA-256 hash;
  - all agent parameters;
  - the opponents, seeds and deal counts;
  - the primary analysis below.
- After the freeze, no agent, model or analysis changes are allowed.

**Final evaluation (fresh, never-used seeds):**

- **Seeds:** from **master seed 4000**: `derive_seed(4000, "eval", opponent, i)`. A
  test confirms they do not overlap any Phase 2, Phase 3, pilot or
  development seed.
- **Opponents:** all 16 configurations, reported separately for the 12
  training-pool configurations and the 4 held-out ones.
- **Matches:** each hero plays seat-swapped matches against each opponent
  with the **same deal seeds** (common random numbers), so the agents
  face identical cards.
- **Primary endpoint:** ML EV Agent minus Frequency EV Agent in bb/100,
  averaged with equal weight over the 16 opponents. This is the value of
  learned opponent modeling. The 95% interval treats the deal pair as the
  unit and pairs hands across agents through the common deals.
- **Key secondary endpoint:** ML EV Agent minus ProbabilityAgent, the
  overall improvement over the mathematical baseline.
- **Other secondary endpoints:**
  - training-pool vs held-out opponents;
  - per style;
  - decision diagnostics (how often the agents' actions differ).
- **Sample size:** set from the development-seed variance before the
  freeze. The starting budget is about 200 deal pairs × 16 opponents ×
  3 heroes ≈ 19,200 hands, roughly 1–2 h on 4 cores, to be confirmed by
  the development speed measurement.

## 7. Performance measurement (Stage 4C)

- Model inference time is measured separately from equity calculation
  time.
- Median and 95th-percentile decision latency are reported for each
  agent.
- No major optimizations unless they are needed for the competition demo.

## 8. Approved refinements (2026-11-12)

1. **Primary comparison:** ML EV minus Frequency EV. ML EV minus
   ProbabilityAgent is the key secondary comparison. Both use paired deals
   and bb/100 with confidence intervals.
2. **EV equations:** specified and unit-tested before the agents; the
   assumptions are documented (section 4).
3. **Counterfactual-state tests:** the hypothetical state must
   - contain only information the acting player can observe;
   - update pot, bets, stacks, position and history correctly;
   - have the opponent as the predicted actor;
   - respect the legal-action mask;
   - preserve the approved history calculation.

   Distribution shift is flagged as a limitation.
4. **Simulator changes:** the shared observation function and the public
   end-of-hand notification, with regression tests showing no change to
   gameplay, legality or logs.
5. **Final evaluation:**
   - development on master seed 5000;
   - final evaluation on fresh master seed 4000;
   - sample size set from development data before the freeze;
   - code and model hashes frozen before the final runs;
   - no tuning on final results.
6. **Performance:** see section 7.
7. **Stages, each reviewed before the next:**
   - 4A: equations, tracking and integration;
   - 4B: both EV agents with controlled tests;
   - 4C: development matches, performance and final evaluation plan;
   - 4D: freeze and final runs.

## 9. Timeline

| Dates | Work |
|---|---|
| Nov 12 | Stage 4A |
| Nov 13–16 | Stage 4B: EV agents |
| Nov 17–20 | Stage 4C: development runs, performance, sample size |
| Nov 21 | Freeze (Stage 4D) |
| Nov 22–24 | Final evaluation runs |
| Nov 25–30 | Analysis and report |
| Dec 1–6 | Streamlit demo, slides, buffer |
