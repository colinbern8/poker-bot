# Phase 4 Plan: Hybrid Agent and Profitability Experiment

Status: **approved 2026-11-12, with refinements (section 8). Stages 4A
and 4B are implemented on `phase-4-ev-agents`, including the
continuing-range correction (section 11). A structural issue remains: the
one-step rule prefers all-in. A decision on it is needed before Stage 4C.**
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

**About the control baseline.** The Frequency EV Agent uses the *global*
training-split action shares (`results/phase3/frequency_prior.json`:
fold 0.043, check/call 0.707, half-pot 0.127, pot 0.113, all-in 0.009),
legal-masked in each hypothetical state. It does not condition on the
betting situation (street, bet size, raises so far) or on the opponent.
It is a deliberately simple control, and it may be weaker than a
context-aware frequency baseline. A gain of ML over this control
therefore includes the value of situation-awareness as well as of
opponent-specific history.

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

## 10. Stage 4B: implementation and finding (2026-11-12)

**Implemented:**

- `agents/ev_agent.py`: a single `EVAgent` with interchangeable opponent
  models:
  - `FrequencyResponseModel`, the control;
  - `MLResponseModel`, the frozen Random Forest B #22, loaded only after
    its SHA-256 is verified.

  Both agents share the same equity function (500 samples), the same EV
  equations, legal actions and tie-breaking (fewest additional chips,
  then class order).
- **Per-decision diagnostics:**
  - equity;
  - the EV of every legal action;
  - raw and masked opponent probabilities for every bet;
  - the hypothetical opponent observation;
  - training coverage of that state (diagnostic only);
  - the opponent's history length;
  - timing, with equity and model inference measured separately.
- **Committed artifacts**, built from the train split only by
  `scripts/build_phase4_artifacts.py`:
  - `results/phase3/frequency_prior.json`, which equals the trained
    Dummy's class prior;
  - `results/phase3/state_coverage.json`, with 26 coverage keys and a low
    threshold of 30 rows.

**Finding: the approved EV rule overvalues large bets, especially all-in.**

- *Cause.* In the "opponent calls" branch the hero's equity is still
  measured against a **random** hand, as if the opponent called with any
  hand (assumption: fixed equity). For a shove of `a` chips into a pot
  `P`, the called branch is worth about `e·(P + 2a) − a`. That is
  strongly positive whenever `e > 0.5`, however unlikely a strong
  opponent is to call with a weak hand.
- *Symptoms.*
  - In the smoke matches, both agents went all-in on 6 of 12 decisions,
    once with 59% preflop equity.
  - With queens on a K-7-2 flop (77% equity), all-in is valued at +102
    (Frequency) and +74 (ML), against +3 for checking.
  - After an all-in a re-raise is impossible, so the masked probability
    goes only to fold or call; the all-in therefore escapes the re-raise
    penalty.
- *Consequence.* Both agents tend toward "shove or check". The primary
  comparison would then depend largely on predicted call rates after an
  all-in, for which coverage is weak (`preflop|facing_all_in=1|raises=1`
  has 0 training rows; the flop equivalent has 55).

No equation was changed. A decision is needed before Stage 4C:

| Option | Change | Pros | Cons |
|---|---|---|---|
| **A. Equity against the calling range (recommended)** | In the "called" branch, value the hero's hand against the opponent's likely **continuing range**: the top share of hands, where the share is the predicted P(call) + P(re-raise). Ranking: preflop by the cached 169-class equity, postflop by made-hand strength on the board. | Removes the main flaw (callers are stronger than random). Explainable. The opponent model then informs both fold chances and caller strength, which strengthens the research question. Same rule for both agents. | About 2–4× equity time per decision. A new approximation (hands are ranked by current strength) that needs its own tests. |
| B. Restrict bet sizes (e.g. no all-in below a fixed equity) | Ad hoc rule | Quick | Arbitrary threshold; doesn't fix the underlying bias for other bet sizes |
| C. Keep as is and document | None | No extra work | The experiment would mostly measure an artifact of the rule |

## 11. Continuing-range correction (Option A, implemented 2026-11-12)

**Change.** In a bet's "called" branch, the hero's equity is now `e_c`, its
equity against the opponent's **estimated continuing range**: the strongest
`q = P(call) + P(re-raise)` fraction of the opponent's unblocked hands.
The fold and re-raise branches are unchanged; a re-raise still loses the
bet. Checks and calls of an opponent's bet still use equity against a
random hand, like the ProbabilityAgent.

**Both agents use the identical calculation.** Only `q` differs, because
it comes from their opponent models. Code: `range_equity.py` and
`ev_agent.called_equities`.

**How hands are ranked** (public information only; the board and the
hero's cards are blockers):

- *Preflop:* the cached 169-class equity, the same table the bots use.
- *Postflop:* made-hand strength on the actual board. Exact on the river.

**Is this justified?** The scripted opponents continue when their own
equity against a random hand clears a threshold, so a strength ranking
matches their rule closely:

- *Preflop:* the match is essentially exact.
- *Postflop:* the made-hand ranking agrees with the bots' own criterion
  with a rank correlation of 0.96 on a dry flop and 0.87 on a draw-heavy
  flop. It under-rates draws, which makes the estimated range slightly too
  strong, erring toward caution.

**Known approximations:**

- The strongest-hand percentile is not a reconstruction of the opponent's
  real range. The bots also bluff randomly.
- Earlier actions in the hand are ignored, so the opponent's range is
  assumed to start uniform. This leaves some residual optimism after the
  opponent has already called earlier streets.

**Estimator (fixed during the review).** The first version used separate
200-sample estimates per bet. Their noise (about ±0.03) was larger than
the correction for wide ranges, and sometimes gave an "equity if called"
*above* the random-hand equity. The final version uses common random
numbers:

- One shared run of 400 samples draws opponent hands from all hands and
  measures both the range equity and the all-hands equity from the same
  draws.
- Only their difference is added to the agent's usual equity. A full range
  gives exactly 0.
- Narrow ranges get top-up samples.

Tests confirm that the corrections for wide ranges are small,
non-positive and low-noise (standard deviation under 0.01 across seeds).

**Audit.** 300 fixed decision situations from the Phase 3 *validation*
split, each decided with the same seed by both agents under the old and
new rules (`results/phase4/stage4b_rule_audit.md`):

| Configuration | All-in | All-in with equity < 0.70 | Median equity of all-ins | Decisions changed |
|---|---|---|---|---|
| Frequency, old | 46.0% | 28.3% | 0.61 | |
| Frequency, new | 43.0% | 25.3% | 0.62 | 3.0% |
| ML, old | 47.3% | 29.7% | 0.61 | |
| **ML, new** | **32.7%** | **15.7%** | **0.71** | **24.7%** |

- **Runtime** (warm caches, new rule): median total 435 ms (ML) and
  523 ms (Frequency); 95th percentile 523 ms and 625 ms. Equity is about
  225 ms, the range correction 190–290 ms, the model 32 ms (ML) or 2 ms
  (Frequency). The old rule took about 230–260 ms in total.
- **Opponent adaptation:** for QQ on a K-7-2 flop, the ML agent's shove EV
  falls from +108 after watching a calling station to +10 after watching
  a nit.

**What the correction does not fix (a decision is needed before
Stage 4C).**

1. **The one-step rule makes all-in the best bet whenever the hero is
   ahead of the calling range.** The called branch is worth
   `e_c·P + (2e_c − 1)·a`, which grows with the bet size `a` whenever
   `e_c > 0.5`. Checking can never compete, because future betting is
   ignored. This comes from the approved "no future betting" assumption,
   not from the range approximation. All-in remains the chosen action in
   33% (ML) and 43% (Frequency) of the audited spots.
2. **All-in responses are poorly covered in training.** 41 of the ML
   agent's 98 all-ins lead to states with fewer than 30 training rows.
   Preflop all-in openings have none.
3. **The Frequency control hardly benefits.** Its global prior gives
   `q ≈ 0.94–0.96` in every state, so its estimated calling range is nearly
   random, and it changed only 3% of decisions. The primary comparison
   would therefore partly measure how differently the two agents handle
   all-ins.

Against the very loose bots (calling stations and maniacs call shoves
about 85% of the time in training), shoving strong hands may genuinely be
profitable, so the high all-in rate is not proven to be a mistake. Stage
4C's development matches on master seed 5000 can measure it per opponent.
