# Phase 4 Plan: Hybrid Agent and Profitability Experiment

Status: **approved 2026-11-12, with refinements (section 8). Stages 4A
and 4B are implemented on `phase-4-ev-agents`, including the
continuing-range correction (section 11). Stage 4C added a situation-aware
frequency control (section 12) and ran the development evaluation on master
seed 5000 (section 13). The project owner approved the pre-freeze decisions
on 2026-11-13 (section 14). Frozen at tag `phase4-freeze` (`d50af2b`); the
final evaluation ran on 2026-11-13 and met the primary success criterion
(section 15). Phase 4 is complete pending review of the pull requests.**
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

## 3. Agents compared (four heroes since Stage 4C)

| Agent | Decision rule | Opponent model |
|---|---|---|
| **ProbabilityAgent** (existing baseline, unchanged) | Equity vs pot-odds thresholds (Phase 2) | none |
| **Frequency EV Agent** (control) | The new expected-value rule (section 4) | Training action frequencies, legal-masked (the Dummy) |
| **Situation-Aware Frequency EV Agent** (stronger control, added in Stage 4C; section 12) | The same expected-value rule | Training action frequencies per betting situation (street, facing a bet, facing an all-in, raises this street), smoothed, legal-masked |
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
opponent-specific history. Stage 4C therefore added the
Situation-Aware Frequency EV Agent (section 12) as a stronger control; the
global agent is kept as a reference.

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

## 12. Situation-aware frequency control (Stage 4C, approved 2026-11-12)

**Model.** Training-split action frequencies conditioned on the public
betting situation the opponent faces:

- the street;
- whether it faces a bet;
- whether that bet is all-in;
- raises this street, capped at 3.

Code: `situation_frequency.py`. Artifact:
`results/phase3/situation_frequency_prior.json` (27 situations, 23,603
train rows). The agent is the same `EVAgent` with the same equity, EV
equations, continuing-range correction and tie-break; only the opponent
model differs. It has no opponent history.

**Sparse situations.** Hierarchical shrinkage toward the parent situation,
`p = (counts + 30 × p_parent) / (rows + 30)`, ordered global → street →
facing a bet → facing an all-in → raises. An unseen situation inherits its
parent. The order and `K = 30` (the existing low-coverage threshold) were
fixed before any development match. No test, held-out or Phase 4 result
was used.

**What it does not control for.** It is stronger than the global baseline
but does not control for every current-hand feature the ML model uses:

- bet and pot sizes beyond "a bet exists";
- position;
- the board;
- earlier streets;
- the opponent's history and identity.

ML minus this control therefore still combines the value of opponent
history with the value of those current-hand features.

## 13. Stage 4C development evaluation (master seed 5000, 2026-11-13)

Full report: `results/phase4/stage4c_dev_report.md` (with
`stage4c_dev_summary.json`).

**Setup**

- 4 heroes × 16 opponents.
- One seat-swapped match of 100 deals (200 hands) per pair: 12,800 hands
  in total.
- Shared deals per opponent (`derive_seed(5000, "dev", i)`).
- Code commit `43d819d` (clean tree); model hash verified.
- 52.9 min on 4 processes.
- 95% intervals use the deal pair as the unit. Groups weight opponents
  equally.

These are development results. They guide decisions and sample size and
are not the experiment's findings.

### 13.1 Results

**Win rate, bb/100 (equal-weight over 16 opponents):**

| Hero | All 16 | Calling station | Maniac | Nit | Solid |
|---|---|---|---|---|---|
| ProbabilityAgent | +383 [+316, +450] | +286 | +1003 | −14 | +257 |
| Global Frequency EV | +782 [+671, +894] | +1928 | +1420 | −51 | −168 |
| Situation-Aware Frequency EV | +770 [+659, +880] | +1953 | +1355 | −67 | −162 |
| ML EV | +831 [+726, +935] | +1891 | +1299 | +3 | +129 |

**Paired differences, bb/100 (same deals):**

| Comparison | All 16 | Held-out 4 | Nit | Solid |
|---|---|---|---|---|
| ML − Situation-Aware | +61 [−11, +133] | +109 [−27, +246] | +70 [−27, +166] | **+292 [+109, +474]** |
| ML − Global | +48 [−24, +121] | +82 [−58, +221] | +54 [−47, +154] | **+297 [+99, +495]** |
| ML − ProbabilityAgent | **+448 [+335, +560]** | **+564 [+334, +794]** | +17 [−88, +121] | −128 [−362, +106] |
| Situation-Aware − Global | −13 [−57, +32] | −27 [−115, +61] | −16 [−59, +27] | +6 [−72, +83] |

Against calling stations and maniacs, ML minus either frequency control is
slightly negative and not significant (−62 and −56 against Situation-Aware).

### 13.2 Decision-policy findings

1. **All-in dominates the EV agents' play.**
   - All-in is chosen in 30.8% (Global), 28.5% (Situation-Aware) and
     22.4% (ML) of hero decisions; preflop it is 41%, 41% and 30%.
   - Hands with a shove carry all of the EV agents' profit: +826 to +914
     bb/100. All other hands lose 48–83 bb/100.
   - The ProbabilityAgent never shoves. 2.6% of its decisions are all-in
     calls.
2. **The shove-heavy policy is exploited by selective opponents.**
   - *Against solid opponents* both frequency EV agents lose to the
     ProbabilityAgent by about 420 bb/100 (interval excludes 0). They lose
     about 1,100–1,300 bb in shove hands that reach showdown.
   - *Against nits* a shove wins only the blinds when the nit folds (about
     1 bb, 97.5% of the time). It loses about 67 bb when called. The net is
     slightly negative for every EV agent.
   - Against calling stations and maniacs, shoving value hands is very
     profitable (+1,460 to +1,975 bb/100 from shove hands) because they
     call wide.
3. **The ML agent adapts to the opponent; the frequency controls cannot.**
   - ML shoves in 12.8% of decisions against nits, against 28–31% for the
     controls (preflop: 14.5% against 42%).
   - Against solid opponents ML shoves in 18.6% of decisions, and its shove
     hands win +2,414 bb, against −667 (Situation-Aware).
   - Against calling stations ML keeps shoving (31.8%).
   - This adaptation explains the significant gains against solid
     opponents. It is consistent with the research hypothesis, but it is
     one style group out of four in a development run.
4. **The situation-aware control plays almost like the global one**
   (−13 [−57, +32] bb/100), although its predictions differ (for example a
   0.19 fold rate facing a flop bet, against about 0.04). Most of its
   pivotal decisions are all-ins, whose responses are fold or call in
   sparse situations, so both controls predict similar call rates.
5. **Low-coverage reliance.**
   - The ML agent's chosen action led to a poorly covered opponent state
     (fewer than 30 training rows) in 15.1% of its decisions: 45% of its
     bets and **67% of its all-ins**. The controls are at 20–21% of their
     decisions.
   - The ML agent's most important decisions therefore depend on model
     extrapolation in rare states. Its predictions there come mostly from
     the opponent-history features. This must be stated as a validity
     limitation.
6. **Engineering checks:**
   - 0 errors or illegal actions in 64 matches;
   - identical cards for every hero against each opponent;
   - fresh-process replays reproduce the matches exactly.

### 13.3 Latency and sample size

**Latency (warm caches):**

| Hero | Median | 95th percentile | Median split |
|---|---|---|---|
| ProbabilityAgent | 215 ms | 286 ms | |
| Global Frequency EV | 414 ms | 591 ms | equity 210, model 2, range 251 ms |
| Situation-Aware Frequency EV | 434 ms | 579 ms | equity 210, model 2, range 257 ms |
| ML EV | 348 ms | 595 ms | equity 206, model 33, range 197 ms |

All are fast enough for an interactive demo.

**Sample size.**

- **Variance.** The per-pair standard deviation of ML minus either
  frequency control is about 1,470 bb/100. For ML minus the
  ProbabilityAgent it is about 2,300.
- **Runtime.** All four heroes against 16 opponents take about 0.85 h per
  100 deal pairs on 4 processes.

| Deal pairs per match | Hands | Runtime (4 cores) | ±95% for ML − Situation-Aware | Smallest detectable effect (80% power) |
|---|---|---|---|---|
| 200 (original plan) | 25,600 | ≈ 1.7 h | ±51 | ≈ 73 |
| 400 | 51,200 | ≈ 3.4 h | ±36 | ≈ 51 |
| **500** | 64,000 | ≈ 4.3 h | ±32 | ≈ 46 |
| 600 | 76,800 | ≈ 5.1 h | ±29 | ≈ 42 |

These figures assume the development standard deviations hold. Longer
matches also give the ML agent more opponent history, which may change its
results compared with these 100-deal matches.

### 13.4 Recommendations (for review before the freeze)

1. **Primary endpoint:** make **ML minus Situation-Aware Frequency EV**
   the primary comparison, because it is the stronger control. Keep ML
   minus Global and ML minus ProbabilityAgent as pre-specified secondary
   comparisons. This must be decided before the freeze. It is not chosen
   from the development results: both controls gave similar estimates.
2. **All-in policy:** the shove-heavy play is a real weakness. It is a
   consequence of the approved one-step rule (no future betting), not a
   code error, and the same rule applies to all three EV agents, so the
   comparison stays fair. Options:
   - **A (recommended for the deadline): freeze the rule unchanged.**
     Report the all-in rates, the per-style results and the loss against
     selective opponents as a primary limitation. Present the ML agent's
     reduced shoving against nits and solid opponents as the mechanism of
     its gains.
   - B: before the freeze, add an approximate future-betting value for
     checks and calls to the shared equations. This changes the method. It
     needs about a day of work and tests plus a new development run, and
     its own parameters would have to be fixed without tuning.
   - Restricting all-ins directly (for example an equity floor) is
     **not** recommended: it is ad hoc, and it was explicitly excluded.
3. **Sample size:** 500 deal pairs per match (≈ 4.3 h on 4 cores,
   ±32 bb/100 for the primary comparison). Use 400 if runtime is tight.
   200 pairs (±51) is likely too few to detect an effect of the size seen
   here.
4. **Low coverage:** keep the low-coverage diagnostics in the final
   report. Add the "ML − control, decisions with and without
   low-coverage choices" breakdown as a pre-specified secondary analysis.

No agent, rule, model or sample size has been frozen. Stage 4D has not
started.

## 14. Pre-freeze decisions and freeze candidate (2026-11-13)

**Decisions approved by the project owner:**

1. Keep the one-step EV decision policy for the competition. Excessive
   all-in play and the lack of future-betting reasoning are documented as
   major limitations.
2. **Primary endpoint:** ML EV minus Situation-Aware Frequency EV.
3. **Secondary comparisons:** ML EV minus ProbabilityAgent, and ML EV minus
   Global Frequency EV.
4. **500 paired deals per match**, subject to the runtime check below.
5. Fresh final master seed **4000**, never used in development.
6. Freeze code, model hash, opponent configurations, evaluation design and
   analysis method before any final run.

**Freeze candidate:**

- `data/manifests/phase4_eval_manifest.json`, built by
  `scripts/build_phase4_manifest.py`;
- `docs/phase4_analysis_plan.md`;
- `src/hybridpoker/phase4_final.py`;
- `scripts/run_phase4_final.py` (`--check`, `--run`, `--report`; resumable).

The development report is now produced by the shared module
`hybridpoker/phase4_report.py`. Regenerating it reproduced the committed
Stage 4C report byte for byte.

**Verification before the freeze**

| Check | Result |
|---|---|
| Branch and HEAD | `phase-4-ev-agents` at `99bb8f3`, clean, equal to origin |
| CI for `99bb8f3` | Run 35 passed on Ubuntu and Windows |
| Local tests at `99bb8f3` | 424 passed (Python 3.12.3) |
| Model SHA-256 | Matches `5d4c8235…7b31` |
| Shared deals and seat swaps | Every hero has deals 0–99 × both seatings with identical cards against each opponent (independent re-check of the development records). Existing tests cover seat-swapped card sharing. A new test checks that identical heroes on shared deals give a paired difference of exactly 0. |
| Interval calculation | Independent recomputation reproduces the reported primary, +60.8 [−10.9, +132.6]. Stratified bootstrap: [−8.5, +135.2]. Batch-means SE 37.2 against i.i.d. 36.6. Mean lag-1 autocorrelation +0.03. The deal pair is a valid independent unit. |
| Tracker cost with long matches | Opponent-history features take about 40 ms per deal, with no growth from 100 to 500 deals |
| Workload | 64 matches of 500 deals. Projected from the development per-match times, scheduled longest-first: about **4.3 h on 4 cores** (16.9 CPU-hours; 2.1 h on 8 cores). Longest match about 23 min. Records about 7 MB compressed; 15 GB RAM and 30 GB disk available. |

## 15. Final evaluation results (Stage 4D, 2026-11-13)

Full report: `results/phase4/stage4d_final_report.md`. Verification:
`results/phase4/stage4d_verification.md`. Dependence sensitivity:
`results/phase4/stage4d_dependence_sensitivity.md`. Detailed summary and
limitations: `docs/progress.md` (Stage 4D).

| Endpoint | bb/100, 95% CI |
|---|---|
| **Primary: ML EV − Situation-Aware Frequency EV** | **+84.8 [+51.6, +117.9]: criterion met** |
| ML EV − Global Frequency EV | +102.1 [+69.6, +134.6] |
| ML EV − ProbabilityAgent | +469.4 [+419.0, +519.9] |

- **Answer to the research question.** Within this experiment, yes.
  Supervised opponent modeling improved the decisions of the shared
  one-step EV agent against a frequency control that knows the betting
  situation:
  - by 84.8 bb/100 averaged over 16 rule-based opponents;
  - by 96.6 bb/100 on the 4 held-out ones (exploratory).
- **Robustness.** The result holds under:
  - the pre-registered bootstrap;
  - three methods allowing for dependence between deals within a match;
  - removal of any single opponent or style (post hoc).
- **Scope.** It does not show that the EV rule is a good poker strategy in
  general:
  - the rule shoves very often;
  - it ignores future betting;
  - it was tested only against bots.
