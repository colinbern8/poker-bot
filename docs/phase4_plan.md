# Phase 4 Plan: Hybrid Agent and Profitability Experiment (DRAFT)

Status: **draft for approval, 2026-11-12. Nothing in this document is
implemented yet.**
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
| **Probability** (existing baseline, unchanged) | Equity vs pot-odds thresholds (Phase 2) | none |
| **Hybrid-Prior** (control) | The new expected-value rule (section 4) | Dummy: training action frequencies, legal-masked |
| **Hybrid-ML** (treatment) | The same expected-value rule | Frozen Random Forest B |

Hybrid-Prior isolates the effect of the new decision rule. **Hybrid-ML
minus Hybrid-Prior isolates the value of the ML opponent model**, which is
the research question. Hybrid-ML minus Probability is the overall gain.

## 4. Hybrid decision rule (explainable, one step ahead)

The equity calculation is the Phase 1 one, unchanged: equity against one
random hand, 500 Monte Carlo samples. For each legal hero action the agent
estimates an expected chip gain, measured from the current point of the
hand:

- **Fold:** 0.
- **Check (no bet faced):** `equity × pot`.
- **Call:** `equity × (pot + to_call) − to_call`. This is the Phase 1
  call EV.
- **Bet or raise of size `b`:**
  1. The agent applies the action to a **copy** of the public game state.
  2. It asks the opponent model for the opponent's response probabilities
     in that hypothetical state: P(fold), P(call) and P(raise), where
     P(raise) covers half-pot, pot and all-in.
  3. It combines them:

     `EV = P(fold) × pot + P(call) × [equity × (pot + 2b) − b] + P(raise) × (−b)`

  The P(raise) term is a deliberate, documented simplification: the hero
  is assumed to give up against a re-raise.
- **Choice:** the action with the highest EV. Ties go to the more passive
  action.
- **Logging:** every decision records a reason, for example
  `bet half-pot: P(fold)=0.42, P(call)=0.51 -> EV +3.1 > check +2.4`.

Known simplifications, stated in the report:

- no betting on later streets;
- equity against a random hand, with no range narrowing from opponent
  actions;
- re-raises treated as losing the bet;
- five abstract bet sizes only.

These apply equally to Hybrid-Prior and Hybrid-ML, so their difference
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
- **Primary endpoint:** Hybrid-ML minus Probability in bb/100, averaged
  with equal weight over the 16 opponents. The 95% interval treats the
  deal pair as the unit and pairs hands across agents through the common
  deals.
- **Key secondary endpoint:** Hybrid-ML minus Hybrid-Prior, the value of
  the ML model.
- **Other secondary endpoints:**
  - training-pool vs held-out opponents;
  - per style;
  - decision diagnostics (how often the agents' actions differ).
- **Sample size:** set from the development-seed variance before the
  freeze. The starting budget is about 200 deal pairs × 16 opponents ×
  3 heroes ≈ 19,200 hands, roughly 1–2 h on 4 cores, to be confirmed by
  the development speed measurement.

## 7. Approvals needed before implementation

1. The three-agent design, with Hybrid-Prior as the control.
2. The one-step EV rule and its simplifications.
3. The behaviour-neutral changes to Phase 2 code (`observation_columns`
   refactor and the end-of-hand notification), protected by byte-identity
   tests.
4. Master seeds 5000 (development) and 4000 (final), the freeze
   procedure, and the primary endpoint.

## 8. Timeline

| Dates | Work |
|---|---|
| Nov 13 | Approve probability model and this plan |
| Nov 14–20 | Implement, with parity and unit tests; development runs; sample size |
| Nov 21 | Freeze |
| Nov 22–24 | Final evaluation runs |
| Nov 25–30 | Analysis and report |
| Dec 1–6 | Streamlit demo, slides, buffer |
