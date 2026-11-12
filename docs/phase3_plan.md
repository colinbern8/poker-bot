# Phase 3 Plan: Supervised Opponent Modeling

Status: **design and implementation plan approved 2026-11-12. Stage 1 (pilot) implemented; awaiting pilot review.**
Branch: `phase-3-machine-learning`.
Competition presentation: **December 7–9, 2026**.

This document is the single source of truth for Phase 3. Future sessions
should read it together with `CLAUDE.md`, `docs/progress.md` and
`docs/simulator.md` before changing any Phase 3 code. If a decision here
needs to change, update this file in the same commit and record the
reason in section 13.

---

## 1. Goal and research question

**Research question (whole project):** Can supervised machine
learning-based opponent modeling improve decisions made by a
probability-based poker agent?

**Phase 3 contribution:** train and evaluate models that predict an
opponent's next abstract action from public information only. Phase 3
answers the sub-question *"Can an opponent's action be predicted better
than a simple baseline, and does knowing the opponent's earlier behaviour
help?"* Using the predictions inside the probability agent is Phase 4.

**Hypotheses (to be tested, not assumed):**

- H1: Random Forest and HistGradientBoosting beat
  `DummyClassifier(strategy="prior")` on macro-F1 and log-loss.
- H2: Feature Set B (adds the opponent's earlier behaviour) beats Feature
  Set A (current hand only), especially on held-out opponent
  configurations and later in a match when more history exists.

Negative or null results are reported honestly (CLAUDE.md rule 10).

## 2. Approved decisions

| # | Decision |
|---|---|
| 1 | Supervised models predict opponent actions from synthetic data produced by scripted `StyleAgent` bots. |
| 2 | About 40,000 decision rows. |
| 3 | Feature Set A (current-hand public observations) vs Feature Set B (A plus the acting player's earlier public behaviour). |
| 4 | Only information observable before the predicted action is used. |
| 5 | matplotlib is added for academic figures. |
| 6 | Public board texture and betting history are encoded as numerical features. |
| 7 | Models: `DummyClassifier(strategy="prior")` (principal probabilistic baseline), `RandomForestClassifier`, `HistGradientBoostingClassifier`. |
| 8 | Metrics: accuracy, macro-F1, log-loss, confusion matrices, per-style results, held-out configuration results. |
| 9 | Reproducible seeds from master seed **3000**; splits by match. |
| 10 | A four-match pilot is built and reviewed before the full dataset is generated. |
| 11 | 16 opponent configurations: 4 presets plus 3 seeded variants of each. |
| 12 | 48 matches: 36 training-pool matches (26 train / 5 validation / 5 test) and 12 held-out matches. 60 seat-swapped deals (120 hands) per match. |
| 13 | Held-out matches are scored only on decisions made by the 4 held-out configurations. |
| 14 | Feature Set B uses only public actions from **completed deals before the current deal**. Both seat-swapped hands of the current deal are excluded. |
| 15 | History is tracked by match-level player role. The style name and private parameters are never features. |
| 16 | Smoothed priors and an observation-count feature handle sparse history. |
| 17 | All five classes are kept in a fixed order. A legal-action mask is applied at inference and probabilities are renormalized. |
| 18 | Hyperparameters are tuned on validation macro-F1 only. |
| 19 | Match-level bootstrap confidence intervals. Permutation feature importance if practical. |
| 20 | Opponent labels, hidden cards, outcomes, identifiers and private configuration parameters never enter model inputs. |

## 3. Opponent configurations (16)

**Presets** (from `agents/simple.py`, unchanged):

| Preset | looseness | aggression | bluff_rate |
|---|---|---|---|
| calling_station | 1.00 | 0.00 | 0.00 |
| maniac | 0.75 | 0.85 | 0.35 |
| nit | 0.10 | 0.30 | 0.00 |
| solid | 0.45 | 0.60 | 0.08 |

**Variants.** For each preset `p` and `k` in 1, 2, 3, a generator
`random.Random(derive_seed(3000, "variant", p, k))` draws three offsets
uniformly from [−0.10, +0.10], in the fixed order looseness, aggression,
bluff_rate. Each offset is added to the preset value, clipped to [0, 1]
and rounded to 3 decimals. The variant is named `p_v{k}`, e.g.
`maniac_v2`. All configurations use `equity_samples = 200`.

Clipping means some calling_station variants gain a little aggression or
bluffing. This is accepted and reported.

**Held-out configurations.** One variant per preset is excluded from
training. `random.Random(derive_seed(3000, "heldout"))` picks a variant
index from {1, 2, 3} for each preset in the fixed order
calling_station, maniac, nit, solid.

- Training pool: 12 configurations (4 presets + 2 variants each).
- Held-out set: 4 configurations (1 variant per preset).

The generated values are written to the committed manifest (section 5).
They are never written into decision rows.

## 4. Matches, seeds and splits

### 4.1 Seeds

All randomness comes from master seed 3000 through the existing
`hybridpoker.seeding.derive_seed`.

| Purpose | Derivation |
|---|---|
| Variant settings | `derive_seed(3000, "variant", preset, k)` |
| Held-out choice | `derive_seed(3000, "heldout")` |
| Training-pool pairings | `derive_seed(3000, "pairing")` |
| Held-out counterparts | `derive_seed(3000, "heldout_pairing")` |
| Role order within a match | `derive_seed(3000, "roles", match_number)` |
| Split assignment | `derive_seed(3000, "split", attempt)` |
| Match master seed | `derive_seed(3000, "match", match_number)`, numbers 0–47 |
| Pilot match seed | `derive_seed(3000, "pilot", i)`, i = 0–3 (never overlaps main seeds) |
| Model random states | `derive_seed(3000, "model", model_name, feature_set)` |
| Bootstrap | `derive_seed(3000, "bootstrap", evaluation_set)` |

Inside a match, deals and agent decisions use the existing Phase 2
derivation from the match master seed.

### 4.2 Training-pool matches (36, numbers 0–35)

- The 12 training-pool configurations are shuffled with the pairing seed.
- The round-robin "circle method" produces 11 rounds of 6 disjoint
  pairs. The first 6 rounds are used: **36 matches, each configuration
  appears in exactly 6 matches, and no pair repeats.**
- In each match, a seeded coin flip decides which configuration takes
  the `hero` role and which takes `villain` (these are only Phase 2's
  role names; both players are bots).

### 4.3 Splits of the training pool (26 / 5 / 5)

- The 36 matches are shuffled with `derive_seed(3000, "split", attempt)`.
  The first 26 are **train**, the next 5 **validation**, the last 5
  **test**.
- The draw is accepted only if (a) every training-pool configuration
  appears in at least 3 train matches and (b) validation and test each
  contain at least one configuration from every preset family.
  Otherwise `attempt` increases by 1. The accepted attempt number is
  recorded.
- The whole match is the unit, so both seatings of a deal and all hands
  of a match stay in one split.

### 4.4 Held-out matches (12, numbers 36–47)

- Each of the 4 held-out configurations plays 3 matches, one against a
  training-pool configuration from each of the **other three** preset
  families. The counterpart within a family is a seeded choice.
- Held-out matches are never used for training, tuning, smoothing priors
  or feature selection.
- **Only rows made by the held-out configuration are scored.** Rows by
  the training-pool counterpart in these matches are not used.

### 4.5 Expected size

Phase 2 bot matches averaged about 5.4–8.2 decisions per hand, so a
120-hand match should give about 650–1,000 rows.

| Set | Matches | Rows (estimate) |
|---|---|---|
| Train | 26 | about 17,000–26,000 |
| Validation | 5 | about 3,300–5,000 |
| Test | 5 | about 3,300–5,000 |
| Held-out (scored rows only, about half) | 12 | about 3,900–6,000 |
| **All generated rows** | 48 | **about 31,000–48,000** |

These are estimates. The pilot measures the real values.

### 4.6 Runtime

About 25 decisions per second per process for bot-only matches (Phase 2
benchmark). That is about 25–30 minutes in one process, or about 7–10
minutes with 4 worker processes. The pilot confirms this.

## 5. Dataset manifest and storage

| What | Where | In git? |
|---|---|---|
| Full manifest | `data/manifests/phase3_manifest.json` | Yes (small) |
| Pilot manifest | `data/manifests/phase3_pilot_manifest.json` | Yes (small) |
| Raw match logs | `data/raw/simulations/phase3_s3000/match_NN/` | No (ignored) |
| Pilot logs | `data/raw/simulations/phase3_pilot_s3000/match_NN/` | No (ignored) |
| Feature tables | `data/processed/phase3/` | No (ignored) |
| Trained models | `models/phase3/` | No (ignored) |
| Metric tables (CSV/JSON) | `results/phase3/` | Yes (small) |
| Figures (PNG) | `docs/figures/phase3/` | Yes (small) |

The manifest records:

- schema version and master seed;
- all 16 configurations with parameters and pool membership;
- for every match: number, master seed, hero and villain configuration,
  split (`train`, `validation`, `test` or `heldout`; the pilot uses
  `pilot_pool` and `pilot_heldout`), the scored role (`both`, or the
  role of the held-out configuration), and the deal count;
- the accepted split attempt number;
- `prior_splits`: the splits allowed to fit the history priors (`train`;
  `pilot_pool` in the pilot).

The manifest deliberately contains **no timestamps, git commit or
software versions**, so rebuilding it gives byte-identical JSON (audit
L11). Those run-level details are recorded in each match's
`metadata.json` (Phase 2 format: versions and git commit) and in
`generation_summary.json` (timings, worker count, machine, versions, git
commit) next to the raw logs. `scripts/build_manifests.py --check`
verifies the committed manifests.

Each match's `metadata.json` (Phase 2 format) also stores the match
number, split and manifest hash in its `extra` field. A run is complete
only if its `metadata.json` exists (Phase 2 atomic-write rule). The
loader refuses incomplete runs and runs whose seed or configurations
disagree with the manifest.

## 6. Prediction target

- Target: `act_category`, the abstract action the actor chose.
- **Fixed class order** (the `ActionCategory` enum order):
  `fold`, `check_call`, `raise_half_pot`, `raise_pot`, `all_in`.
- Every model is checked to have `classes_` in exactly this order, and
  log-loss is always computed with `labels=` set to this order.
- The category is the actor's intent. When clamping turns a pot raise
  into an all-in, the label is still `raise_pot` (Phase 2 convention).

## 7. Features

### 7.1 Allowed inputs

The feature builder receives only:

- `obs_` columns (public, before the decision), and
- three grouping keys used for ordering and history, never as features:
  `id_match`, `id_deal_index`, and the **player key** (section 7.4).

`hidden_`, `label_`, `out_`, `act_` (except as the target), all other
`id_` columns, the manifest's configuration names and parameters are
never passed to the feature builder.

### 7.2 Feature Set A: current-hand public observations

Prefix `a_`. Chip amounts are divided by the big blind.

- **Situation:** street (one-hot), position (one-hot), pot, amount to
  call, pot odds, own and opponent stacks, effective stack,
  stack-to-pot ratio, own and opponent bets, facing a bet (0/1).
- **Legal actions:** the five `obs_legal_*` flags and the legal bet
  sizes in big blinds (0 where illegal, with the flag showing
  legality).
- **Board texture** from `obs_board`: number of board cards, highest
  rank, number of broadway cards, paired board, trips on board, largest
  suit count (monotone / flush-draw), straight connectivity (most
  distinct ranks within any 5-rank window, ace counting high and low),
  and rank spread.
- **Betting history** from `obs_history` (current hand only, actions
  before this decision):
  - actions and raises this street (existing columns);
  - raises this hand by the actor and by the opponent;
  - the previous action's category (one-hot, including "none");
  - the preflop aggressor (actor / opponent / none);
  - whether the opponent is all-in;
  - number of streets with a bet or raise so far.

`obs_big_blind` is constant and dropped. Raw strings (`obs_board`,
`obs_history`) are not model inputs; only the numbers derived from them
are.

### 7.3 Feature Set B: A plus earlier behaviour

Prefix `b_`. Computed for the acting player, from **that player's own
public actions in deals with a smaller `id_deal_index` in the same
match**. Both hands of the current deal are excluded, because the two
seatings of a deal share the same cards and board.

| Feature | Meaning |
|---|---|
| `b_log_prior_hands` | log(1 + number of earlier hands observed) |
| `b_log_prior_decisions` | log(1 + number of earlier decisions observed) |
| `b_open_{check, half, pot, allin}` | Smoothed frequency of each action when not facing a bet |
| `b_face_{fold, call, half, pot, allin}` | Smoothed frequency of each action when facing a bet |
| `b_vpip` | Smoothed share of hands where the player voluntarily put chips in preflop |
| `b_pfr` | Smoothed share of hands where the player raised preflop |

**Smoothing:** `rate = (count + α × prior) / (n + α)` with α = 10
pseudo-observations, fixed before any results are seen. The priors are
the population rates computed **from train-split rows only** and frozen.
With no history, each feature equals its prior and the count features
are 0.

Showdown cards from earlier hands are not used, even though they would
be public at a real table. This keeps the comparison focused on betting
behaviour.

### 7.4 Player key (history grouping without labels)

In a seat-swapped match the `hero` role sits on the button when
`id_seat_swap = 0` and in the big blind when `id_seat_swap = 1`. The
player key is therefore derived from public seating information:

```
player_role = "hero" if (id_seat_swap == 0) == (obs_position == "button") else "villain"
```

This identifies *which seat at the table* acted, as a real player would
know, without reading `label_actor_agent` or `label_actor_role`. A test
confirms it equals `label_actor_role` in every row. That test is the only
place the label is read.

## 8. Models and training

| Model | Notes |
|---|---|
| `DummyClassifier(strategy="prior")` | Principal probabilistic baseline. Ignores features, so it is the same for A and B. |
| `RandomForestClassifier` | Fit separately on A and on B. |
| `HistGradientBoostingClassifier` | Fit separately on A and on B. **`early_stopping=False`**, because its built-in early stopping holds out random rows, which would break match-level splitting. |

**Tuning:** a small fixed grid per model, fit on train and scored by
**validation macro-F1 after legal-action masking**. Nothing else is used
to choose hyperparameters. Planned grids:

- Random Forest: `n_estimators` 300; `max_depth` {None, 12, 24};
  `min_samples_leaf` {1, 5, 20}; `max_features` {"sqrt", 0.5};
  `class_weight` {None, "balanced"}.
- HistGradientBoosting: `learning_rate` {0.05, 0.1}; `max_iter`
  {200, 400}; `max_leaf_nodes` {15, 31, 63}; `l2_regularization`
  {0, 1}; `class_weight` {None, "balanced"}.

The final models are fit on **train only** with the selected settings.
Validation is used only for selection, and test and held-out data only
for the final report.

**Legal-action mask at inference:**

1. Predict probabilities for all five classes.
2. Set probabilities of illegal actions (from `obs_legal_*`) to 0.
3. Renormalize the legal probabilities to sum to 1 (if a model gave
   every legal action 0, use a uniform distribution over legal actions).
4. Raise every legal probability to at least 1e-6 and renormalize again,
   so each row sums to exactly 1. This keeps log-loss finite when a model
   gives 0 to a legal action.
5. Keep the raw (unmasked) probabilities unchanged alongside the masked
   ones for diagnostics, e.g. probability wasted on illegal actions.
6. The predicted class is the most probable legal action.

The same masking is applied to every model, including the Dummy
baseline, so comparisons are fair. Implemented and tested in Stage 1 as
`action_probabilities.mask_and_renormalize`.

## 9. Evaluation

**Evaluation sets:**

- **Test** (5 matches): seen configurations, new matches and cards.
- **Held-out** (12 matches): unseen configurations, scored on the
  held-out player's rows only.

**Metrics** (masked predictions): accuracy, macro-F1, log-loss, and
confusion matrices (counts and row-normalized). Also reported:

- per-style results by the actor's preset family, using labels for
  evaluation only;
- results by street;
- B minus A as a function of how many earlier hands were observed.

**Uncertainty:** match-level bootstrap with 2,000 resamples. Matches are
resampled with replacement and the metric is recomputed on all their
rows. B vs A and model vs Dummy differences are bootstrapped on the
same resamples (paired). With only 5 test matches the test intervals
will be wide, and this is stated. The 12-match held-out set gives the
stronger evidence.

**Permutation importance** (if practical): on the test set, by feature
group (situation, legal actions, board texture, betting history,
opponent history), measured as the drop in masked macro-F1 over 5
repeats.

**Sanity check (evaluation only):** compare each configuration's
predicted raise and fold rates with its true parameters from the
manifest.

**Figures** (matplotlib, saved to `docs/figures/phase3/`):

1. Class distribution by preset family.
2. Model comparison with bootstrap CIs (macro-F1, log-loss) on test and
   held-out.
3. Confusion matrices of the best A and B models.
4. Per-style macro-F1, A vs B.
5. B minus A against the number of earlier hands observed.
6. Permutation importance by feature group.

## 10. Leakage and reproducibility tests

| # | Test | Checks |
|---|---|---|
| L1 | Column allowlist | Model input names start only with `a_` or `b_`. No name contains `id_`, `hidden_`, `label_`, `out_`, `act_`, a style name or a parameter name (`looseness`, `aggression`, `bluff_rate`). |
| L2 | Hidden-column scrambling | Shuffling every `hidden_`, `label_`, `out_` column and the non-key `id_` columns leaves all features unchanged. |
| L3 | Future truncation (A) | A row's A features are unchanged when all later decisions of the hand are deleted. |
| L4 | Future truncation (B) | B features for deal *d* are unchanged when the actions of deal *d* and all later deals are altered or deleted. |
| L5 | Mirror-hand exclusion | Changing actions in the first seating of deal *d* does not change B features in the second seating of deal *d*. |
| L6 | Public replay | A features rebuilt with fake hole cards match exactly (extends the Phase 2 audit). |
| L7 | Player key | The derived player key equals `label_actor_role` in every row (the only test that reads a label). |
| L8 | Split integrity | No match is in two splits. No held-out configuration appears in train, validation or test. Held-out scored rows come only from held-out configurations. |
| L9 | Seed uniqueness | All 48 match seeds and 4 pilot seeds are distinct. |
| L10 | Train-only priors | Changing validation, test or held-out rows does not change the smoothing priors. |
| L11 | Manifest reproducibility | Building the manifest twice gives byte-identical JSON. Re-running one match from the manifest gives a byte-identical `decisions.csv`. |
| L12 | Class order and masking | `classes_` follows the fixed order. Masked probabilities are 0 for illegal actions and sum to 1. |
| L13 | Shuffled-label control | A model trained on shuffled labels performs no better than the Dummy baseline. This is a report check, not a unit test. |

## 11. Implementation stages and review gates

**Stage 0: plan.** This document. Done.

**Stage 1: pilot (stops for review).**

- Configuration generator, manifest builder, parallel generation script,
  loader, Feature Sets A and B, and leakage tests L1–L11.
- Pairing-balance tests: every training-pool configuration plays exactly
  6 matches, no pair repeats, no self-pairs, per-split appearances add
  up, and every held-out configuration meets three different families.
- Probability masking, clipping and renormalization, with raw
  probabilities kept for diagnostics, and its tests.
- Each audit is also tested against a deliberately leaky input, so a
  passing audit is meaningful.

Stage 1 files:

| File | Purpose |
|---|---|
| `src/hybridpoker/opponent_configs.py` | 16 configurations, held-out choice |
| `src/hybridpoker/dataset_manifest.py` | Pairings, roles, seeds, splits, manifest JSON |
| `src/hybridpoker/dataset_generation.py` | Play one manifest match and write its logs |
| `src/hybridpoker/dataset_loader.py` | Load complete runs, check them against the manifest |
| `src/hybridpoker/features.py` | Feature Set A |
| `src/hybridpoker/opponent_history.py` | Feature Set B, player role, priors |
| `src/hybridpoker/action_probabilities.py` | Fixed class order, legal mask, clipping |
| `src/hybridpoker/leakage.py` | Audits L1–L11 |
| `scripts/build_manifests.py` | Write or `--check` the committed manifests |
| `scripts/generate_dataset.py` | Parallel generation from a manifest |
| `scripts/pilot_report.py` | Pilot report (`results/phase3/pilot_report.md`) |
| `data/manifests/*.json` | Committed manifests |
| `tests/test_*` (7 new files + `conftest.py`) | Tests for all of the above |

- 4 pilot matches with pilot seeds:
  - 3 training-pool pairings covering all four preset families;
  - 1 held-out configuration against a training-pool configuration;
  - 60 deals each.
- A pilot report with:
  - rows per match;
  - class counts overall and per family;
  - rows per actor;
  - feature summaries;
  - B-feature behaviour across deals;
  - timing and the extrapolated full runtime;
  - the full test results.
- Pilot data is never used for training, tuning or reported results.

**Stage 2: full dataset.** Generate the 48 matches from the committed
manifest, run the data audit, and report the real row counts.

**Stage 3: training and evaluation.** Add matplotlib (pinned) and the
model, evaluation and figure code with tests L12–L13. Then tune, run the
final evaluation, and write the figures and results. Update
`docs/progress.md` and open the Phase 3 pull request.

**Proposed timeline:**

| Dates | Work |
|---|---|
| Nov 12–14 | Stage 1 pilot, then review |
| Nov 15–17 | Stage 2 full dataset |
| Nov 18–22 | Stage 3 training, evaluation, figures; Phase 3 PR |
| Nov 23–Dec 1 | Phase 4: use predictions in the probability agent; evaluation matches |
| Dec 2–6 | Streamlit demo, slides, buffer |
| Dec 7–9 | Competition |

Phase 4 is where the research question is finally answered, so Phase 3
must not overrun much past Nov 22.

## 12. Acceptance criteria

**Pilot (Stage 1):**

- All 4 pilot matches complete with `metadata.json`, 120 hands each, and
  manifest-consistent seeds and configurations.
- Re-running one pilot match gives a byte-identical `decisions.csv`.
- Leakage tests L1–L11 pass, and the full test suite passes on
  Python 3.12.
- The derived player key matches `label_actor_role` in 100% of rows.
- B features equal their priors with zero counts in deal 0, and the
  counts never decrease across deals.
- No unexpected missing values in the feature table.
- Class counts are reported. Any class with fewer than 1% of rows is
  flagged.
- Measured decisions per second and the extrapolated full runtime are
  reported.

**Phase 3 overall:**

- 48 matches generated from the committed manifest. Real row counts are
  reported, and splits are 26/5/5 plus 12 held-out.
- Dummy, RF-A, RF-B, HGB-A and HGB-B are evaluated on test and held-out
  with accuracy, macro-F1, log-loss, confusion matrices, per-style
  results and bootstrap CIs.
- Hyperparameters are selected on validation macro-F1 only, and the
  selected values are recorded.
- Figures and metric tables are committed. Models and data are not.
- All leakage tests pass, and the shuffled-label control is reported.
- `docs/progress.md` is updated with the decisions, tests, actual
  results and limitations.

## 13. Known limitations and open risks

- **Synthetic opponents.** The models learn rules we wrote. Held-out
  configurations test generalization only to nearby variants of the same
  four styles.
- **Small test split.** 5 matches give wide intervals.
- **Rare classes.** `all_in` (and `fold` for calling-station-like
  configurations) may be rare, which makes macro-F1 noisy.
- **Bot equity noise.** The bots use 200-sample equity, so some actions
  are partly random given the public information. This caps achievable
  accuracy, which is expected.
- **Short history.** 60 deals per match limits how much earlier
  behaviour Feature Set B can see.
- **Schedule.** If the pilot shows much lower rows per match or much
  slower generation than estimated, the deal count or worker count may
  change, but only with approval and a note here.

## 14. Change log

| Date | Change |
|---|---|
| 2026-11-12 | Plan written from the approved decisions. |
| 2026-11-12 | Implementation plan and six detail decisions approved. Added: pairing-balance tests; probability clipping with renormalization and preserved raw probabilities. Manifest excludes git commit and versions (kept in run metadata and the generation summary) so it rebuilds byte-identically. Pilot split names `pilot_pool` / `pilot_heldout`. |
