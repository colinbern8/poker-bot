# HybridPoker Progress Log

## Phase 0: Project setup (2026-11-11)

### Completed work

- Python 3.12 project with a `src/` layout and the `hybridpoker` package.
- Dependencies declared in `pyproject.toml` (ranges) and pinned in
  `requirements.txt` (exact versions) for reproducibility.
- GitHub Codespaces configuration (`.devcontainer/devcontainer.json`) using the
  official Python 3.12 dev container image.
- GitHub Actions workflow running pytest on Ubuntu and Windows.
- Smoke tests for the environment and PokerKit heads-up No-Limit Hold'em.
- README with setup instructions for Codespaces, Linux/macOS, and Windows.

### Important engineering decisions

| Decision | Reason |
|---|---|
| `src/` layout, editable install | Tests import the installed package, not files that just happen to be in the current folder. |
| `venv` + `pip` (no Poetry/Conda/uv) | Built into Python, so it works the same on Windows, Linux, and Codespaces with nothing extra to learn. |
| Exact pins in `requirements.txt` | Experiments must be reproducible, so library upgrades happen only on purpose. |
| `requires-python = ">=3.12,<3.13"` | The project is standardized on 3.12. Codespaces' default image ships 3.14. |
| PokerKit pinned to `0.7.7` (range `<0.8`) | PokerKit's `create_state` API has changed between minor versions. |
| Platform markers for `watchdog` and `colorama` | `pip freeze` on Linux misses Windows-only and macOS-excluded packages. |

### PokerKit API notes (verified against 0.7.7)

- `NoLimitTexasHoldem.create_state(automations, ante_trimming_status,
  raw_antes, raw_blinds_or_straddles, min_bet, raw_starting_stacks,
  player_count)`. The default `mode` is `Mode.TOURNAMENT`.
- **Heads-up seating:** with blinds `(1, 2)`, player index 0 posts the
  **big blind** and index 1 posts the **small blind** and acts first preflop.
  Later phases must keep this in mind when mapping positions.
- Hole-card and board dealing are not automated in our tests. They are called
  explicitly with `deal_hole()` / `deal_board()`.
- PokerKit shuffles its deck with its own randomness. Seeding it for
  reproducible experiments is future work, not Phase 0.

### Tests executed

Run on 2026-11-11 in a fresh Python 3.12.3 virtual environment (Linux):

```
$ python -m pytest
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0
collected 13 items
tests/test_smoke.py .............                                        [100%]
============================== 13 passed in 1.15s ==============================
```

Windows (GitHub Actions, run 43113218578, `windows-latest`):
`platform win32 -- Python 3.12.10, pytest-9.2.0` with `13 passed in 2.17s`.
Ubuntu (same run): success.

### Experimental results

None. Phase 0 contains no experiments.

### Known problems (resolved)

- Codespaces setup was verified manually by the project owner. The virtual
  environment must be activated in each new terminal
  (`source .venv/bin/activate`); the README now says so.
- Windows compatibility was confirmed by GitHub Actions (above).
- Seeded card dealing is handled in Phase 1 by our own equity simulator.

Merged into `main` through pull request #1.

## Phase 1: Probability engine (2026-11-12)

Branch: `phase-1-equity-engine`.

### Completed work

- `src/hybridpoker/equity.py`:
  - `estimate_equity` is a seeded Monte Carlo equity estimate for
    1 to 9 opponents, with win, tie and loss rates and the standard error.
  - `exact_river_equity` is exact heads-up river equity by enumerating
    all 990 opponent hands.
  - `summarize_pot_shares` and `hero_pot_share` hold the split-pot
    accounting.
  - `parse_cards` provides input parsing and validation.
- `src/hybridpoker/pot_odds.py`:
  - `pot_odds` gives the break-even equity.
  - `call_expected_value` gives the simplified EV of calling compared with
    folding.
- `scripts/benchmark_equity.py` measures speed.
- `docs/probability_engine.md` is a plain-language explanation of the math.

### Important engineering decisions

| Decision | Reason |
|---|---|
| Own Monte Carlo loop instead of PokerKit's `calculate_equities` | PokerKit's function has no seed parameter, so results could not be reproduced. PokerKit is still used for all hand evaluation (`StandardHighHand`). |
| `seed` is a required keyword argument | Forces every experiment to record its seed (CLAUDE.md rule 8). |
| Split pots count as 1/n of the pot | Correct equity for ties. A royal-flush board gives 1/(players). |
| Standard error from fractional pot shares | A binary win/loss indicator misstates uncertainty when splits are common. |
| `pot_size` includes the opponent's bet and excludes the hero's call; `call_amount > pot_size` is rejected | One consistent convention across the project. The check catches the most common mistake, leaving the bet out of the pot. It is listed as assumption 7 of the simplified heads-up API in `docs/probability_engine.md`. |
| Exact enumeration limited to heads-up river | 990 combinations take about 0.3 s. Two opponents would need about 450,000, which takes minutes. |
| No decision rule (call/fold) in Phase 1 | CLAUDE.md rule 7 says to separate math from decision rules. Rules come in Phase 2. |
| No new dependencies | Only the standard library and PokerKit are used. |

### Tests executed

Local run (Linux, Python 3.12.3), `python -m pytest`: `82 passed in 12.83s`.

GitHub Actions run 43276364776 on commit `398dabe`:

- `windows-latest`: `platform win32 -- Python 3.12.10`, `82 passed in 20.70s`
- `ubuntu-latest`: Python 3.12.16, `82 passed in 21.72s`

This includes `test_seeded_results_are_identical_across_platforms`, so the
same seed gives the same equity on Windows as on Linux.

After adding the hand-counted river tests (8 more tests), the local run gave
`90 passed in 16.38s`. CI results for the final commit are shown on the
Phase 1 pull request.

- `tests/test_equity.py` (51 tests):
  - hand-counted river examples, with exact answers derived on paper and
    checked against both exact enumeration and Monte Carlo
  - validation
  - split pots for 1, 2, 3, 5 and 9 opponents
  - the standard-error formula
  - exact river enumeration
  - Monte Carlo within 4 SE of exact in 4 river spots
  - reproducibility, including fixed regression values checked on Windows
    in CI
  - preflop sanity orderings
- `tests/test_pot_odds.py` (26 tests):
  - worked examples
  - EV equals 0 exactly at pot odds
  - EV extremes
  - validation
- `tests/test_smoke.py` (13 tests): unchanged from Phase 0.

### Benchmark measurements

`python scripts/benchmark_equity.py` on the cloud development container
(Linux, 4 vCPU Intel Xeon, Python 3.12.3), with 5,000 samples per spot:

| Spot | Samples per second | Equity ± SE |
|---|---|---|
| AsKh preflop, 1 opponent | 3,257 | 0.665 ± 0.007 |
| AsKh on Td9d2c, 1 opponent | 3,096 | 0.500 ± 0.007 |
| AsKh on Td9d2c4s7h, 1 opponent | 3,435 | 0.345 ± 0.007 |
| AsKh preflop, 5 opponents | 1,608 | 0.278 ± 0.006 |
| Exact river (990 hands) | 0.288 s total | |

These are timings on one machine, not accuracy results.

### Monte Carlo vs exact river equity (from development checks)

3,000 to 4,000 samples, single seeds. All estimates were within 2 SE of the
exact value:

| Hand | Board | Exact equity | Exact tie rate |
|---|---|---|---|
| AsKd | Ah7c2d9sJh | 0.8889 | 0.0061 |
| Td9d | 8d7c2s3hKh | 0.1258 | 0.0091 |
| 2c3d | AhKhQhJh5s | 0.0081 | 0.0162 |
| 5c5d | AhAdAsKcKd | 0.4773 | 0.9545 |

### Mathematical limitations

- Opponent hands are assumed uniformly random. This ignores what their
  betting reveals. The ML opponent model is meant to address this.
- Call EV assumes no future betting, full equity realization, a heads-up
  final pot, no rake, and `call_amount <= pot_size`. These are the
  simplified heads-up API assumptions in `docs/probability_engine.md`.
- Independent verification covers hand-counted river spots only. Preflop,
  flop and turn equities are checked for consistency (ordering, multiway
  trends, agreement with the same engine), not against published values.
- The standard error uses a normal approximation. It is unreliable for very
  small `sample_count` and is NaN for a single sample.
- Exact enumeration is heads-up river only.

### Known problems

- Speed: about 3,000 samples per second heads-up. A decision with 1,000
  samples costs about 0.3 s, so a 10,000-hand experiment with several
  decisions per hand could take hours. Options for Phase 2 and 4 are
  fewer samples, caching preflop equities, or exact river calculation.

### Next milestone

Phase 2: heads-up simulator, a probability-based agent, simple opponent
bots, and hand logging. It needs a plan approved before work starts.

Phase 1 was merged into `main` through pull request #2.

## Phase 2: Heads-up simulator, agents and hand logs (2026-11-12)

Branch: `phase-2-simulator`.

### Completed work

- `game.py`: heads-up NLHE settings (blinds 1/2, 100 BB stacks reset each
  hand, no antes or rake, PokerKit cash-game mode).
- `actions.py`:
  - five abstract categories with legal-action masking
  - pot-size raise amounts clamped to PokerKit's legal range
  - actual chip amounts (`bet_to`, `chips_added`, all-in flag)
- `observation.py`: the acting player's view. It includes own cards only,
  never the opponent's.
- `simulator.py`:
  - seeded deck handed card-by-card to PokerKit
  - full hand loop including all-in run-outs
  - state recorded before each decision
  - chip-conservation check
- `agents/`:
  - `RandomAgent`
  - `StyleAgent` with presets calling_station, maniac, nit and solid
  - `ProbabilityAgent` (baseline), with a logged reason for every decision
  - a cached preflop equity per hand class
- `match.py`:
  - seat-swapped (duplicate) matches with paired deal seeds
  - one master seed per match
  - opaque `match_id`
- `metrics.py`: bb/100 with a 95% CI using the deal pair as the
  independent unit.
- `hand_log.py`:
  - `decisions.csv`, `hands.csv` and `metadata.json`
  - prefixed column groups
  - the `feature_columns()` selector
  - atomic writes: a run is complete only if `metadata.json` exists
- Scripts: `run_match.py` and `benchmark_simulator.py`.
- Docs: `docs/simulator.md`.

### Important engineering decisions

| Decision | Reason |
|---|---|
| Own seeded deck, cards handed to PokerKit | PokerKit's own dealing cannot be seeded; this makes every hand replayable. |
| `fold` masked when checking is free | Folding for free is never better. PokerKit also warns about it. |
| Category and actual amounts logged separately | Clamping can turn a pot raise into an all-in. The category is the intent; the amounts are what happened. |
| Seat-swapped deals with paired seeds | Both agents play both sides of every deal, which cancels much of the card luck. |
| CI over deal pairs, not hands | The two hands of a pair share cards and are correlated. A test shows identical strategies give exactly 0 per pair. |
| Column prefixes (`id_`, `obs_`, `act_`, `hidden_`, `label_`, `out_`) and `feature_columns()` | Model inputs can only come from public, pre-decision information. |
| Bot parameters only in `metadata.json` | Private bot settings never appear in per-decision rows. |
| Opaque `match_id` from full agent configs, seed and deal count | Combined logs can be grouped and split by independent match. |
| Atomic log writes, flushed output | An interrupted run cannot leave files that look complete. This was found when the long matches were stopped (see below). |

### Tests executed

Local run (Linux, Python 3.12.3), `python -m pytest`: 183 tests passed.

| File | Tests | Covers |
|---|---|---|
| `test_actions.py` | 14 | legal masks, raise sizes, clamping, all-in, chip movements, observations |
| `test_simulator.py` | 17 | showdown, split and fold payouts, uncalled raises, all-in run-outs, legality and payout consistency over 400 random hands, no opponent-card leakage, reproducibility |
| `test_agents.py` | 29 | random legality, style differences, calling station never folds or raises, probability decisions on river spots with known exact equities |
| `test_match.py` | 7 | seat swapping, paired cards, identical strategies give 0 per pair, reproducibility, match ids |
| `test_metrics.py` | 9 | bb/100 and CI on hand-computed examples |
| `test_hand_log.py` | 17 | column groups, public-information replay, pre-action state, feature selector exclusions, ids, amounts, byte-identical logs, atomic writes |

The GitHub Actions results for the final commit are shown on the Phase 2
pull request.

### Simulator speed (clean benchmark)

`python scripts/benchmark_simulator.py --deals 50`: one process on an idle
machine (Linux container, 4 vCPU Intel Xeon, Python 3.12.3), default equity
samples (probability 500, style bots 200), preflop cache filled first
(36.9 s), 100 hands per matchup.

| Matchup | Hands/s | Decisions/s | ms per decision | Decisions per hand |
|---|---|---|---|---|
| random vs random (no equity) | 1,312.9 | 4,463.9 | 0.22 | 3.40 |
| solid vs solid (scripted only) | 2.7 | 22.3 | 44.89 | 8.18 |
| maniac vs nit (scripted only) | 4.9 | 26.8 | 37.35 | 5.43 |
| probability vs solid | 1.6 | 12.9 | 77.67 | 8.26 |

**Bottleneck.** A cProfile run of probability vs solid attributed 99.9% of
run time to `estimate_equity`, including 96.2% inside PokerKit's hand
evaluator (`StandardHighHand.from_game`). Game logic (legal actions and
PokerKit state updates) took about 0.1%. The random-vs-random row confirms
this: without equity, a decision costs 0.22 ms instead of 37–78 ms.

### Sanity matches (diagnostic only)

The five planned 1,000-deal matches were stopped after about 12 minutes to
save time. None had finished, so **no results from them exist**. One
(vs random) had finished playing but was interrupted while writing its
logs; that led to the atomic-write fix above.

Replacement diagnostic matches: probability agent vs each opponent, 75
seat-swapped deals (150 hands) each, seed 2026, default settings, 5
processes in parallel (139 s in total). **These results are diagnostic
only. With 75 deal pairs the intervals are wide, and they are not evidence
about agent strength.** Their purpose is to show that full matches complete
and log correctly.

| Opponent | Hero bb/100 | 95% CI (deal pairs) | SE paired | SE naive (per hand) |
|---|---|---|---|---|
| random | +1167.3 | [+441.5, +1893.2] | 370.3 | 420.5 |
| calling_station | +175.3 | [+115.0, +235.7] | 30.8 | 44.6 |
| maniac | +1118.0 | [+644.3, +1591.7] | 241.7 | 439.0 |
| nit | −15.7 | [−249.8, +218.4] | 119.4 | 137.3 |
| solid | +87.3 | [−215.1, +389.7] | 154.3 | 289.7 |

All five runs wrote complete logs: 150 hands each, 5,569 decision rows in
total (2,805 hero, 2,764 villain). The paired standard error was below the
naive one in every match.

### Log audit on the diagnostic logs

- 45 columns, of which 26 are `obs_` features.
- Rebuilding every row's `obs_` values from the public history and board
  with **fake hole cards** gave **0 mismatches in 5,569 rows**. The
  features contain no private information.
- `(id_match, id_hand_index, id_decision_index)` is present and unique for
  every row.
- `act_bet_to` equals the legal amount for the chosen category in every
  row. 1,387 bet and raise rows used 59 distinct actual amounts. 56
  half-pot or pot raises were capped to all-in and are recorded as such
  (`act_is_all_in`).

### Phase 3 dataset-generation plan (recommended, not yet implemented)

**Goal:** 20,000–50,000 opponent decision rows, reproducible and split by
independent match.

1. **Generate with scripted agents only.** Use StyleAgent against
   StyleAgent, at 22–27 decisions/s per process. Both seats are bots with
   known settings, so every row is a labelled opponent decision. Keep the
   probability agent for evaluation matches, where its 500 samples matter.
2. **Vary opponent parameters.** Use the 4 presets plus 3 randomly shifted
   variants of each (looseness, aggression and bluff rate each moved by up
   to ±0.10, kept within [0, 1]). That gives 16 configurations, each with a
   unique name such as `maniac_v2`. Variant parameters are drawn from a
   seeded generator, written to a dataset manifest, and stored in each
   match's `metadata.json`, never in rows.
3. **Matches and seeds.** Each match pairs one configuration with a
   counterpart drawn from the other configurations, for 60 deals
   (120 hands, about 600–1,000 rows). Every match gets its own master seed
   from a recorded list (e.g. 10000, 10001, …), and seeds are never reused.
   About 48 matches gives about 40,000 rows.
4. **Run time.** At about 25 decisions/s per process, 40,000 rows take
   about 27 minutes in one process. Running 4 independent matches at once
   on 4 cores should take about 7–10 minutes, plus a cache warm-up per
   process. This estimate needs confirming in Phase 3.
5. **Splits, fixed before any training.**
   - Group every split by `id_match`. Both seatings of a deal and all hands
     of a match stay together, so shared cards never cross splits.
   - Train / validation / test: about 70 / 15 / 15 of the matches built from
     the 12 training configurations.
   - **Held-out evaluation set:** the 4 configurations never seen in
     training (one variant per preset), in their own matches with new
     seeds. This tests generalization to unseen opponents, not just new
     cards.
   - Record the split assignment in the manifest.
6. **Leakage rules.**
   - Model inputs come only from `feature_columns()`.
   - Targets come from `act_category` (and optionally `act_bet_to`).
   - Labels come from `label_actor_agent`, for evaluation only.
   - `id_*`, `hidden_*`, `label_*` and `out_*` columns are never inputs.
     `id_deal_seed` determines the cards.
   - The `obs_history` text must be turned into features using only actions
     before the current decision. It already contains nothing else.
7. **Storage.** Raw runs go under `data/raw/simulations/<dataset>/`, which
   git ignores. The small manifest (configurations, seeds, splits) is
   committed so the dataset can be regenerated exactly.

### Known problems and limitations

- **Speed.** Monte Carlo equity dominates run time. Statistically
  meaningful probability-agent evaluations (hundreds to thousands of deal
  pairs) will take tens of minutes to hours. They are not optimized in this
  phase, by decision.
- **No progress output.** `run_match.py` reports only at the end of each
  match; there are no partial results.
- **Synthetic opponents.** Models will learn rules we wrote. This is stated
  openly and can be checked against known parameters.
- **Abstract actions.** Only five actions, with fixed 100 BB stacks.
- **Untuned thresholds.** The probability agent's thresholds are set by
  reasoning, not tuned.
- **Normal-approximation CI.** It is too narrow with very few deal pairs.

### Next milestone

Phase 3: generate the dataset per the plan above, engineer features from
`obs_` columns, and train supervised opponent models against a simple
baseline. It needs a plan approved before work starts.

## Phase 3: Supervised opponent modeling (in progress)

Branch: `phase-3-machine-learning`.

The approved design is in [`docs/phase3_plan.md`](phase3_plan.md). It
covers the 16 opponent configurations, the 48 matches from master seed
3000, the 26/5/5 + 12 held-out match splits, Feature Sets A and B, the
models, metrics and leakage tests.

### Stage 1: pilot (2026-11-12): completed work

- `opponent_configs.py`: 4 presets plus 3 seeded variants each. One
  variant per preset is held out (calling_station_v3, maniac_v3, nit_v1,
  solid_v2).
- `dataset_manifest.py`:
  - 36 training-pool matches from a round-robin schedule (each
    configuration plays exactly 6, no repeated pairs);
  - a balanced 26/5/5 split, accepted on the first draw;
  - 12 held-out matches;
  - a 4-match pilot with separate seeds.
  Both manifests are committed under `data/manifests/` and rebuild
  byte-identically.
- `dataset_generation.py`, `scripts/generate_dataset.py`: parallel
  generation from a manifest.
- `dataset_loader.py`: accepts only complete runs that match the manifest
  (seed, configurations, manifest hash).
- `features.py`: Feature Set A, 47 features (situation, legal actions,
  board texture, betting history).
- `opponent_history.py`: Feature Set B, with 13 history features from
  earlier deals only. The mirror hand is excluded, the player role comes
  from public seating, and rates use alpha = 10 smoothing toward
  train-only priors.
- `action_probabilities.py`:
  - fixed class order;
  - legal-action mask, a 1e-6 floor and renormalization;
  - raw probabilities kept for diagnostics.
- `leakage.py`: audits L1–L11. Each is also tested against a deliberately
  leaky input.
- `scripts/pilot_report.py` produces
  [`results/phase3/pilot_report.md`](../results/phase3/pilot_report.md).

### Engineering decisions (Stage 1)

| Decision | Reason |
|---|---|
| Manifest has no timestamps, versions or git commit | It must rebuild byte-identically. Those details are in each run's `metadata.json` and in `generation_summary.json`. |
| Feature builders select an explicit column list first | They physically cannot read hidden, label, outcome or id columns. |
| Player role from `id_seat_swap` and `obs_position` | Tracks "the opponent at this seat" the way a real player could, without reading labels. |
| History counts are summed per deal, then shifted by one deal | Excluding the whole current deal (both seatings) is built into the computation, not a filter that could be forgotten. |
| Audits tested against deliberate leaks | A passing audit only means something if it fails on a leak. |

### Pilot results (4 matches, 60 deals each; validation only, not used for any reported result)

- **Size:** 3,669 decision rows (3,133 scored). Rows per match ranged
  from 698 (maniac vs nit_v2) to 1,052; the mean was 917, or 5.8–8.8
  decisions per hand.
- **Runtime:** 93.8 s wall time with 4 worker processes on 4 CPUs.
  Each process ran at 14.0 decisions/s while running in parallel, about
  half the single-process rate from Phase 2. The preflop-cache warm-up
  took up to 16.7 s per worker.
- **Projection (extrapolated, not measured):** about 44,000 rows for
  48 matches, about 13 min wall time with 4 workers.
- **Actions:**

  | Action | Rows | Share |
  |---|---|---|
  | check_call | 2,810 | 76.6% |
  | raise_half_pot | 420 | 11.4% |
  | raise_pot | 258 | 7.0% |
  | fold | 147 | 4.0% |
  | all_in | 34 | 0.9% |

  `all_in` is below 1% and is flagged as a rare class. Action mixes differ
  clearly by style: calling-station configurations chose check_call 95%
  of the time; maniacs raised 57% of the time; nits folded 14% of the
  time.
- **Features:**
  - No missing values in A or B.
  - All history rates lie within [0, 1].
  - At deal 0 the history features equal the priors, with zero counts.
  - Earlier-hand counts never decrease.
  - `a_legal_check_call` is constant: check or call is always legal.
  - By deals 40–59, average history features differ by style. For
    example, the facing-a-bet fold rate is 0.28 for nits and 0.01 for
    calling stations, against a prior of 0.15. This is descriptive only,
    not a model result.
- **Audits:** L1–L11 all passed on the pilot data:
  - 3,669 rows replayed with fake hole cards gave 0 mismatches;
  - the player role matched in 100% of rows;
  - all 52 seeds are distinct;
  - a regenerated match was byte-identical (223,957 bytes).

### Tests executed (Stage 1)

Local, Linux, Python 3.12.3: `python -m pytest` gave
`279 passed in 101.36s` (183 from earlier phases and 96 new).

### Known problems and limitations (Stage 1)

- **Held-out configurations are close to their presets.** For example,
  calling_station_v3 differs from calling_station only by bluff rate
  0.027. The held-out evaluation therefore tests generalization to near
  variants, not to new styles.
- **Rare classes.** `all_in` is 0.9% of rows overall, and calling
  stations never chose `raise_pot` or `all_in`. Macro-F1 will be noisy on
  rare classes, and accuracy alone would be misleading: always predicting
  check_call scores 77%.
- **Family balance.** Maniac matches have shorter hands, so maniacs
  produce fewer rows (296 of 3,669 in the pilot).
- **Sampled audits.** L3 truncation checks a sample of 150 rows, and L4/L5
  check sampled deals per match. L2 and L6–L10 cover every row.
- **Parallel speed.** Throughput per process roughly halves when 4
  processes share 4 CPUs.

### Next milestone (Stage 1)

The project owner approved the pilot on 2026-11-12 and authorized Stage 2.

### Stage 2: full dataset (2026-11-12)

Generated from the committed manifest with
`scripts/generate_dataset.py --workers 4`, then audited with
`scripts/dataset_report.py --regenerate 0 47`. The full report is
[`results/phase3/dataset_report.md`](../results/phase3/dataset_report.md).
Raw logs are under `data/raw/simulations/phase3_s3000/` and are not
committed. No strategies, features or classes were changed, nothing was
rebalanced, and no models were trained.

**Completeness and runtime (measured)**

- 48 of 48 runs have the `metadata.json` completion marker.
- Every run has 120 hands, the manifest seed and the manifest hash, and
  no leftover temporary files.
- Every run passed the loader's configuration checks.
- Wall time was 12.8 min with 4 workers for 43,856 decisions:
  57.1 decisions/s overall, 14.9 per process. Matches took 33–83 s each.
  The pilot projection had been about 44,000 rows and 13 min.

**Decision counts (measured)**

| Set | Matches | Rows used |
|---|---|---|
| train | 26 | 23,603 |
| validation | 5 | 4,252 |
| test | 5 | 4,895 |
| held-out (scored rows only) | 12 | 5,570 |
| **Total used** | 48 | **38,320** |

Another 5,536 rows are the training-pool counterparts' decisions in
held-out matches. They are generated but never trained on or scored.

By style family (rows used):

| Family | Rows |
|---|---|
| calling_station | 11,386 |
| solid | 9,847 |
| maniac | 8,777 |
| nit | 8,310 |

**Split separation**

- No match is in more than one split.
- No held-out configuration has rows in train, validation or test.
- Only the 4 held-out configurations are scored in the held-out set.
- Every validation and test configuration also appears in train.
- No deal seed is shared between matches.

**Action classes (rows used)**

| Action | Rows | Share |
|---|---|---|
| check_call | 27,696 | 72.3% |
| raise_half_pot | 4,724 | 12.3% |
| raise_pot | 3,930 | 10.3% |
| fold | 1,612 | 4.2% |
| all_in | 358 | 0.9% |

By style family:

- **nit:** fold 14.2%, never all-in.
- **calling_station:** check_call 97.1%; it never chose raise_pot or
  all-in, and folded twice in total.
- **maniac:** raised 59% of the time (half-pot 28.0%, pot 28.9%,
  all-in 2.0%).
- **solid:** in between, with 1.9% all-in.

**Conditioned on legal availability**

- **fold** was legal in 36.7% of decisions (only when facing a bet) and
  was chosen in 11.5% of those. By family: nit 35.1%, solid 10.8%,
  maniac 2.1%, calling_station 0.0%.
- **all_in** was legal in 98.3% of decisions and was chosen in 1.0% of
  those. By family: solid 1.9%, maniac 2.0%, nit and calling_station 0.
- check_call was always legal, so `a_legal_check_call` is constant.

**Rare classes and imbalance across sets**

- all_in is the only class below 1% overall. It is below 1% within
  train (0.9%) and validation (0.4%, only 19 rows). Test has 69 all_in
  rows and the held-out set 57.
- Class shares differ from train by at most 6.3 percentage points
  (validation check_call +6.3 pp; test +4.7 pp; held-out +2.7 pp). Only
  validation crosses the 5 pp descriptive flag that was set before the
  data was examined.
- The cause is which configurations landed in the 5 validation matches:
  only 366 solid rows and 756 maniac rows, against 1,612 calling_station
  rows. Shares also vary widely between individual matches; for
  example, check_call ranges from 45.6% to 100% across train matches.

**Audits on the full dataset:** L1–L11 all passed.

- L2, L6–L10 and L11a cover every row:
  - 43,856 rows replayed with fake hole cards gave 0 mismatches;
  - the player role matched in 100% of rows;
  - the priors were fitted on 23,603 train rows only.
- L3 checked 300 rows from 6 matches covering every split.
- L4 and L5 checked 282 and 240 sampled deals.
- L11b: matches 0 (validation) and 47 (held-out) were regenerated, and
  both their `decisions.csv` and `hands.csv` are byte-identical.

**Tests:**

- Local, Python 3.12.3: 285 passed.
- GitHub Actions on commit `9b860f1`: Ubuntu (Python 3.12.16) 285 passed
  in 118.22 s; Windows (Python 3.12.10) 285 passed in 115.57 s.

### Problems found in Stage 2

- **Mixed commits recorded during generation (no effect on the data).**
  The report code was committed while generation was running. Matches
  written before that commit recorded `8177a6b` (7 matches); later ones
  recorded `9b860f1` (41). The commits differ only by the new
  `reporting.py`, which generation does not import. `git diff` confirms
  the generation, agent, simulator and logging code is identical.
  `generate_dataset.py` now also records the commit and whether there
  were uncommitted changes at the **start** of a run.
- **Broken Markdown tables.** The column name `chosen | legal` broke the
  report's tables. `md_table` now escapes pipe characters (tested), and
  the report was regenerated.

### Implications for Stage 3 (for decision before training)

- **Validation is the most imbalanced set.** It has only 19 all_in rows.
  Validation macro-F1, the approved tuning metric, gives all_in one fifth
  of its weight, so model selection may be swayed by a handful of rows.
- **Per-style test results are thin.** Test has 352 maniac rows (one
  configuration) and 450 nit rows (one configuration). The held-out set
  is more even (1,206–1,655 rows per family, one configuration each).
- **Accuracy is easy to inflate.** Predicting check_call every time
  scores 72% on rows used.

### Next milestone (Stage 2)

The project owner approved Stage 2 on 2026-11-12 and authorized Stage 3.
Validation macro-F1 stays the selection metric, and per-class precision,
recall and F1 are reported.

### Stage 3: model training and evaluation (2026-11-12)

Full results are in
[`results/phase3/model_results.md`](../results/phase3/model_results.md),
with figures in `docs/figures/phase3/`. Trained models
(`models/phase3/`) and predictions (`data/processed/phase3/`) are kept
locally and not committed.

**Completed work**

- `opponent_model.py`:
  - feature matrices;
  - a train-only preprocessor that fixes the column order, drops
    columns that are constant in training data (only
    `a_legal_check_call`) and rejects missing values;
  - seeded models;
  - probabilities always in the fixed five-class order.
- `evaluation.py`:
  - accuracy, macro-F1, balanced accuracy, log-loss and per-class
    precision/recall/F1, all checked against scikit-learn;
  - match-level and paired bootstrap.
- `scripts/train_opponent_models.py` ran the full approved grids (36
  Random Forest and 48 HistGradientBoosting settings, each on A and B):
  - fitted on train;
  - selected by validation macro-F1 of legal-masked predictions;
  - every selected model, refitted, gave bit-identical predictions;
  - shuffled-label controls trained with the same settings.
  It took 29.2 min.
- `scripts/evaluate_opponent_models.py` produced:
  - metrics on raw and legal-masked probabilities;
  - 2,000-resample match-level bootstrap;
  - paired comparisons;
  - per-style, per-street and history-length breakdowns;
  - permutation importance by feature group;
  - eight figures.
- `scripts/shuffled_control_spread.py`: a diagnostic described under
  unexpected results.
- matplotlib 3.11.2 added and pinned, with its dependencies.

**Engineering decisions (Stage 3)**

| Decision | Reason |
|---|---|
| Random Forest with `n_jobs=1` | With several threads, scikit-learn adds the trees' probabilities in thread-completion order, so a refit differed in the last digits. One thread makes results bit-reproducible. |
| HistGradientBoosting with `early_stopping=False` | Its internal early stopping holds out random rows, which would break match-level splitting. |
| A class missing from training would get probability 0 (then floored), never be dropped | All five classes are always scored; the task never changes silently. No selected model was missing any class. |
| Primary metrics on legal-masked probabilities; raw reported separately | Masked predictions are what an agent would use. Raw results show how much probability each model put on illegal actions. |
| Per-style comparison uses log-loss and balanced accuracy, not macro-F1 | Macro-F1 averages over all five classes, so styles that never use some actions are capped. |
| No rebalancing of the data | The approved grid's `class_weight` option was the only reweighting, and it was chosen by validation. |

**Results (legal-masked; 95% match-level bootstrap intervals)**

| Model | Test macro-F1 (5 matches) | Held-out macro-F1 (12 matches) | Test log-loss | Held-out log-loss |
|---|---|---|---|---|
| Dummy (prior) | 0.172 [0.159, 0.180] | 0.168 [0.149, 0.183] | 0.814 | 0.867 |
| Random Forest A | 0.252 [0.228, 0.269] | 0.257 [0.235, 0.271] | 1.035 | 1.033 |
| **Random Forest B** | **0.361 [0.315, 0.386]** | **0.372 [0.310, 0.413]** | 0.832 | 0.824 |
| HistGradientBoosting A | 0.241 [0.205, 0.271] | 0.258 [0.222, 0.284] | 1.282 | 1.249 |
| **HistGradientBoosting B** | **0.366 [0.337, 0.383]** | **0.389 [0.325, 0.435]** | 0.892 | 0.886 |

- **Every trained model beats the Dummy on macro-F1 and balanced
  accuracy.** All paired intervals exclude 0.
- **No model has better accuracy than the Dummy.** Always predicting
  check_call scores 0.75 (test) and 0.72 (held-out).
- **Feature Set B beats A** on both sets:

  | Paired B − A | Test | Held-out |
  |---|---|---|
  | RF macro-F1 | +0.108 [+0.060, +0.149] | +0.114 [+0.070, +0.144] |
  | RF log-loss | −0.203 [−0.256, −0.152] | −0.209 [−0.284, −0.148] |
  | HGB macro-F1 | +0.125 [+0.074, +0.160] | +0.131 [+0.077, +0.182] |
  | HGB log-loss | −0.391 [−0.568, −0.169] | −0.363 [−0.599, −0.127] |

- **The gain is not a hyperparameter artifact.** The selected A and B
  models use different settings, so every grid setting was compared on
  validation with identical settings: B beat A in all 36 RF and all 48
  HGB settings. The macro-F1 gain ranged +0.10 to +0.20 and log-loss
  improved by 0.06 to 0.58.
- **Permutation importance (RF-B, held-out):**
  - Shuffling the opponent-history group costs 0.133 macro-F1 and adds
    0.50 to log-loss. It is the most important group, ahead of the
    current situation (0.066).
  - For RF-A, current-hand betting history (0.069) and situation (0.064)
    matter most.
- **History length:** RF-B's log-loss falls from 0.94 (0–9 earlier
  hands) to 0.81–0.83 (10+ hands). RF-A stays flat.
- **Per-class F1:**

  | Class | RF-B test | RF-B held-out | RF-A test | RF-A held-out |
  |---|---|---|---|---|
  | fold | 0.49 | 0.39 | 0.15 | 0.14 |
  | raise_pot | 0.28 | 0.39 | 0.18 | 0.21 |
  | all_in | 0.00 | 0.02 | 0.02 | 0.00 |

  all_in is essentially not predicted. HGB does best on all_in, with
  F1 0.11 (test) and 0.05 (held-out).
- **Per style (held-out log-loss, Dummy → RF-B):**

  | Style | Dummy | RF-B |
  |---|---|---|
  | calling_station | 0.359 | 0.197 |
  | maniac | 1.406 | 1.308 |
  | nit | 0.947 | 0.891 |
  | solid | 0.893 | 1.054 (worse than Dummy) |

**Unexpected results and explanations**

1. **Selected models are poorly calibrated.** Their log-loss is no better
   than the Dummy's: RF-B vs Dummy is +0.019 [−0.069, +0.095] on test and
   −0.043 [−0.137, +0.048] on held-out. The A models are worse than the
   Dummy. Validation macro-F1 favours deep forests and class-balanced
   weights, which give overconfident or distorted probabilities:
   - RF-B train log-loss is 0.31, against 0.68 on validation;
   - shuffling feature groups in RF-A *improves* its log-loss, a sign of
     overconfidence.

   The same grid contains much better-calibrated settings. The lowest
   validation log-loss is 0.521 for RF-B and 0.527 for HGB-B, against
   0.786 for the Dummy, at macro-F1 0.38–0.40. That is a validation-only
   observation; those settings were not evaluated on test data.
2. **Shuffled-label controls score slightly above the Dummy on
   macro-F1.** This is chance, not leakage:
   - The Dummy always predicts check_call, so any model that sometimes
     guesses other legal actions earns some minority-class F1.
   - A random legal guess drawn from the prior scores 0.204–0.205.
   - Every shuffled control has much worse log-loss than the Dummy.
   - HGB-B's control reached 0.221 on held-out. Retrained on five other
     shuffles it ranged 0.166–0.206 held-out and 0.172–0.190 test, with
     log-loss 1.21–1.27.
3. **Lower accuracy than the Dummy** is the expected price of predicting
   minority actions; it is why macro-F1 and log-loss are the main
   metrics.
4. **all_in is not learned.** There are only 213 training rows, and the
   bots' all-in decisions depend on private equity (≥ 0.9) that public
   features cannot reveal.

**Limitations**

- **Test set.** It has only 5 matches, so its intervals are rough, and
  per-style test results rest on one configuration for maniac and nit.
  The 12 held-out matches give stronger evidence.
- **Held-out opponents are close variants** of the training styles.
- **Synthetic opponents** follow rules we wrote.
- **Overfitting.** RF-B's train macro-F1 is 0.86, against 0.45 on
  validation.

**Provenance**

- Training started at commit `c709e4f` with the then-uncommitted
  training code, which was committed as `778aa77` a few minutes later.
  `git diff` confirms the training code is unchanged since `778aa77`.
- `train_opponent_models.py` now also records whether the working tree
  had uncommitted changes at start.

**Tests:** local, Python 3.12.3: 306 passed (285 earlier and 21 new).

### Next milestone

Stage 3 review by the project owner, before merging or starting Phase 4.
Phase 4 will feed predicted probabilities into the agent, so it needs a
decision on calibration: keep the macro-F1-selected models, or select by
validation log-loss for the agent.
