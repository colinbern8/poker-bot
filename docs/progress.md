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
models, metrics and leakage tests. No Phase 3 code or data exists yet.
The four-match pilot comes first and stops for review.
