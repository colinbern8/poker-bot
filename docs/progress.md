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
| `pot_size` includes the opponent's bet and excludes the hero's call; `call_amount > pot_size` is rejected | One consistent convention across the project. The check catches the most common mistake, leaving the bet out of the pot. |
| Exact enumeration limited to heads-up river | 990 combinations take about 0.3 s. Two opponents would need about 450,000, which takes minutes. |
| No decision rule (call/fold) in Phase 1 | CLAUDE.md rule 7 says to separate math from decision rules. Rules come in Phase 2. |
| No new dependencies | Only the standard library and PokerKit are used. |

### Tests executed

Local run (Linux, Python 3.12.3), `python -m pytest`: `82 passed in 12.83s`.

- `tests/test_equity.py` (43 tests):
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
  final pot, and no rake (see `docs/probability_engine.md`).
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
