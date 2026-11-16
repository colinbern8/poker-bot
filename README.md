# HybridPoker

An explainable Texas Hold'em poker AI for a university AI course project.

**Research question:** Can supervised machine learning-based opponent modeling
improve decisions made by a probability-based poker agent?

**Current status:** Phase 4 (EV agents and the final evaluation) is complete and
frozen. Phase 5A adds the tested game engine for playing against the Hybrid ML
Agent; the Streamlit screen comes next ([`docs/phase5_plan.md`](docs/phase5_plan.md)).
See [`docs/progress.md`](docs/progress.md) for progress,
[`docs/probability_engine.md`](docs/probability_engine.md) for the equity math, and
[`docs/simulator.md`](docs/simulator.md) for the simulator, agents, and statistics.

## Technology

- Python **3.12** (required; other versions are not supported)
- [PokerKit](https://github.com/uoftcprg/pokerkit) for poker rules and hand evaluation
- NumPy, pandas, scikit-learn, Streamlit
- pytest for testing

## Project layout

```
poker-bot/
├── src/hybridpoker/      # project source code (the `hybridpoker` package; frozen in Phase 4)
├── src/hybridpoker_demo/ # Phase 5: human-vs-AI game controller (outside the frozen package)
├── tests/                # automated tests (pytest); tests/acceptance/ is opt-in
├── scripts/              # helper scripts (benchmarks, matches)
├── docs/                 # progress log and explanations of the math
├── pyproject.toml        # project settings and dependency ranges
├── requirements.txt      # exact tested dependency versions
├── .devcontainer/        # GitHub Codespaces configuration
└── .github/workflows/    # automated tests on Ubuntu and Windows
```

## Setup

### Option A: GitHub Codespaces (recommended)

1. On GitHub, click **Code → Codespaces → Create codespace on this branch**.
2. Wait for the setup to finish. It automatically installs Python 3.12, creates
   a `.venv` virtual environment, and installs all dependencies.
3. Open a new terminal. If the prompt does not start with `(.venv)`, activate
   the virtual environment first:

   ```bash
   source .venv/bin/activate
   ```

4. Run the tests (see [Running tests](#running-tests)).

> **Already have a Codespace that shows Python 3.14 (or anything other than 3.12)?**
> It was created before this configuration existed. Open the Command Palette
> (`Ctrl+Shift+P` / `Cmd+Shift+P`) and run **Codespaces: Rebuild Container**,
> or delete it and create a new Codespace.

Check that it worked:

```bash
python --version
```

It should print `Python 3.12.x`.

### Option B: Linux or macOS

Install Python 3.12 first (from [python.org](https://www.python.org/downloads/)
or your package manager), then run from the repository folder:

```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements.txt -e .
```

### Option C: Windows (PowerShell)

Install Python 3.12 from [python.org](https://www.python.org/downloads/windows/)
(this also installs the `py` launcher), then run from the repository folder:

```powershell
py -3.12 -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt -e .
```

If activation fails with *"running scripts is disabled on this system"*, run
this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### What the install command does

`python -m pip install -r requirements.txt -e .` installs:

- every library at the **exact version** listed in `requirements.txt`, so
  everyone gets an identical, reproducible environment, and
- the `hybridpoker` package itself in **editable** mode (`-e .`), so changes
  you make in `src/hybridpoker/` take effect without reinstalling.

### Restoring the frozen Phase 4 opponent model

The ML EV Agent needs `models/phase3/random_forest_logloss_B.joblib`
(Random Forest B #22). Model files are kept out of git, so a fresh
Codespace does not have one. The agent loads it only if its SHA-256 equals
the approved hash, `5d4c8235…7b31`.

**Fastest (seconds):** keep a copy of the verified file somewhere outside
git, for example a private cloud drive or a GitHub release asset you
download by hand. Then copy it in with a hash check:

```bash
python scripts/restore_phase4_model.py --from /path/to/random_forest_logloss_B.joblib
```

The script refuses a file whose hash differs.

**Check the file already in place:**

```bash
python scripts/restore_phase4_model.py
```

**No copy available (about 20 minutes on 4 cores):** rebuild it. The Phase 3
dataset is deterministic, and the refit reproduces the same file byte for
byte with Python 3.12 and the pinned `requirements.txt` (verified on
2026-11-12).

```bash
python scripts/restore_phase4_model.py --rebuild
```

The rebuild regenerates `data/raw/simulations/phase3_s3000` (git-ignored),
checks that the committed training artifacts reproduce exactly, refits only
the selected model, and verifies the hash. It does not rerun any other
Phase 3 experiment.

## Running tests

With the virtual environment active:

```bash
python -m pytest
```

Expected result without the frozen model file (as in GitHub Actions):
`484 passed, 5 skipped` as of Phase 5A. The skipped tests are the two that need
the frozen model file (`tests/test_ev_agent.py` and `tests/test_phase4_final.py`)
and the three opt-in acceptance tests in `tests/acceptance/`. With the model in
place, the two model tests run as well. The full suite takes about 3–5 minutes.

The acceptance tests use the real frozen model and run only when asked (they
fail, rather than skip, if the model is missing or differs):

```bash
HYBRIDPOKER_REAL_MODEL=1 python -m pytest tests/acceptance -v -s
```

If you see `No module named pytest`, the virtual environment is not active.
Activate it (see [Setup](#setup)) and try again.

Useful variations:

```bash
python -m pytest -v
```

```bash
python -m pytest tests/test_smoke.py
```

Tests also run automatically on **Ubuntu and Windows** through GitHub Actions
on every push. Results appear as a ✓ or ✗ next to each commit on GitHub.

## Using the probability engine

```python
from hybridpoker.equity import estimate_equity
from hybridpoker.pot_odds import call_expected_value, pot_odds

# Ace-king on a T-9-2 flop against one random hand, 2000 simulations.
result = estimate_equity("AsKh", "Td9d2c", opponent_count=1, sample_count=2000, seed=42)
print(result.equity, result.standard_error)

# Facing a 10-chip bet into a 20-chip pot: pot_size = 30, call_amount = 10.
print(pot_odds(pot_size=30, call_amount=10))  # 0.25
print(call_expected_value(result.equity, pot_size=30, call_amount=10))
```

Measure engine speed on your machine:

```bash
python scripts/benchmark_equity.py
```

## Running matches

Play a seeded, seat-swapped match between the probability agent and a bot,
and write hand logs:

```bash
python scripts/run_match.py --villain calling_station --deals 500 --seed 1
```

Play the probability agent against every bot in turn:

```bash
python scripts/run_match.py --villain all --deals 300 --seed 1
```

Each deal is played twice with seats swapped, so `--deals 500` means 1,000
hands. Results are printed as bb/100 with a 95% confidence interval. Logs go to
`data/raw/simulations/<run>/` (`decisions.csv`, `hands.csv`,
`metadata.json`), which git ignores. Agents: `probability`, `random`,
`calling_station`, `maniac`, `nit`, `solid`.

## Playing against the Hybrid ML Agent (Phase 5A, no screen yet)

`hybridpoker_demo.HeadsUpController` runs a human-versus-AI match one action
at a time: 200-chip stacks every hand, blinds 1/2, the button alternating,
and the five abstract actions. It needs the frozen model (see
[Restoring the frozen Phase 4 opponent model](#restoring-the-frozen-phase-4-opponent-model)).

```python
from hybridpoker_demo.controller import HeadsUpController, HumanTurn
from hybridpoker_demo.resources import build_ml_agent, load_frozen_ml_model, warm_caches

warm_caches()                                     # once, about 15-40 s
model = load_frozen_ml_model()                    # SHA-256 checked
game = HeadsUpController(build_ml_agent(model))   # random seed; match_seed=... to rehearse
while isinstance(game.view, HumanTurn):
    turn = game.view
    print(turn.observation.hole_cards, turn.observation.board_cards, turn.legal_categories)
    game.submit(turn.ticket, turn.legal_categories[0])   # the human's choice
print(game.view.human_net, game.scoreboard)        # the AI's reasoning: game.view.ai_decisions
```

How it works and why it reproduces the frozen agent exactly:
[`docs/phase5_plan.md`](docs/phase5_plan.md).

## Updating dependencies

Only change dependency versions deliberately, because a new version can change
experimental results. To upgrade:

1. Update the version in `requirements.txt`. If it is a new library, also add it to `pyproject.toml`.
2. Reinstall:

   ```bash
   python -m pip install -r requirements.txt -e .
   ```

3. Run the tests:

   ```bash
   python -m pytest
   ```

4. Record the change in `docs/progress.md`.
