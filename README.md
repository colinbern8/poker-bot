# HybridPoker

An explainable Texas Hold'em poker AI for a university AI course project.

**Research question:** Can supervised machine learning-based opponent modeling
improve decisions made by a probability-based poker agent?

**Current status:** Phase 1, the probability engine (equity and pot odds).
See [`docs/progress.md`](docs/progress.md) for progress and
[`docs/probability_engine.md`](docs/probability_engine.md) for how the math works.

## Technology

- Python **3.12** (required; other versions are not supported)
- [PokerKit](https://github.com/uoftcprg/pokerkit) for poker rules and hand evaluation
- NumPy, pandas, scikit-learn, Streamlit
- pytest for testing

## Project layout

```
poker-bot/
├── src/hybridpoker/      # project source code (the `hybridpoker` package)
├── tests/                # automated tests (pytest)
├── scripts/              # helper scripts (e.g. benchmarks)
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

## Running tests

With the virtual environment active:

```bash
python -m pytest
```

Expected result: all tests pass (`82 passed` as of Phase 1).

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
