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

Windows results come from GitHub Actions. They were not run locally.

### Experimental results

None yet. Phase 0 contains no experiments.

### Known problems

- GitHub Codespaces setup was validated for JSON syntax only. It has not been
  run in an actual Codespace yet.
- Windows compatibility depends on the GitHub Actions run.
- PokerKit card dealing is not yet seeded/reproducible.

### Next milestone

Phase 1, to be defined and approved before work starts.
