# Phase 4D final-evaluation checkpoint backup (generated data)

This orphan branch holds **only** backup copies of completed match checkpoints from the frozen
Stage 4D final evaluation (tag `phase4-freeze`, commit `d50af2b`, master seed 4000). It shares no
history with `main` and **must never be merged into `main` or any code branch**.

- `checkpoints/jobs/<hero>__<opponent>.json.gz`: one finished 500-deal match each (64 in total).
- `checkpoints/replays/...`: the 8 short reproducibility replays.
- `checkpoints/INDEX.json`: SHA-256, hero, opponent, hand count and result hash of every file.
- `backup_checkpoints.py`: validates each finished file against the frozen manifest
  (hero, opponent, seed, deal count, no error, complete hands, result hash) before copying.

To restore after losing the evaluation machine, copy `checkpoints/jobs` and `checkpoints/replays` to
`data/processed/phase4_final/` in a checkout of `phase4-freeze`, then rerun
`python scripts/run_phase4_final.py --run --workers 4`: completed matches are skipped and only the
missing ones are played, with identical results.
