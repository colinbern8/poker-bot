# Phase 4D final-evaluation checkpoint backup (generated data)

This orphan branch holds **only** backup copies of completed match checkpoints from the frozen
Stage 4D final evaluation (tag `phase4-freeze`, commit `d50af2b`, master seed 4000). It shares no
history with `main` and **must never be merged into `main` or any code branch**.

- `checkpoints/jobs/<hero>__<opponent>.json.gz`: one finished 500-deal match each (64 in total).
- `checkpoints/replays/...`: the 8 short reproducibility replays.
- `checkpoints/INDEX.json`: SHA-256, hero, opponent, hand count and result hash of every file.
- `backup_checkpoints.py`: validates each finished file against the frozen manifest
  (hero, opponent, seed, deal count, no error, complete hands, result hash) before copying.

Modes of `backup_checkpoints.py` (run from this worktree, for example `/workspaces/phase4-checkpoints`):

| Command | What it does |
|---|---|
| `python backup_checkpoints.py --repo /workspaces/poker-bot --restore` | Copies the backed-up matches into the repo after checking SHA-256 and the manifest |
| `python backup_checkpoints.py --repo /workspaces/poker-bot --validate` | Checks the matches on disk; copies nothing |
| `python backup_checkpoints.py --repo /workspaces/poker-bot` | One backup now (validated new matches, commit, push) |
| `python backup_checkpoints.py --repo /workspaces/poker-bot --every 15` | Backs up every 15 minutes; stops by itself once all 64 matches, 8 replays and the final report are saved |

The full step-by-step guide is `docs/phase4_codespaces_guide.md` on branch `claude/dazzling-knuth-r4bzf1`.
