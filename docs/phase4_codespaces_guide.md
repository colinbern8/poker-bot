# Finishing the Phase 4D final evaluation in GitHub Codespaces

A step-by-step guide that works from a browser on a Chromebook or on Windows.

**Where the run stands now:**
- 4 of the 64 final matches are finished: all four agents against the `solid` opponent.
- They are backed up and validated on the branch `phase4-final-checkpoints`.
- This guide restores them in your own Codespace and plays the remaining 60 matches plus 8 short reproducibility replays.

**Nothing in the experiment changes:**
- You run the frozen code at tag `phase4-freeze` (commit `d50af2b`).
- You use the same model, seeds, agents and analysis.
- The runner refuses to start if anything differs.

**How long it takes:**

| Part | Time |
|---|---|
| Setup | about 15 minutes |
| Evaluation | about 4 hours on 4 cores |
| Your attention | needed only at the start and the end |

**Pasting commands.** Each grey box holds one command. Copy it, click in the terminal, and paste:
- **Chromebook:** `Ctrl+Shift+V`, or right-click → **Paste**.
- **Windows:** `Ctrl+V`, or right-click → **Paste**.

Then press **Enter**. Wait for each command to finish (the prompt `$` comes back) before the next one.

---

## Step 0: Two settings before you start (once)

### 1. Make the idle timeout as long as possible

A codespace stops after a period with no activity. The default is 30 minutes; the maximum is 240 minutes (4 hours).

1. On github.com, click your profile picture (top right) → **Settings**.
2. In the left sidebar, under "Code, planning, and automation", click **Codespaces**.
3. Under **Default idle timeout**, enter `240` and click **Save**.

This applies only to codespaces you create **after** saving, so do it before Step 1.

How the timer works:
- Typing, the mouse, and also **terminal output** reset it.
- Step 7 keeps a terminal printing progress, so the timer keeps resetting while the evaluation runs.
- Every codespace also has a hard limit of 12 hours. That is enough here.

### 2. Check your free quota

GitHub Pro includes 180 core-hours a month. A 4-core machine uses 4 core-hours per hour, so this run needs about 16–20 core-hours.

You can see what you have used under **Settings → Billing and licensing** (Codespaces usage).

---

## Step 1: Create a 4-core codespace

1. Open https://github.com/colinbern8/poker-bot.
2. Click the green **Code** button → **Codespaces** tab.
3. Click **…** (three dots) → **New with options…**
4. Choose:
   - **Branch:** `phase-4-ev-agents`
   - **Machine type:** **4-core**
5. Click **Create codespace**.

The browser editor opens. The repository's setup runs automatically and takes a few minutes. It installs Python 3.12, creates `.venv` and installs the pinned libraries. Wait until the terminal at the bottom says the setup ("postCreateCommand") has finished.

Open a fresh terminal with **Terminal → New Terminal** (or the `+` in the terminal panel).

**Check that you have 4 CPU cores:**

```bash
nproc
```

It must print `4`. If it prints `2`, change the machine type:
1. Go to https://github.com/codespaces.
2. Click **…** next to this codespace → **Change machine type** → **4-core** → **Update codespace**.
3. Reopen the codespace and run `nproc` again.

---

## Step 2: Switch to the frozen commit

```bash
cd /workspaces/poker-bot
```

```bash
git fetch origin --tags
```

```bash
git checkout phase4-freeze
```

Git prints a message about a **"detached HEAD"**. That is expected: you are now on the exact frozen commit, not on a branch.

**Check it.** This must print `d50af2b04e00a58a2fe5104ae59da8f2f2ea7669`:

```bash
git rev-parse HEAD
```

**This must print nothing** (no changed files):

```bash
git status --short --untracked-files=no
```

---

## Step 3: Check Python and the libraries

```bash
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. Check the version; it must print `Python 3.12.x`:

```bash
python --version
```

**Re-confirm the pinned libraries.** The setup already installed them, so this mostly prints "Requirement already satisfied":

```bash
python -m pip install -r requirements.txt -e .
```

---

## Step 4: Restore the trained model and check its SHA-256

The model file is not stored in git. Choose **one** of the two ways.

### A. Upload the model file (fastest, about 1 minute)

1. Download the file `random_forest_logloss_B.joblib` (21 MB) that Claude sent you in the chat.
2. In the codespace terminal, create the folder:

   ```bash
   mkdir -p models/phase3
   ```

3. In the **Explorer** panel on the left:
   - expand `models`;
   - right-click the `phase3` folder → **Upload…**;
   - choose the downloaded file.

   If you don't see `models`, click the refresh icon at the top of the Explorer. Dragging the file onto the `phase3` folder also works.

4. Verify it:

   ```bash
   python scripts/restore_phase4_model.py
   ```

### B. Rebuild it from scratch (about 20–25 minutes, no download)

This regenerates the training data (deterministic) and refits only the selected model:

```bash
python scripts/restore_phase4_model.py --rebuild --workers 4
```

### What to expect

Either way, the last line must say:

```
models/phase3/random_forest_logloss_B.joblib: SHA-256 matches the frozen model (5d4c8235e142a2fdd23a2b370def8183aabfad72b7998120cbe4fb711dda7b31).
```

If it says **does NOT match**, stop and ask Claude. Do not continue.

---

## Step 5: Retrieve and validate the four completed matches

**Get the backup branch** as a second folder next to the repository. It does not change your checkout.

```bash
git fetch origin phase4-final-checkpoints
```

```bash
git worktree add /workspaces/phase4-checkpoints phase4-final-checkpoints
```

**Copy the 4 matches into place.** Each file's SHA-256 is checked against the backup index, and its contents against the frozen manifest:

```bash
python /workspaces/phase4-checkpoints/backup_checkpoints.py --repo /workspaces/poker-bot --restore
```

Expected: four lines starting `restored and verified: jobs/…` (`ev_global`, `ev_ml`, `ev_situation`, `probability`, each `vs solid, 1000 hands`).

**Check them once more:**

```bash
python /workspaces/phase4-checkpoints/backup_checkpoints.py --repo /workspaces/poker-bot --validate
```

Expected: four `OK` lines and `4/64 matches and 0/8 replays on disk, 0 invalid`.

---

## Step 6: Confirm the runner will skip the four completed matches

```bash
python -c "from hybridpoker.phase4_final import FINAL_DIR, job_path, jobs_from_manifest, load_manifest; main, replays = jobs_from_manifest(load_manifest()); done = [j for j in main if job_path(FINAL_DIR, j[0], j[1].name).exists()]; print(f'{len(done)} of {len(main)} matches already complete; the runner will play {len(main) - len(done)} matches plus {len(replays)} short replays')"
```

Expected:

```
4 of 64 matches already complete; the runner will play 60 matches plus 8 short replays
```

**Run the full preflight check:**

```bash
python scripts/run_phase4_final.py --check
```

All five lines must start with `OK`:
- manifest and frozen files;
- environment;
- model;
- hero configurations;
- git tag and clean tree.

If any line says `FAIL`, stop and ask Claude.

---

## Step 7: Start the evaluation and the automatic backups

### 1. Start the evaluation in the background, with 4 workers

```bash
nohup python scripts/run_phase4_final.py --run --workers 4 >> /workspaces/final_run.log 2>&1 &
```

After a few seconds, check that it started:

```bash
tail -n 3 /workspaces/final_run.log
```

You should see `OK   git tag and clean tree` and then `68 of 72 matches to play (4 done)`.

### 2. Start the automatic backups

These save every newly finished match to GitHub every 15 minutes. They stop by themselves when everything, including the final report, is saved. They run inside your codespace and do not need Claude.

```bash
cd /workspaces/phase4-checkpoints && nohup python backup_checkpoints.py --repo /workspaces/poker-bot --every 15 >> /workspaces/backup.log 2>&1 & cd /workspaces/poker-bot
```

Check it:

```bash
tail -n 2 /workspaces/backup.log
```

It should say `nothing new: 4/64 matches…` (the first new match finishes after 15–25 minutes).

### 3. Keep the codespace active

Open a **second terminal** (`+` in the terminal panel) and run:

```bash
tail -f /workspaces/final_run.log
```

It prints a line each time a match finishes. That output keeps resetting the idle timer.

You can leave the browser tab open and walk away. Closing the tab does not stop the codespace by itself. If your computer sleeps and the codespace times out anyway, nothing finished is lost; see Step 9.

The first results appear after about 15–25 minutes. The full run takes about 4 hours.

---

## Step 8: Check progress safely (any time)

These commands only **read**, so they are safe while the run is going. Open the codespace from https://github.com/codespaces if it is not open, then in a terminal:

```bash
cd /workspaces/poker-bot && source .venv/bin/activate
```

**Is the evaluation still running?**

```bash
pgrep -f "run_phase4_final.py --run" > /dev/null && echo "evaluation RUNNING" || echo "evaluation NOT running"
```

**How many of the 64 matches are finished:**

```bash
ls data/processed/phase4_final/jobs | wc -l
```

**The last few finished matches:**

```bash
tail -n 5 /workspaces/final_run.log
```

**The last backup:**

```bash
tail -n 2 /workspaces/backup.log
```

You can also check from any device without opening the codespace. Open branch `phase4-final-checkpoints` on GitHub → `checkpoints/INDEX.json`; the `"complete"` field shows how many matches are backed up.

**Do not** start a second evaluation while one is running. Always check with the `pgrep` command first.

---

## Step 9: Resume after an interruption

**If the codespace stopped** (idle timeout, 12-hour limit, or you stopped it):
- running programs end;
- the files in `/workspaces` are kept, including every finished match.

Reopen the codespace from https://github.com/codespaces, open a terminal, and run these in order.

**1. Go to the repository and activate Python:**

```bash
cd /workspaces/poker-bot && source .venv/bin/activate
```

**2. Make sure the evaluation is not already running.** It must print `not running`:

```bash
pgrep -f "run_phase4_final.py --run" > /dev/null && echo "running: do not restart" || echo "not running"
```

**3. Confirm you are still on the frozen commit.** It must print `d50af2b04e00a58a2fe5104ae59da8f2f2ea7669`:

```bash
git rev-parse HEAD
```

**4. Validate the saved matches.** Every line must be `OK`:

```bash
python /workspaces/phase4-checkpoints/backup_checkpoints.py --repo /workspaces/poker-bot --validate
```

**5. Run the preflight check.** All five lines must be `OK`:

```bash
python scripts/run_phase4_final.py --check
```

**6. Resume the evaluation.** It skips every finished match and replays only the unfinished ones, with identical seeds:

```bash
nohup python scripts/run_phase4_final.py --run --workers 4 >> /workspaces/final_run.log 2>&1 &
```

**7. Restart the automatic backups:**

```bash
cd /workspaces/phase4-checkpoints && nohup python backup_checkpoints.py --repo /workspaces/poker-bot --every 15 >> /workspaces/backup.log 2>&1 & cd /workspaces/poker-bot
```

**8. Keep a progress terminal open** (as in Step 7):

```bash
tail -f /workspaces/final_run.log
```

### Other problems

- **The codespace was deleted.** Create a new one and repeat Steps 1–6. Step 5 restores every match that was backed up, then continue with Step 7.
- **Validation reports `INVALID` for a file.** This is very unlikely, because files are written in one step. Delete only that file. The runner will play that match again:

  ```bash
  rm data/processed/phase4_final/jobs/NAME.json.gz
  ```

- **A backup push fails.** The backup is still committed in `/workspaces/phase4-checkpoints` and is pushed at the next attempt. `tail /workspaces/backup.log` shows the message.

---

## Step 10: When it finishes

The run is finished when the evaluation log ends with `wrote results/phase4/stage4d_final_report.md`:

```bash
tail -n 3 /workspaces/final_run.log
```

At its next run (within 15 minutes) the backup:
- saves the last matches, the replays, the final report and the run log to `phase4-final-checkpoints`;
- prints `All matches, replays and the final report are backed up. Stopping.`

Then:

1. Check it with `tail -n 2 /workspaces/backup.log`.
2. Tell Claude that the run is finished. Claude reads everything from the backup branch, runs the pre-registered dependence sensitivity check, and reports the results.
3. **Do not** read the results and change anything. **Do not** rerun the evaluation.
4. Once Claude has confirmed the backup, stop the codespace to save quota. At https://github.com/codespaces, click **…** → **Stop codespace**.

   Keep it, rather than delete it, until the results are written up.

**Do not** commit files to `phase-4-ev-agents` or `main` from the codespace. The generated data belongs only on the backup branch, and must never be merged into `main`.

---

## Notes

- **Reproducibility.** Each match is fully determined by the frozen code and its seed. The 8 replays at the end re-play the start of some matches in fresh processes and check that they match exactly.
- **Different machines.** The first 4 matches were played in the earlier cloud environment (2026-11-13, 01:08–01:32 UTC), and the rest are played in your codespace. Each match is played entirely on one machine, so the comparison between agents is unaffected. Only decision-time (latency) figures come from two kinds of hardware; the report will note this.
- **What the runner checks before starting.** It refuses to play unless all of these match the frozen manifest:
  - the code and data file hashes;
  - the Python and library versions;
  - the model's SHA-256;
  - each agent's settings;
  - the `phase4-freeze` tag on a clean checkout.
