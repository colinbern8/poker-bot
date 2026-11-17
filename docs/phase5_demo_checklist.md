# Presentation-Machine Checklist (Phase 5B demo)

Use this on the computer that will run the demonstration, **at least two
days before** the presentation (December 7–9, 2026), and again on the day.
Tick every box. Commands run from the repository folder.

## 1. Install (once)

- [ ] Python **3.12** is installed: `python --version` (3.13 is not supported).
- [ ] Get the code (latest `main`) and create the virtual environment:

  Linux / macOS:
  ```bash
  python3.12 -m venv .venv && source .venv/bin/activate
  ```
  Windows (PowerShell):
  ```powershell
  py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
  ```
- [ ] Install the exact, pinned versions:
  ```bash
  python -m pip install -r requirements.txt -e .
  ```

## 2. Restore the verified frozen model

The model file (about 21 MB) is not in git. Copy it in with a hash check:

```bash
python scripts/restore_phase4_model.py --from /path/to/random_forest_logloss_B.joblib
```

- [ ] The last line says **"SHA-256 matches the frozen model (5d4c8235…7b31)"**.
  If it says *does NOT match*, stop: the demo refuses to start with a wrong file
  and never switches to another agent.
- [ ] Keep a second copy of the file on a USB stick.

## 3. Run the tests

- [ ] Full suite (about 4–6 minutes):
  ```bash
  python -m pytest
  ```
  Expected: no failures. Without `HYBRIDPOKER_REAL_MODEL=1` the opt-in
  acceptance tests are reported as skipped.
- [ ] Real-model acceptance tests (the engine and the screen, with the real model):
  ```bash
  HYBRIDPOKER_REAL_MODEL=1 python -m pytest tests/acceptance -v -s
  ```
  Windows (PowerShell):
  ```powershell
  $env:HYBRIDPOKER_REAL_MODEL="1"; python -m pytest tests/acceptance -v -s
  ```
- [ ] All acceptance tests **pass**. Write down the printed numbers:
  - model load + SHA-256 check: ______ s
  - cache warm-up: ______ s
  - AI decision time: median ______ ms, p95 ______ ms, max ______ ms
- [ ] For a reliable 95th percentile, also measure about 50 or more AI decisions:
  ```bash
  HYBRIDPOKER_REAL_MODEL=1 HYBRIDPOKER_ACCEPTANCE_HANDS=20 python -m pytest tests/acceptance/test_real_model_app.py -v -s
  ```
  Each line says whether the targets were met: p95 decision time under 2 s,
  startup under 90 s. These are **targets, not test failures**. If they are
  missed, write down the numbers and follow the speed advice in step 4.

## 4. Launch and check startup and speed

```bash
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in the browser. The demo listens only on
this computer, never asks for an email, and sends no usage statistics (see
`.streamlit/config.toml`).

- [ ] The loading spinner appears, then the welcome page.
- [ ] The sidebar shows "SHA-256 verified" and the load and warm-up times.
  Warm-up took 15–40 s on test machines, and happens once per start.
- [ ] Play 10 hands. Each AI move should take about 1–2 s; Phase 4 measured
  0.65 s median and 1.4 s 95th percentile. If it is much slower, close other
  programs and plug in the charger.
- [ ] Check the explanation panel after a hand, and the AI's cards at a
  showdown.
- [ ] Do not edit project files while the app is running: Streamlit notices
  source changes and can pause the page.

## 5. Offline operation

- [ ] Turn off Wi-Fi, restart the app, and play 3 hands. Everything must
  work: the app needs no network.
- [ ] Turn off sleep and screen-saver; plug in the charger.
- [ ] Set the browser zoom so the table and the buttons fit on the projector
  (try 110–125%).

## 6. Fixed-seed rehearsal

A rehearsal seed deals the same cards every time. With the same clicks, the
AI makes the same decisions (the agent and the deal are fully seeded).

- [ ] In the sidebar choose **Rehearsal seed**, type a number (for example
  `2026`), and click **Start new match**. The seed is cleared from the box
  and never shown during play.
- [ ] Write down a short script: the clicks for 3–4 hands that show a fold, a
  call, a showdown, and an explanation with a coverage warning.
- [ ] Rehearse it twice and check that the cards and AI actions are the same.
- [ ] For the live audience, use **Random deal** unless the rehearsed story
  is needed.

## 7. On the day

- [ ] Start the app **5 minutes early** (warm-up), open the page, and click
  **Start match** once to check.
- [ ] Explain that the AI's cards stay hidden until a showdown, and that its
  reasoning appears after each hand.
- [ ] Say clearly that demo games are not research evidence. The evaluation
  result is in `results/phase4/stage4d_final_report.md`.
- [ ] Expect the AI to go all-in often; it did so in 23% of its decisions in
  the evaluation.

## 8. Fallback procedure

| Problem | What to do |
|---|---|
| Page shows "model file is missing" or "does not match" | Restore the model from the USB copy (step 2) and reload the page |
| "Something went wrong" message during a hand | The game is unchanged: click the action again, or **Start new match**. Afterwards, find the message's reference code in the terminal window: the full error is logged there |
| Page frozen or browser closed | Reload the page (this starts a new match), or stop the app with `Ctrl+C` and start it again |
| The laptop fails | Use the second machine prepared with this checklist |
| No working machine | Show the screen recording and screenshots of the rehearsed hands (record them during rehearsal), then present the Phase 4 results |

- [ ] Record a 2–3 minute screen video of the rehearsed hands, and keep
  screenshots of the table, a showdown and an explanation.
