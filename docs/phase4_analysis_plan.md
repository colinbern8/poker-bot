# Phase 4 Final Evaluation: Analysis Plan (Stage 4D)

Status: **freeze candidate, written before any final hand is played.**

- It is approved for content (project owner, 2026-11-13). The freeze itself
  is still pending approval.
- The machine-readable version is in
  `data/manifests/phase4_eval_manifest.json` (field `analysis`).
- The code that carries it out is `src/hybridpoker/phase4_final.py` and
  `scripts/run_phase4_final.py`.
- If this text and the manifest ever disagree, the manifest is what was
  frozen.

## 1. Question and hypothesis

> Can supervised machine-learning-based opponent modeling improve
> decisions made by a probability-based poker agent?

The final evaluation tests this with agents that share one decision rule
(the one-step EV rule) and differ only in their opponent model.

- **Null hypothesis (H0):** the ML EV Agent and the Situation-Aware Frequency
  EV Agent win at the same rate, averaged with equal weight over the 16
  opponents.
- **Alternative (H1):** the rates differ (two-sided).

## 2. Design (fixed in the manifest)

| Item | Value |
|---|---|
| Heroes | ProbabilityAgent, Global Frequency EV, Situation-Aware Frequency EV, ML EV, with exact configurations recorded |
| Opponents | All 16 configurations: 12 training-pool, 4 held-out |
| Matches | One per hero and opponent: 64 matches, plus 8 short reproducibility replays |
| Deals per match | 500, each played twice with seats swapped (1,000 hands) |
| Total | 64,000 hands |
| Seeds | Master seed 4000, never used before. Match seed `derive_seed(4000, "eval", opponent, 0)`, shared by all four heroes |
| Shared deals | All heroes get identical cards and seats against the same opponent. The opponent's random choices use the same seeds. |
| Memory | Every hero starts each match with no memory. The ML agent learns the opponent's public actions during the match only. |
| Model | Random Forest B #22, SHA-256 `5d4c8235…7b31` |
| Code | SHA-256 of every source file and training artifact in the manifest, plus git tag `phase4-freeze` |

**Seed safety.** Tests confirm that neither the final match seeds nor
their 8,000 deal seeds overlap:

- the Phase 2 seeds;
- the Phase 3 dataset and pilot seeds;
- any development seed from master seed 5000.

## 3. Metric

- **Unit: the deal pair**, the two hands of one deal with the seats
  swapped. The hero's net chips over both hands are converted to bb/100:
  `chips ÷ 2 hands ÷ 2 chips per big blind × 100`.
- **Why the deal pair:** the two hands share the same cards, so they are not
  independent. Different deals are independent.
- **The development data support this:**
  - the lag-1 autocorrelation of paired differences was close to 0;
  - standard errors from batches of 10 deals matched the plain ones;
  - a stratified bootstrap gave nearly the same intervals as the normal
    approximation.

**Paired difference.** For two heroes A and B against the same opponent,
`d_i = A_i − B_i` for each deal pair `i`. Both heroes played exactly the
same cards, so most card luck cancels.

## 4. Primary endpoint

**ML EV Agent minus Situation-Aware Frequency EV Agent**, in bb/100.

1. **For each opponent `o`:** compute the mean of `d_i` over its 500 deal
   pairs, `m_o`, and its standard error, `SE_o = SD(d_i) / √500`.
2. **Estimate:** the equal-weight average over the 16 opponents,
   `M = (1/16) Σ m_o`.
3. **95% interval:** `M ± 1.96 × √(Σ SE_o²) / 16`. The opponents are played
   with independent seeds, so their errors add in quadrature.
4. **Decision (two-sided, α = 0.05):**
   - if the interval is entirely above 0, the ML agent won more;
   - if it is entirely below 0, it won less;
   - otherwise, no difference was detected, and the interval shows the range
     of effects compatible with the data.

   The report's wording comes from this rule automatically
   (`phase4_final.conclusion`).
5. **Precision expected from the development run.** The development run
   measured a per-pair standard deviation of about 1,460 bb/100. With 500
   pairs:
   - the expected half-width is about ±32 bb/100;
   - the smallest true effect detected with 80% power is about 46 bb/100.

   The development point estimate (+61) is **not** used for anything else.

**Why equal weight?** Each opponent configuration counts once, so a
high-variance opponent cannot dominate the result. This was fixed before
any final data existed.

## 5. Secondary endpoints (pre-specified)

Both use the same estimator and 95% interval:

1. ML EV minus Global Frequency EV, against the simpler control;
2. ML EV minus ProbabilityAgent, against the original mathematical agent.

They are **not adjusted for multiplicity** and are reported as supporting
evidence. The conclusion about the research question rests on the
primary endpoint.

## 6. Opponent groups and individual opponents

- **Subgroups (exploratory):**
  - training-pool (12) vs held-out (4) opponents;
  - each style (calling station, maniac, nit, solid), 4 opponents each.

  The estimator is the same, equal weight within the group. Subgroup
  intervals are reported but not used for confirmatory claims. Held-out
  results are of particular interest because the model never saw those
  opponents.
- **Individual opponents:** descriptive only. With 16 opponents, about one
  interval would exclude 0 by chance even if nothing differed.
- **Development findings to check, not to confirm.** The development run
  suggested that the ML agent's gains come from solid and nit opponents,
  where it shoves less. The final report shows whether this pattern
  repeats. It is presented as descriptive.

## 7. Descriptive results and diagnostics

These come from the same tables as the development report
(`hybridpoker.phase4_report`):

- win rates per hero and opponent;
- all six pairwise comparisons;
- all-in rates (overall, per street, per style);
- net results of hands containing a hero shove;
- reliance on poorly covered hypothetical states (fewer than 30 training
  rows);
- median and 95th-percentile decision latency.

## 8. Sensitivity analysis

A stratified bootstrap of the primary endpoint:

- deal pairs are resampled within each opponent;
- 10,000 replicates, with seed `derive_seed(4000, "bootstrap", "primary")`;
- the result is reported next to the normal interval.

The decision uses the normal interval. A large disagreement between the
two would be reported as a limitation.

## 9. Validity checks (reported before any result is read)

1. All 64 matches finished with no error or illegal action. Each has 1,000
   hands.
2. Every hero received identical cards against each opponent.
3. The 8 replays (first 6 deals, re-played in fresh processes) reproduce
   their matches exactly.
4. The model hash, file hashes, library versions and hero configurations
   matched the manifest, and the checkout carried the `phase4-freeze` tag.
   `scripts/run_phase4_final.py` refuses to play otherwise.

## 10. Problems during the run

- **Interruption** (for example a restarted machine). Each finished match
  is saved immediately. Re-running the same command plays only the
  missing matches. Matches are fully seeded, so the results are identical
  to an uninterrupted run.
- **A match error or illegal action** makes the run **invalid**. No
  primary result is reported from an incomplete run (enforced in code).
  - The error is reported.
  - Any fix is a documented protocol deviation and needs a new freeze.
  - Fresh seeds are needed only if the bug could have been influenced by
    seeing final results.

## 11. What will not be done

- No change to agents, model, opponents, seeds, sample size or analysis
  after the freeze.
- No re-running with other seeds to obtain a different result, and no
  dropping or re-weighting of opponents.
- No tuning of anything on final results, now or later in the project.
  Later phases (for example the demo) may change code. The final results
  stay tied to the frozen commit.

## 12. Limitations to report with the results

1. **Excessive all-in play and no future-betting reasoning.**
   - The one-step EV rule makes all-in the best bet whenever the hero is
     ahead of the estimated calling range.
   - In development, the EV agents chose all-in in 22–31% of decisions.
     Selective opponents exploited this.
   - This was accepted for the competition and applies equally to all
     three EV agents.
2. **The continuing-range model is an approximation.** It ranks hands by
   current strength and ignores the earlier action.
3. **Poorly covered states.** In development, 67% of the ML agent's
   all-ins led to opponent states with fewer than 30 training rows.
4. **The Situation-Aware control is incomplete.** It conditions on the
   betting situation only, not on bet sizes, position, the board or
   opponent history. ML minus this control therefore still includes the
   value of those current-hand features as well as of opponent modeling.
5. **Synthetic opponents.** All opponents are rule-based bots. Held-out
   bots are close variants of the training styles. Results may not carry
   over to human players.
6. **Precision.** Precision assumes the development variance. The final
   intervals are what count.

## 13. Freeze procedure (Stage 4D, after approval)

1. The candidate commit contains this plan and the manifest. CI must pass
   on Ubuntu and Windows.
2. Confirm that `python scripts/build_phase4_manifest.py --check` reports
   *identical*.
3. Tag the commit `phase4-freeze` and push the tag.
4. Confirm that `python scripts/run_phase4_final.py --check` reports
   every check OK.
5. Run `python scripts/run_phase4_final.py --run --workers 4`: about 4.3 h
   on 4 cores, resumable.
6. Commit the generated report. Do not change the frozen commit.
