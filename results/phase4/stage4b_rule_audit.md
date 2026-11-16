# Stage 4B: rule audit on fixed validation-split spots

300 decision situations sampled (seeded) from the Phase 3 validation split, rebuilt from public history and board plus the deciding player's own cards. Each decided by four configurations with the same seed. Not a match; no opponent history for the ML agent (training priors). Caches warmed.

## Chosen actions

| configuration | spots | fold | check_call | raise_half_pot | raise_pot | all_in | all-in with equity < 0.70 |
|---|---|---|---|---|---|---|---|
| frequency_old | 300 | 2.3% | 50.7% | 0.7% | 0.3% | 46.0% | 28.3% |
| frequency_new | 300 | 2.3% | 53.7% | 0.7% | 0.3% | 43.0% | 25.3% |
| ml_old | 300 | 2.3% | 39.7% | 8.3% | 2.3% | 47.3% | 29.7% |
| ml_new | 300 | 2.3% | 63.0% | 1.0% | 1.0% | 32.7% | 15.7% |

Decisions changed by the fix: Frequency 3.0%, ML 24.7%. Frequency and ML (new rule) choose the same action in 88.3% of spots.

## Decision latency (warm caches)

| configuration | component | median ms | 95th pct ms |
|---|---|---|---|
| frequency_new | equity | 224.6 | 272.9 |
| frequency_new | model | 1.9 | 2.5 |
| frequency_new | range | 287.3 | 364.8 |
| frequency_new | total | 522.6 | 624.6 |
| ml_new | equity | 225.9 | 278.3 |
| ml_new | model | 32.3 | 39.8 |
| ml_new | range | 190.4 | 215.0 |
| ml_new | total | 435.0 | 522.7 |
| frequency_old | equity | 225.1 | 275.8 |
| frequency_old | model | 1.9 | 2.6 |
| frequency_old | range | 0.0 | 0.0 |
| frequency_old | total | 229.3 | 282.0 |
| ml_old | equity | 222.5 | 258.0 |
| ml_old | model | 32.4 | 39.5 |
| ml_old | range | 0.0 | 0.0 |
| ml_old | total | 257.9 | 301.3 |

By street (new rule, total):

| configuration | street | median ms | 95th pct ms |
|---|---|---|---|
| frequency_new | flop | 517.8 | 628.6 |
| frequency_new | preflop | 202.1 | 230.6 |
| frequency_new | river | 538.3 | 643.1 |
| frequency_new | turn | 560.6 | 629.6 |
| ml_new | flop | 459.6 | 549.4 |
| ml_new | preflop | 227.7 | 251.1 |
| ml_new | river | 277.7 | 342.9 |
| ml_new | turn | 461.6 | 536.8 |
