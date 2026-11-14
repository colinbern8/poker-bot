# Dependence sensitivity (final evaluation)

Secondary sensitivity check requested after the freeze and written before any final hand was played (`scripts/phase4_dependence_sensitivity.py`). It does **not** change the registered primary result; see the script's header for the methods and the reading rule.

| comparison | role | registered (i.i.d.) | batch means (10 batches) | Newey-West (lag 10) | block bootstrap (10) | SE ratio batch / HAC | lag-1 to lag-5 autocorrelation |
|---|---|---|---|---|---|---|---|
| ev_ml - ev_situation | primary | +84.8 [+51.6, +117.9] | +84.8 [+51.6, +118.0] | +84.8 [+52.0, +117.5] | [+52.0, +117.5] | 1.00 / 0.99 | -0.014, -0.007, +0.002, +0.005, +0.018 |
| ev_ml - ev_global | secondary | +102.1 [+69.6, +134.6] | +102.1 [+67.0, +137.2] | +102.1 [+69.2, +135.0] | [+69.1, +135.8] | 1.08 / 1.01 | -0.007, +0.008, +0.002, +0.008, +0.010 |
| ev_ml - probability | secondary | +469.4 [+419.0, +519.9] | +469.4 [+421.8, +517.0] | +469.4 [+418.3, +520.5] | [+417.5, +519.4] | 0.94 / 1.01 | -0.003, +0.011, +0.001, +0.009, +0.001 |

Does each interval exclude 0?

| comparison | registered | batch means | Newey-West | block bootstrap |
|---|---|---|---|---|
| ev_ml - ev_situation | yes | yes | yes | yes |
| ev_ml - ev_global | yes | yes | yes | yes |
| ev_ml - probability | yes | yes | yes | yes |
