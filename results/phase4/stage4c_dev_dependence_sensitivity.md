# Dependence sensitivity (development run)

Secondary sensitivity check requested after the freeze and written before any final hand was played (`scripts/phase4_dependence_sensitivity.py`). It does **not** change the registered primary result; see the script's header for the methods and the reading rule.

| comparison | role | registered (i.i.d.) | batch means (10 batches) | Newey-West (lag 10) | block bootstrap (10) | SE ratio batch / HAC | lag-1 to lag-5 autocorrelation |
|---|---|---|---|---|---|---|---|
| ev_ml - ev_situation | primary | +60.8 [-10.9, +132.6] | +60.8 [-12.1, +133.8] | +60.8 [-5.9, +127.6] | [-6.8, +125.4] | 1.02 / 0.93 | +0.027, -0.008, -0.010, -0.015, -0.057 |
| ev_ml - ev_global | secondary | +48.2 [-24.3, +120.7] | +48.2 [-25.9, +122.2] | +48.2 [-17.6, +113.9] | [-17.8, +111.2] | 1.02 / 0.91 | -0.015, -0.022, +0.006, -0.047, -0.008 |
| ev_ml - probability | secondary | +447.6 [+334.9, +560.3] | +447.6 [+340.5, +554.6] | +447.6 [+348.8, +546.4] | [+347.5, +549.5] | 0.95 / 0.88 | -0.055, -0.044, -0.013, -0.014, -0.014 |

Does each interval exclude 0?

| comparison | registered | batch means | Newey-West | block bootstrap |
|---|---|---|---|---|
| ev_ml - ev_situation | no | no | no | no |
| ev_ml - ev_global | no | no | no | no |
| ev_ml - probability | yes | yes | yes | yes |
