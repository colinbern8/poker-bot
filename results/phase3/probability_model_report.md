# Phase 4 probability model: selection and calibration

**Decision context.** The Stage 3 classification models were selected by validation macro-F1 and were poorly calibrated. *After inspecting those results*, the project owner decided that the model feeding the Phase 4 agent is selected by validation log-loss instead. Stage 3 models and results are unchanged. Test and held-out data were not used for this choice; they are reported here for evaluation only.

## 1. Selection (validation only)

- Rule: lowest validation log-loss of legal-masked probabilities among Feature Set B settings (Random Forest and HistGradientBoosting grids); ties: higher validation macro-F1.
- Candidates: 84 Feature Set B settings.
- **Winner: Random Forest, grid #22: {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20, "max_features": 0.5, "class_weight": null}**, seed 2305977157.
- Validation log-loss 0.5209 (masked), macro-F1 0.379, accuracy 0.793. Refit reproduced the grid value exactly.
- Best HistGradientBoosting B setting: grid #2 {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 1.0, "class_weight": null}, validation log-loss 0.5274.
- RF minus HGB on validation (5 matches, paired bootstrap): -0.007 [-0.012, -0.001] log-loss.
- Runner-up margin 0.0010; the five lowest validation log-losses are all Random Forest settings within 0.002 of each other, so the choice is not a single lucky setting.

| model | grid_index | val_log_loss | val_macro_f1 |
|---|---|---|---|
| random_forest | 22 | 0.5209 | 0.3786 |
| random_forest | 34 | 0.5219 | 0.3782 |
| random_forest | 10 | 0.5220 | 0.3784 |
| random_forest | 32 | 0.5228 | 0.3665 |
| random_forest | 8 | 0.5228 | 0.3653 |

## 2. Probability quality on every set (point estimates)

`masked` = legal-action-masked and renormalized (what the agent uses; primary). `raw` = the model's own probabilities over all five actions. ECE = top-label expected calibration error (10 bins).

| model | set | rows | log_loss_masked | log_loss_raw | brier_masked | brier_raw | ece_masked | raw_mass_on_illegal |
|---|---|---|---|---|---|---|---|---|
| Dummy (prior) | train | 23603 | 0.899 | 0.932 | 0.467 | 0.469 | 0.024 | 0.031 |
| Dummy (prior) | validation | 4252 | 0.786 | 0.818 | 0.392 | 0.395 | 0.043 | 0.031 |
| Dummy (prior) | test | 4895 | 0.814 | 0.848 | 0.412 | 0.413 | 0.030 | 0.033 |
| Dummy (prior) | heldout | 5570 | 0.867 | 0.899 | 0.446 | 0.448 | 0.007 | 0.031 |
| RF-B, macro-F1 selected (Stage 3) | train | 23603 | 0.310 | 0.312 | 0.165 | 0.166 | 0.103 | 0.002 |
| RF-B, macro-F1 selected (Stage 3) | validation | 4252 | 0.678 | 0.693 | 0.381 | 0.381 | 0.063 | 0.001 |
| RF-B, macro-F1 selected (Stage 3) | test | 4895 | 0.832 | 0.853 | 0.459 | 0.460 | 0.078 | 0.003 |
| RF-B, macro-F1 selected (Stage 3) | heldout | 5570 | 0.824 | 0.834 | 0.451 | 0.452 | 0.040 | 0.002 |
| HGB-B, macro-F1 selected (Stage 3) | train | 23603 | 0.676 | 0.676 | 0.379 | 0.379 | 0.035 | 0.000 |
| HGB-B, macro-F1 selected (Stage 3) | validation | 4252 | 0.747 | 0.747 | 0.422 | 0.422 | 0.059 | 0.000 |
| HGB-B, macro-F1 selected (Stage 3) | test | 4895 | 0.892 | 0.892 | 0.502 | 0.502 | 0.042 | 0.000 |
| HGB-B, macro-F1 selected (Stage 3) | heldout | 5570 | 0.886 | 0.886 | 0.490 | 0.490 | 0.056 | 0.000 |
| RF-B, log-loss selected (Phase 4 candidate) | train | 23603 | 0.567 | 0.568 | 0.317 | 0.317 | 0.036 | 0.001 |
| RF-B, log-loss selected (Phase 4 candidate) | validation | 4252 | 0.521 | 0.521 | 0.288 | 0.288 | 0.016 | 0.001 |
| RF-B, log-loss selected (Phase 4 candidate) | test | 4895 | 0.656 | 0.658 | 0.350 | 0.350 | 0.016 | 0.002 |
| RF-B, log-loss selected (Phase 4 candidate) | heldout | 5570 | 0.676 | 0.677 | 0.362 | 0.362 | 0.013 | 0.001 |
| HGB-B, best log-loss setting | train | 23603 | 0.536 | 0.536 | 0.303 | 0.303 | 0.043 | 0.000 |
| HGB-B, best log-loss setting | validation | 4252 | 0.527 | 0.528 | 0.291 | 0.291 | 0.009 | 0.000 |
| HGB-B, best log-loss setting | test | 4895 | 0.662 | 0.662 | 0.352 | 0.352 | 0.014 | 0.000 |
| HGB-B, best log-loss setting | heldout | 5570 | 0.679 | 0.679 | 0.366 | 0.366 | 0.016 | 0.000 |

Classification metrics for the same models (legal-masked):

| model | set | macro_f1_masked | balanced_accuracy_masked | accuracy_masked |
|---|---|---|---|---|
| Dummy (prior) | train | 0.166 | 0.200 | 0.707 |
| Dummy (prior) | validation | 0.174 | 0.200 | 0.770 |
| Dummy (prior) | test | 0.172 | 0.200 | 0.754 |
| Dummy (prior) | heldout | 0.168 | 0.200 | 0.724 |
| RF-B, macro-F1 selected (Stage 3) | train | 0.863 | 0.969 | 0.895 |
| RF-B, macro-F1 selected (Stage 3) | validation | 0.454 | 0.522 | 0.728 |
| RF-B, macro-F1 selected (Stage 3) | test | 0.361 | 0.405 | 0.658 |
| RF-B, macro-F1 selected (Stage 3) | heldout | 0.372 | 0.406 | 0.663 |
| HGB-B, macro-F1 selected (Stage 3) | train | 0.649 | 0.851 | 0.725 |
| HGB-B, macro-F1 selected (Stage 3) | validation | 0.444 | 0.567 | 0.713 |
| HGB-B, macro-F1 selected (Stage 3) | test | 0.366 | 0.469 | 0.593 |
| HGB-B, macro-F1 selected (Stage 3) | heldout | 0.389 | 0.486 | 0.618 |
| RF-B, log-loss selected (Phase 4 candidate) | train | 0.428 | 0.392 | 0.769 |
| RF-B, log-loss selected (Phase 4 candidate) | validation | 0.379 | 0.363 | 0.793 |
| RF-B, log-loss selected (Phase 4 candidate) | test | 0.295 | 0.278 | 0.758 |
| RF-B, log-loss selected (Phase 4 candidate) | heldout | 0.324 | 0.322 | 0.734 |
| HGB-B, best log-loss setting | train | 0.575 | 0.514 | 0.783 |
| HGB-B, best log-loss setting | validation | 0.395 | 0.385 | 0.788 |
| HGB-B, best log-loss setting | test | 0.331 | 0.323 | 0.758 |
| HGB-B, best log-loss setting | heldout | 0.342 | 0.341 | 0.732 |

## 3a. test set (5 matches), 95% bootstrap intervals

| model | log-loss | Brier |
|---|---|---|
| Dummy (prior) | 0.814 [0.665, 1.024] | 0.412 [0.327, 0.536] |
| RF-B, macro-F1 selected (Stage 3) | 0.832 [0.682, 1.002] | 0.459 [0.374, 0.554] |
| HGB-B, macro-F1 selected (Stage 3) | 0.892 [0.693, 1.138] | 0.502 [0.388, 0.646] |
| RF-B, log-loss selected (Phase 4 candidate) | 0.656 [0.521, 0.812] | 0.350 [0.276, 0.441] |
| HGB-B, best log-loss setting | 0.662 [0.524, 0.817] | 0.352 [0.280, 0.444] |

Paired differences, candidate minus comparison (negative log-loss/Brier = candidate better):

| comparison | log-loss | Brier | macro-F1 |
|---|---|---|---|
| candidate - Dummy (prior) | -0.157 [-0.211, -0.108] | -0.062 [-0.094, -0.038] | +0.123 [-0.000, +0.180] |
| candidate - RF-B, macro-F1 selected (Stage 3) | -0.176 [-0.210, -0.146] | -0.109 [-0.131, -0.087] | -0.066 [-0.146, -0.038] |
| candidate - HGB-B, macro-F1 selected (Stage 3) | -0.235 [-0.324, -0.168] | -0.153 [-0.203, -0.110] | -0.071 [-0.173, -0.014] |
| candidate - HGB-B, best log-loss setting | -0.006 [-0.011, -0.002] | -0.003 [-0.005, -0.000] | -0.036 [-0.049, -0.015] |

## 3b. Held-out set (12 matches), 95% bootstrap intervals

| model | log-loss | Brier |
|---|---|---|
| Dummy (prior) | 0.867 [0.647, 1.104] | 0.446 [0.307, 0.597] |
| RF-B, macro-F1 selected (Stage 3) | 0.824 [0.571, 1.073] | 0.451 [0.301, 0.597] |
| HGB-B, macro-F1 selected (Stage 3) | 0.886 [0.594, 1.175] | 0.490 [0.320, 0.656] |
| RF-B, log-loss selected (Phase 4 candidate) | 0.676 [0.460, 0.891] | 0.362 [0.234, 0.494] |
| HGB-B, best log-loss setting | 0.679 [0.459, 0.898] | 0.366 [0.236, 0.497] |

Paired differences, candidate minus comparison (negative log-loss/Brier = candidate better):

| comparison | log-loss | Brier | macro-F1 |
|---|---|---|---|
| candidate - Dummy (prior) | -0.191 [-0.253, -0.128] | -0.084 [-0.120, -0.049] | +0.156 [+0.068, +0.189] |
| candidate - RF-B, macro-F1 selected (Stage 3) | -0.148 [-0.194, -0.098] | -0.089 [-0.118, -0.059] | -0.048 [-0.098, -0.029] |
| candidate - HGB-B, macro-F1 selected (Stage 3) | -0.210 [-0.294, -0.127] | -0.128 [-0.175, -0.081] | -0.065 [-0.117, -0.041] |
| candidate - HGB-B, best log-loss setting | -0.002 [-0.011, +0.005] | -0.003 [-0.006, -0.001] | -0.018 [-0.047, -0.003] |

## 4. Calibration diagnostics

### 4a. Top-label expected calibration error

| model | ECE validation | ECE test+held-out |
|---|---|---|
| Dummy (prior) | 0.043 | 0.011 |
| RF-B, macro-F1 selected (Stage 3) | 0.063 | 0.058 |
| HGB-B, macro-F1 selected (Stage 3) | 0.059 | 0.043 |
| RF-B, log-loss selected (Phase 4 candidate) | 0.016 | 0.014 |
| HGB-B, best log-loss setting | 0.009 | 0.011 |

Reliability bins of the candidate, test + held-out:

| bin_low | bin_high | rows | mean_predicted | observed |
|---|---|---|---|---|
| 0.300 | 0.400 | 383 | 0.372 | 0.339 |
| 0.400 | 0.500 | 921 | 0.452 | 0.429 |
| 0.500 | 0.600 | 1284 | 0.551 | 0.543 |
| 0.600 | 0.700 | 2054 | 0.654 | 0.651 |
| 0.700 | 0.800 | 1182 | 0.744 | 0.791 |
| 0.800 | 0.900 | 1673 | 0.856 | 0.870 |
| 0.900 | 1.000 | 2968 | 0.966 | 0.961 |

### 4b. Per-action calibration of the candidate

Per-class ECE compares predicted P(action) with how often the action occurred, in 10 probability bins. With few occurrences (all_in) the bins are sparse and the numbers are rough.

| set | action | occurrences | per-class ECE |
|---|---|---|---|
| validation | fold | 222 | 0.007 |
| validation | check_call | 3274 | 0.017 |
| validation | raise_half_pot | 411 | 0.005 |
| validation | raise_pot | 326 | 0.009 |
| validation | all_in | 19 | 0.001 |
| test+heldout | fold | 366 | 0.006 |
| test+heldout | check_call | 7726 | 0.011 |
| test+heldout | raise_half_pot | 1311 | 0.014 |
| test+heldout | raise_pot | 936 | 0.015 |
| test+heldout | all_in | 126 | 0.003 |

Equal-count reliability bins of the candidate (test + held-out, rows where the action was legal) for raise_pot and all_in:

**raise_pot**

| bin_low | bin_high | rows | mean_predicted | observed |
|---|---|---|---|---|
| 0.0000 | 0.0002 | 1036 | 0.0000 | 0.0000 |
| 0.0002 | 0.0021 | 1022 | 0.0010 | 0.0000 |
| 0.0021 | 0.0077 | 1029 | 0.0042 | 0.0000 |
| 0.0077 | 0.0213 | 1029 | 0.0130 | 0.0049 |
| 0.0213 | 0.0565 | 1029 | 0.0407 | 0.0233 |
| 0.0565 | 0.1152 | 1029 | 0.0837 | 0.0554 |
| 0.1152 | 0.1520 | 1029 | 0.1346 | 0.1108 |
| 0.1520 | 0.1831 | 1029 | 0.1678 | 0.1535 |
| 0.1831 | 0.2602 | 1029 | 0.2089 | 0.2021 |
| 0.2602 | 0.6488 | 1029 | 0.3938 | 0.3596 |

**all_in**

| bin_low | bin_high | rows | mean_predicted | observed |
|---|---|---|---|---|
| 0.0000 | 0.0000 | 2425 | 0.0000 | 0.0000 |
| 0.0000 | 0.0010 | 1692 | 0.0002 | 0.0000 |
| 0.0010 | 0.0087 | 2057 | 0.0040 | 0.0019 |
| 0.0087 | 0.0171 | 2058 | 0.0132 | 0.0190 |
| 0.0171 | 0.1022 | 2058 | 0.0285 | 0.0403 |

### 4c. Calibration in the large (mean predicted / observed share)

| model | set | fold | check_call | raise_half_pot | raise_pot | all_in |
|---|---|---|---|---|---|---|
| Dummy (prior) | validation | 0.014 / 0.052 | 0.730 / 0.770 | 0.130 / 0.097 | 0.116 / 0.077 | 0.009 / 0.004 |
| Dummy (prior) | test | 0.015 / 0.029 | 0.733 / 0.754 | 0.129 / 0.109 | 0.114 / 0.093 | 0.009 / 0.014 |
| Dummy (prior) | heldout | 0.016 / 0.040 | 0.731 / 0.724 | 0.129 / 0.139 | 0.115 / 0.086 | 0.009 / 0.010 |
| RF-B, macro-F1 selected (Stage 3) | validation | 0.079 / 0.052 | 0.576 / 0.770 | 0.192 / 0.097 | 0.132 / 0.077 | 0.020 / 0.004 |
| RF-B, macro-F1 selected (Stage 3) | test | 0.051 / 0.029 | 0.526 / 0.754 | 0.240 / 0.109 | 0.164 / 0.093 | 0.020 / 0.014 |
| RF-B, macro-F1 selected (Stage 3) | heldout | 0.050 / 0.040 | 0.534 / 0.724 | 0.210 / 0.139 | 0.185 / 0.086 | 0.021 / 0.010 |
| HGB-B, macro-F1 selected (Stage 3) | validation | 0.103 / 0.052 | 0.546 / 0.770 | 0.193 / 0.097 | 0.130 / 0.077 | 0.029 / 0.004 |
| HGB-B, macro-F1 selected (Stage 3) | test | 0.067 / 0.029 | 0.492 / 0.754 | 0.235 / 0.109 | 0.176 / 0.093 | 0.030 / 0.014 |
| HGB-B, macro-F1 selected (Stage 3) | heldout | 0.074 / 0.040 | 0.508 / 0.724 | 0.206 / 0.139 | 0.182 / 0.086 | 0.030 / 0.010 |
| RF-B, log-loss selected (Phase 4 candidate) | validation | 0.055 / 0.052 | 0.760 / 0.770 | 0.102 / 0.097 | 0.078 / 0.077 | 0.006 / 0.004 |
| RF-B, log-loss selected (Phase 4 candidate) | test | 0.033 / 0.029 | 0.743 / 0.754 | 0.120 / 0.109 | 0.094 / 0.093 | 0.009 / 0.014 |
| RF-B, log-loss selected (Phase 4 candidate) | heldout | 0.045 / 0.040 | 0.718 / 0.724 | 0.117 / 0.139 | 0.111 / 0.086 | 0.009 / 0.010 |
| HGB-B, best log-loss setting | validation | 0.055 / 0.052 | 0.762 / 0.770 | 0.097 / 0.097 | 0.077 / 0.077 | 0.008 / 0.004 |
| HGB-B, best log-loss setting | test | 0.035 / 0.029 | 0.746 / 0.754 | 0.116 / 0.109 | 0.096 / 0.093 | 0.007 / 0.014 |
| HGB-B, best log-loss setting | heldout | 0.038 / 0.040 | 0.722 / 0.724 | 0.119 / 0.139 | 0.112 / 0.086 | 0.009 / 0.010 |

### 4d. One-vs-rest AUC (test + held-out)

How well each action's probability ranks rows where it happened above rows where it did not (0.5 = no ranking ability). The Dummy gives every row with the same legal actions the same probabilities.

| model | fold | check_call | raise_half_pot | raise_pot | all_in |
|---|---|---|---|---|---|
| Dummy (prior) | 0.844 | 0.477 | 0.544 | 0.629 | 0.632 |
| RF-B, macro-F1 selected (Stage 3) | 0.958 | 0.784 | 0.659 | 0.829 | 0.785 |
| HGB-B, macro-F1 selected (Stage 3) | 0.966 | 0.795 | 0.653 | 0.830 | 0.850 |
| RF-B, log-loss selected (Phase 4 candidate) | 0.969 | 0.806 | 0.708 | 0.856 | 0.855 |
| HGB-B, best log-loss setting | 0.968 | 0.803 | 0.706 | 0.850 | 0.857 |

## 5. Overfitting check

Train vs unseen-data log-loss. A large gap means the model memorizes training rows.

| model | train log-loss | validation | test | held-out | gap validation - train | train macro-F1 | validation macro-F1 |
|---|---|---|---|---|---|---|---|
| Dummy (prior) | 0.899 | 0.786 | 0.814 | 0.867 | -0.113 | 0.166 | 0.174 |
| RF-B, macro-F1 selected (Stage 3) | 0.310 | 0.678 | 0.832 | 0.824 | 0.367 | 0.863 | 0.454 |
| HGB-B, macro-F1 selected (Stage 3) | 0.676 | 0.747 | 0.892 | 0.886 | 0.071 | 0.649 | 0.444 |
| RF-B, log-loss selected (Phase 4 candidate) | 0.567 | 0.521 | 0.656 | 0.676 | -0.047 | 0.428 | 0.379 |
| HGB-B, best log-loss setting | 0.536 | 0.527 | 0.662 | 0.679 | -0.008 | 0.575 | 0.395 |

## 6. Per-style log-loss (legal-masked)

The style-prior reference predicts each row from the training action frequencies of the acting player's *true* style family. It uses labels and cannot be deployed; it shows how much knowing the style alone would give. This investigation is descriptive only: nothing here changes the selected model.

| set | style | rows | matches | Dummy (prior) | RF-B, macro-F1 selected (Stage 3) | HGB-B, macro-F1 selected (Stage 3) | RF-B, log-loss selected (Phase 4 candidate) | HGB-B, best log-loss setting | style-prior reference (uses labels; not deployable) |
|---|---|---|---|---|---|---|---|---|---|
| validation | calling_station | 1612 | 2 | 0.367 | 0.226 | 0.201 | 0.129 | 0.127 | 0.137 |
| validation | maniac | 756 | 2 | 1.438 | 1.299 | 1.445 | 1.118 | 1.137 | 1.159 |
| validation | nit | 1518 | 3 | 0.871 | 0.758 | 0.867 | 0.554 | 0.563 | 0.654 |
| validation | solid | 366 | 1 | 0.938 | 1.049 | 1.212 | 0.874 | 0.884 | 0.932 |
| test | calling_station | 1649 | 3 | 0.415 | 0.317 | 0.305 | 0.219 | 0.220 | 0.250 |
| test | maniac | 352 | 1 | 1.518 | 1.282 | 1.489 | 1.160 | 1.183 | 1.203 |
| test | nit | 450 | 1 | 0.949 | 0.896 | 1.140 | 0.700 | 0.692 | 0.785 |
| test | solid | 2444 | 4 | 0.956 | 1.104 | 1.155 | 0.870 | 0.879 | 0.947 |
| heldout | calling_station | 1655 | 3 | 0.359 | 0.197 | 0.169 | 0.149 | 0.142 | 0.118 |
| heldout | maniac | 1311 | 3 | 1.406 | 1.308 | 1.475 | 1.133 | 1.138 | 1.167 |
| heldout | nit | 1206 | 3 | 0.947 | 0.891 | 0.905 | 0.713 | 0.709 | 0.761 |
| heldout | solid | 1398 | 3 | 0.893 | 1.054 | 1.166 | 0.840 | 0.856 | 0.890 |

### 6a. Solid opponents (test + held-out): where the loss comes from

| true action | rows | dummy mean -log p(true) | rf_ll mean -log p(true) | rf_ll mean p(action) on solid rows | observed share on solid rows |
|---|---|---|---|---|---|
| fold | 114 | 3.080 | 1.399 | 0.035 | 0.030 |
| check_call | 2686 | 0.313 | 0.370 | 0.682 | 0.699 |
| raise_half_pot | 519 | 2.027 | 1.948 | 0.139 | 0.135 |
| raise_pot | 435 | 2.140 | 1.827 | 0.129 | 0.113 |
| all_in | 88 | 4.669 | 3.905 | 0.014 | 0.023 |

## 7. The rare all-in class

Where all-in rows occur (rows used):

| set | all_in rows | share | calling_station | maniac | nit | solid |
|---|---|---|---|---|---|---|
| train | 213 | 0.0090 | 0 | 122 | 0 | 91 |
| validation | 19 | 0.0045 | 0 | 13 | 0 | 6 |
| test | 69 | 0.0141 | 0 | 8 | 0 | 61 |
| heldout | 57 | 0.0102 | 0 | 30 | 0 | 27 |

All-in rows by street and family (all sets):

| street | maniac | solid |
|---|---|---|
| flop | 50 | 42 |
| river | 89 | 100 |
| turn | 34 | 43 |

Probability given to all-in (test + held-out, legal-masked):

| model | mean p(all_in) \| all_in | mean p(all_in) \| other | AUC | average precision | base rate |
|---|---|---|---|---|---|
| Dummy (prior) | 0.0094 | 0.0091 | 0.6324 | 0.0165 | 0.0120 |
| RF-B, macro-F1 selected (Stage 3) | 0.0460 | 0.0204 | 0.7845 | 0.0288 | 0.0120 |
| HGB-B, macro-F1 selected (Stage 3) | 0.1090 | 0.0291 | 0.8505 | 0.0444 | 0.0120 |
| RF-B, log-loss selected (Phase 4 candidate) | 0.0265 | 0.0088 | 0.8547 | 0.0552 | 0.0120 |
| HGB-B, best log-loss setting | 0.0282 | 0.0080 | 0.8574 | 0.0463 | 0.0120 |

## Figures

- `docs/figures/phase3/fig9_reliability.png`
- `docs/figures/phase3/fig10_class_reliability.png`
- `docs/figures/phase3/fig11_probability_quality.png`
