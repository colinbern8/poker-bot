"""Diagnostic: spread of the shuffled-label control across shuffle seeds (HGB, Feature Set B).

    python scripts/shuffled_control_spread.py

Retrains the selected HistGradientBoosting-B settings on five additional
random permutations of the training labels and prints legal-masked test
and held-out macro-F1 and log-loss. Used to check whether one shuffled
control scoring above the random-guess references is chance variation.
Prints only; writes nothing.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from hybridpoker.action_probabilities import mask_and_renormalize
from hybridpoker.dataset_loader import load_dataset, prior_rows
from hybridpoker.dataset_manifest import load_manifest
from hybridpoker.evaluation import evaluate
from hybridpoker.opponent_history import fit_priors
from hybridpoker.opponent_model import FeaturePreprocessor, class_probabilities, encode_targets, feature_matrix, legal_matrix, make_model, model_seed
from hybridpoker.reporting import evaluation_set
m = load_manifest(Path("data/manifests/phase3_manifest.json"))
f = load_dataset(m, Path("data/raw/simulations/phase3_s3000")); f["eval_set"] = evaluation_set(f)
pri = fit_priors(prior_rows(f, m)); used = f["eval_set"].notna().to_numpy()
X = feature_matrix(f, "B", pri).loc[used]; y = encode_targets(f.loc[used, "act_category"]); legal = legal_matrix(f.loc[used])
sets = f.loc[used, "eval_set"].to_numpy(); tr = sets == "train"
prep = FeaturePreprocessor().fit(X[tr]); x = prep.transform(X)
params = json.load(open("results/phase3/training_summary.json"))["feature_sets"]["B"]["models"]["hist_gradient_boosting"]["params"]
for seed in [101, 202, 303, 404, 505]:
    yy = np.random.default_rng(seed).permutation(y[tr])
    model = make_model("hist_gradient_boosting", params, model_seed("hist_gradient_boosting", "B")).fit(x[tr], yy)
    out = []
    for s in ("test", "heldout"):
        r = sets == s
        e = evaluate(y[r], mask_and_renormalize(class_probabilities(model, x[r]), legal[r]).masked)
        out.append(f"{s} macro-F1 {e['macro_f1']:.3f} log-loss {e['log_loss']:.3f}")
    print("shuffle seed", seed, " | ".join(out), flush=True)
