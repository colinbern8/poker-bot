"""Select the Phase 4 probability model: Feature Set B, lowest validation log-loss.

    python scripts/select_probability_model.py

Decision context (documented in docs/phase3_plan.md): the Stage 3
classification models were selected by validation macro-F1 and turned out
to be poorly calibrated. **After inspecting those results**, the project
owner decided that the model feeding the Phase 4 decision agent is chosen
by probability quality instead. The Stage 3 classification models and
results are kept unchanged.

Rule (fixed before refitting):

* candidates: every Feature Set B setting in the existing Random Forest and
  HistGradientBoosting grids (results/phase3/tuning_results.csv, fitted on
  train, scored on validation);
* primary criterion: lowest validation log-loss of the **legal-masked**
  probabilities (what the agent will use); ties: higher validation macro-F1;
* test and held-out data are not used for the choice.

The lowest-log-loss setting of each family is refitted on the train split
with the same seed as in the grid, and its validation log-loss must match
the grid value. Outputs: models/phase3/<family>_logloss_B.joblib and
data/processed/phase3/predictions_logloss.parquet (git-ignored), and
results/phase3/probability_model_selection.json (committed).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hybridpoker.action_probabilities import CLASS_ORDER, mask_and_renormalize
from hybridpoker.dataset_loader import load_dataset, prior_rows
from hybridpoker.dataset_manifest import DEFAULT_MANIFEST_DIR, MAIN_MANIFEST_NAME, load_manifest
from hybridpoker.evaluation import evaluate, match_stats, paired_difference, resample_weights
from hybridpoker.hand_log import _git_commit
from hybridpoker.opponent_history import fit_priors
from hybridpoker.opponent_model import (
    FeaturePreprocessor,
    class_probabilities,
    encode_targets,
    feature_matrix,
    grid,
    legal_matrix,
    make_model,
    model_seed,
)
from hybridpoker.reporting import evaluation_set
from hybridpoker.seeding import derive_seed

RAW = Path("data/raw/simulations/phase3_s3000")
MODEL_DIR = Path("models/phase3")
PROCESSED = Path("data/processed/phase3/predictions_logloss.parquet")
RESULTS = Path("results/phase3")
FAMILIES = ("random_forest", "hist_gradient_boosting")


def main() -> None:
    start = time.perf_counter()
    tuning = pd.read_csv(RESULTS / "tuning_results.csv")
    candidates = tuning[(tuning["feature_set"] == "B") & tuning["model"].isin(FAMILIES)].copy()
    ranked = candidates.sort_values(["val_log_loss", "val_macro_f1"], ascending=[True, False])
    family_best = {f: ranked[ranked["model"] == f].iloc[0] for f in FAMILIES}
    winner_family = ranked.iloc[0]["model"]

    manifest = load_manifest(DEFAULT_MANIFEST_DIR / MAIN_MANIFEST_NAME)
    frame = load_dataset(manifest, RAW)
    frame["eval_set"] = evaluation_set(frame)
    priors = fit_priors(prior_rows(frame, manifest))
    used = frame["eval_set"].notna().to_numpy()
    sets = frame.loc[used, "eval_set"].to_numpy()
    train, val = sets == "train", sets == "validation"
    y = encode_targets(frame.loc[used, "act_category"])
    legal = legal_matrix(frame.loc[used])
    features = feature_matrix(frame, "B", priors).loc[used]
    prep = FeaturePreprocessor().fit(features[train])
    x = prep.transform(features)

    out = frame.loc[used, ["id_match", "eval_set", "label_actor_family", "obs_street"]].reset_index(names="row")
    summary = {"rule": "lowest validation log-loss of legal-masked probabilities among Feature Set B settings "
                       "(Random Forest and HistGradientBoosting grids); ties: higher validation macro-F1",
               "decision_note": "Chosen after inspecting the Stage 3 classification results (macro-F1-selected "
                                "models were poorly calibrated). Test and held-out data were not used for the choice.",
               "candidates": int(len(candidates)), "families": {}, "winner": None}
    val_stats = {}
    for family, row in family_best.items():
        index = int(row["grid_index"])
        params = grid(family)[index]
        seed = model_seed(family, "B")
        model = make_model(family, params, seed).fit(x[train], y[train])
        proba = class_probabilities(model, x)
        masked = mask_and_renormalize(proba[val], legal[val]).masked
        val_scores = evaluate(y[val], masked)
        if not np.isclose(val_scores["log_loss"], row["val_log_loss"], atol=1e-6):
            raise RuntimeError(f"{family}: refit validation log-loss {val_scores['log_loss']} != grid {row['val_log_loss']}")
        name = f"{family}_logloss_B"
        joblib.dump({"model": model, "preprocessor": prep, "feature_set": "B", "params": params, "priors": priors,
                     "class_order": CLASS_ORDER, "selection": "validation log-loss"}, MODEL_DIR / f"{name}.joblib")
        for i, cls in enumerate(CLASS_ORDER):
            out[f"p_{name}_{cls}"] = proba[:, i]
        val_stats[family] = match_stats(y[val], masked, out.loc[val, "id_match"].to_numpy())
        rank_in_family = int((ranked[ranked["model"] == family]["val_log_loss"] < row["val_log_loss"]).sum()) + 1
        summary["families"][family] = {"grid_index": index, "params": params, "seed": seed,
                                       "val_log_loss": val_scores["log_loss"], "val_macro_f1": val_scores["macro_f1"],
                                       "val_accuracy": val_scores["accuracy"], "rank_in_family": rank_in_family,
                                       "refit_matches_grid": True}
    weights = resample_weights(len(val_stats[FAMILIES[0]].matches), 2000, derive_seed(3000, "bootstrap", "validation_selection") % 2**32)
    diff = paired_difference(val_stats["random_forest"], val_stats["hist_gradient_boosting"], weights)
    summary["validation_rf_minus_hgb"] = diff
    summary["winner"] = {"family": winner_family, **summary["families"][winner_family]}
    summary["runner_up_margin_val_log_loss"] = float(ranked.iloc[1]["val_log_loss"] - ranked.iloc[0]["val_log_loss"])
    summary["top5"] = ranked.head(5)[["model", "grid_index", "val_log_loss", "val_macro_f1"]].to_dict("records")
    summary["git_commit_at_start"] = _git_commit()
    summary["seconds"] = time.perf_counter() - start

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED, index=False)
    (RESULTS / "probability_model_selection.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                                               encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("winner", "runner_up_margin_val_log_loss", "top5")}, indent=2, default=str))
    print("validation RF - HGB log-loss:", diff["log_loss"])


if __name__ == "__main__":
    main()
