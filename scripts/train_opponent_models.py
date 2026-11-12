"""Stage 3: train and tune the opponent-action models.

    python scripts/train_opponent_models.py

For each feature set (A, B):

1. History priors (Feature Set B) and the feature preprocessor are fitted
   on **train-split rows only**.
2. Every hyperparameter combination in the approved grids is fitted on the
   train split and scored on the validation split by **macro-F1 of the
   legal-masked predictions** (ties: lower masked log-loss, then grid order).
3. The selected model is refitted once more to confirm training is
   reproducible (identical validation probabilities).
4. A shuffled-label control (training labels randomly permuted) is fitted
   with the selected settings.

Outputs:

* models/phase3/*.joblib (git-ignored): fitted models and preprocessors;
* data/processed/phase3/predictions.parquet (git-ignored): raw class
  probabilities of every model for every used row;
* results/phase3/tuning_results.csv and training_summary.json (committed).

Test and held-out rows are only *predicted* here, never used for any
choice. Evaluation is done by scripts/evaluate_opponent_models.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hybridpoker.action_probabilities import CLASS_ORDER, mask_and_renormalize
from hybridpoker.dataset_loader import load_dataset, prior_rows
from hybridpoker.dataset_manifest import DEFAULT_MANIFEST_DIR, MAIN_MANIFEST_NAME, load_manifest, manifest_hash
from hybridpoker.evaluation import evaluate
from hybridpoker.hand_log import _git_commit
from hybridpoker.opponent_history import fit_priors, prior_hand_count
from hybridpoker.opponent_model import (
    FEATURE_SETS,
    FeaturePreprocessor,
    class_probabilities,
    encode_targets,
    feature_matrix,
    grid,
    legal_matrix,
    make_model,
    missing_training_classes,
    model_seed,
)
from hybridpoker.reporting import evaluation_set
from hybridpoker.seeding import derive_seed

DEFAULT_RAW = Path("data/raw/simulations/phase3_s3000")
MODEL_DIR = Path("models/phase3")
PROCESSED_DIR = Path("data/processed/phase3")
RESULTS_DIR = Path("results/phase3")
TUNED_MODELS = ("random_forest", "hist_gradient_boosting")
KEEP_COLUMNS = ("id_match", "id_match_number", "id_deal_index", "obs_street", "act_category",
                "label_actor_family", "label_actor_agent", "eval_set")


def working_tree_dirty() -> bool | None:
    """True if tracked files had uncommitted changes when training started (None if git is unavailable)."""
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def validation_scores(model, x_val, y_val, legal_val) -> dict:
    proba = class_probabilities(model, x_val)
    masked = mask_and_renormalize(proba, legal_val).masked
    m = evaluate(y_val, masked)
    raw = evaluate(y_val, proba)
    return {"val_macro_f1": m["macro_f1"], "val_log_loss": m["log_loss"], "val_accuracy": m["accuracy"],
            "val_balanced_accuracy": m["balanced_accuracy"], "val_macro_f1_raw": raw["macro_f1"],
            "val_all_in_f1": m["per_class"]["all_in"]["f1"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--quick", action="store_true", help="first 2 grid points only (smoke test)")
    args = parser.parse_args()
    start_all = time.perf_counter()
    commit = _git_commit()
    dirty = working_tree_dirty()

    manifest = load_manifest(DEFAULT_MANIFEST_DIR / MAIN_MANIFEST_NAME)
    frame = load_dataset(manifest, args.raw)
    frame["eval_set"] = evaluation_set(frame)
    priors = fit_priors(prior_rows(frame, manifest))  # train split only
    used = frame["eval_set"].notna().to_numpy()
    sets = frame.loc[used, "eval_set"].to_numpy()
    rows = {name: sets == name for name in ("train", "validation", "test", "heldout")}
    y = encode_targets(frame.loc[used, "act_category"])
    legal = legal_matrix(frame.loc[used])
    print({k: int(v.sum()) for k, v in rows.items()}, flush=True)

    predictions = frame.loc[used, list(KEEP_COLUMNS)].reset_index(names="row")
    predictions["y"] = y
    for i, name in enumerate(CLASS_ORDER):
        predictions[f"legal_{name}"] = legal[:, i]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tuning_rows, summary = [], {"feature_sets": {}, "priors": priors.to_dict()}
    for feature_set in FEATURE_SETS:
        features = feature_matrix(frame, feature_set, priors if feature_set == "B" else None).loc[used]
        if feature_set == "B":
            predictions["prior_hands"] = prior_hand_count(features).to_numpy()
        prep = FeaturePreprocessor().fit(features[rows["train"]])
        x = prep.transform(features)
        x_train, y_train = x[rows["train"]], y[rows["train"]]
        x_val, y_val, legal_val = x[rows["validation"]], y[rows["validation"]], legal[rows["validation"]]
        set_summary = {"columns": prep.columns, "dropped_constant": prep.dropped_constant, "models": {}}
        print(f"\nFeature set {feature_set}: {len(prep.columns)} columns; dropped {prep.dropped_constant}", flush=True)

        fitted = {}
        dummy = make_model("dummy", {}, seed=0).fit(x_train, y_train)
        fitted["dummy"] = (dummy, {}, validation_scores(dummy, x_val, y_val, legal_val))

        for model_name in TUNED_MODELS:
            seed = model_seed(model_name, feature_set)
            best = None
            combos = grid(model_name)[:2] if args.quick else grid(model_name)
            for index, params in enumerate(combos):
                t0 = time.perf_counter()
                model = make_model(model_name, params, seed).fit(x_train, y_train)
                fit_seconds = time.perf_counter() - t0
                scores = validation_scores(model, x_val, y_val, legal_val)
                tuning_rows.append({"feature_set": feature_set, "model": model_name, "grid_index": index,
                                    **{k: ("None" if v is None else v) for k, v in params.items()},
                                    **scores, "fit_seconds": fit_seconds})
                key = (scores["val_macro_f1"], -scores["val_log_loss"])
                if best is None or key > best[0]:
                    best = (key, index, params, model, scores)
                print(f"  {model_name} {feature_set} #{index:02d} macro-F1 {scores['val_macro_f1']:.4f} "
                      f"log-loss {scores['val_log_loss']:.4f} ({fit_seconds:.1f}s)", flush=True)
            _, index, params, model, scores = best
            refit = make_model(model_name, params, seed).fit(x_train, y_train)
            reproducible = bool(np.array_equal(class_probabilities(refit, x_val), class_probabilities(model, x_val)))
            fitted[model_name] = (model, params, scores)
            set_summary["models"][model_name] = {"selected_grid_index": index, "params": params, "seed": seed,
                                                 "validation": scores, "refit_identical": reproducible,
                                                 "missing_training_classes": missing_training_classes(model)}
            print(f"  -> selected #{index}: {params}; refit identical: {reproducible}", flush=True)

            # Shuffled-label control: same settings, training labels randomly permuted.
            shuffle_rng = np.random.default_rng(derive_seed(3000, "shuffled_labels", model_name, feature_set) % 2**32)
            control = make_model(model_name, params, seed).fit(x_train, shuffle_rng.permutation(y_train))
            fitted[f"{model_name}_shuffled"] = (control, params, validation_scores(control, x_val, y_val, legal_val))

        set_summary["models"]["dummy"] = {"params": {"strategy": "prior"}, "validation": fitted["dummy"][2],
                                          "missing_training_classes": missing_training_classes(dummy)}
        for name, (model, params, _) in fitted.items():
            proba = class_probabilities(model, x)
            for i, cls in enumerate(CLASS_ORDER):
                predictions[f"p_{name}_{feature_set}_{cls}"] = proba[:, i]
            joblib.dump({"model": model, "preprocessor": prep, "feature_set": feature_set, "params": params,
                         "priors": priors if feature_set == "B" else None, "class_order": CLASS_ORDER},
                        MODEL_DIR / f"{name}_{feature_set}.joblib")
        summary["feature_sets"][feature_set] = set_summary

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PROCESSED_DIR / "predictions.parquet", index=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning_rows).to_csv(RESULTS_DIR / "tuning_results.csv", index=False, lineterminator="\n",
                                     float_format="%.6f")
    summary.update({
        "selection_rule": "max validation macro-F1 of legal-masked predictions; ties: lower masked log-loss, then grid order",
        "rows": {k: int(v.sum()) for k, v in rows.items()},
        "manifest_hash": manifest_hash(manifest),
        "quick": args.quick,
        "seconds": time.perf_counter() - start_all,
        "versions": {"python": platform.python_version(), "scikit-learn": version("scikit-learn"),
                     "numpy": version("numpy"), "pandas": version("pandas"), "git_commit_at_start": commit,
                     "working_tree_dirty_at_start": dirty},
    })
    (RESULTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                                        encoding="utf-8")
    print(f"\nDone in {summary['seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
