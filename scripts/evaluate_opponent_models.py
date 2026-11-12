"""Stage 3: evaluate the trained opponent models and draw the figures.

    python scripts/evaluate_opponent_models.py

Reads data/processed/phase3/predictions.parquet and models/phase3/ (both
written by train_opponent_models.py) and writes:

* results/phase3/model_results.md and metrics.json (committed);
* docs/figures/phase3/*.png (committed).

**Which probabilities are used.** Every model outputs raw probabilities for
the five actions. The *primary* results use the **legal-masked**
probabilities (illegal actions set to 0, floor 1e-6, renormalized); the
predicted action is the most probable *legal* action. Raw-probability
metrics are reported separately as a diagnostic, together with the
probability mass each model put on illegal actions.

Uncertainty: match-level bootstrap (2,000 resamples of whole matches),
paired for comparisons. The test set has only 5 matches, so its intervals
are rough; the held-out set has 12.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hybridpoker.action_probabilities import CLASS_ORDER, mask_and_renormalize  # noqa: E402
from hybridpoker.dataset_loader import load_dataset  # noqa: E402
from hybridpoker.dataset_manifest import DEFAULT_MANIFEST_DIR, MAIN_MANIFEST_NAME, load_manifest  # noqa: E402
from hybridpoker.evaluation import (  # noqa: E402
    METRICS,
    evaluate,
    match_stats,
    paired_difference,
    resample_weights,
    summarize,
)
from hybridpoker.features import FEATURE_GROUPS_A  # noqa: E402
from hybridpoker.opponent_history import B_FEATURES  # noqa: E402
from hybridpoker.opponent_model import class_probabilities, feature_matrix  # noqa: E402
from hybridpoker.reporting import evaluation_set, md_table  # noqa: E402
from hybridpoker.seeding import derive_seed  # noqa: E402

PREDICTIONS = Path("data/processed/phase3/predictions.parquet")
MODEL_DIR = Path("models/phase3")
RESULTS_DIR = Path("results/phase3")
FIGURE_DIR = Path("docs/figures/phase3")
DEFAULT_RAW = Path("data/raw/simulations/phase3_s3000")
SETS = ("train", "validation", "test", "heldout")
REPORT_SETS = ("test", "heldout")
RESAMPLES = 2000
FAMILIES = ("calling_station", "maniac", "nit", "solid")
HISTORY_BINS = [0, 10, 30, 60, 90, 120]
PERMUTATION_REPEATS = 5

# Reference palette (dataviz skill, light mode): fixed categorical order.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
COLOR_A, COLOR_B, COLOR_DUMMY = SERIES[0], SERIES[1], "#8a8984"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

MODEL_LABELS = {
    "dummy_A": "Dummy (prior)",
    "random_forest_A": "Random Forest - A", "random_forest_B": "Random Forest - B",
    "hist_gradient_boosting_A": "HistGradientBoosting - A", "hist_gradient_boosting_B": "HistGradientBoosting - B",
    "random_forest_shuffled_A": "RF shuffled labels - A", "random_forest_shuffled_B": "RF shuffled labels - B",
    "hist_gradient_boosting_shuffled_A": "HGB shuffled labels - A",
    "hist_gradient_boosting_shuffled_B": "HGB shuffled labels - B",
}
MAIN_MODELS = ("dummy_A", "random_forest_A", "random_forest_B", "hist_gradient_boosting_A", "hist_gradient_boosting_B")


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(rows: int, cols: int, size: tuple[float, float]):
    fig, axes = plt.subplots(rows, cols, figsize=size, facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        style_axes(ax)
    return fig, axes


def save(fig, name: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return str(path)


def model_probabilities(pred: pd.DataFrame, model: str) -> np.ndarray:
    name, feature_set = model.rsplit("_", 1)
    return pred[[f"p_{name}_{feature_set}_{c}" for c in CLASS_ORDER]].to_numpy()


def fmt_ci(entry: dict, digits: int = 3) -> str:
    return f"{entry['value']:.{digits}f} [{entry['ci_low']:.{digits}f}, {entry['ci_high']:.{digits}f}]"


def fmt_diff(entry: dict, digits: int = 3) -> str:
    return f"{entry['value']:+.{digits}f} [{entry['ci_low']:+.{digits}f}, {entry['ci_high']:+.{digits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--skip-permutation", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()

    pred = pd.read_parquet(PREDICTIONS)
    summary = json.loads((RESULTS_DIR / "training_summary.json").read_text(encoding="utf-8"))
    legal = pred[[f"legal_{c}" for c in CLASS_ORDER]].to_numpy(dtype=bool)
    y = pred["y"].to_numpy()
    models = [m for m in MODEL_LABELS if f"p_{m.rsplit('_', 1)[0]}_{m.rsplit('_', 1)[1]}_fold" in pred.columns]
    masks = {s: (pred["eval_set"] == s).to_numpy() for s in SETS}

    probs = {}
    for model in models:
        raw = model_probabilities(pred, model)
        masked = mask_and_renormalize(raw, legal)
        probs[model] = {"raw": raw, "masked": masked.masked, "illegal_mass": masked.illegal_mass}

    # Dummy A and B must be identical (it ignores features).
    if "dummy_B" in pred.columns or "p_dummy_B_fold" in pred.columns:
        assert np.allclose(model_probabilities(pred, "dummy_A"), model_probabilities(pred, "dummy_B"))

    # --- Point metrics on every set, raw and masked ---------------------------------------
    point = {m: {s: {"masked": evaluate(y[masks[s]], probs[m]["masked"][masks[s]]),
                     "raw": evaluate(y[masks[s]], probs[m]["raw"][masks[s]]),
                     "mean_illegal_mass_raw": float(probs[m]["illegal_mass"][masks[s]].mean())}
                 for s in SETS} for m in models}

    # --- Bootstrap on test and held-out --------------------------------------------------
    boot, diffs = {}, {}
    for s in REPORT_SETS:
        ids = pred.loc[masks[s], "id_match"].to_numpy()
        stats = {m: match_stats(y[masks[s]], probs[m]["masked"][masks[s]], ids) for m in models}
        weights = resample_weights(len(stats["dummy_A"].matches), RESAMPLES, derive_seed(3000, "bootstrap", s) % 2**32)
        boot[s] = {m: summarize(stats[m], weights) for m in models}
        boot[s]["_matches"] = len(stats["dummy_A"].matches)
        diffs[s] = {}
        for m in models:
            if m != "dummy_A":
                diffs[s][f"{m} - dummy"] = paired_difference(stats[m], stats["dummy_A"], weights)
        for family in ("random_forest", "hist_gradient_boosting"):
            diffs[s][f"{family}: B - A"] = paired_difference(stats[f"{family}_B"], stats[f"{family}_A"], weights)

    # Best model per feature set = higher validation macro-F1 (the approved selection metric).
    best = {}
    for fs in ("A", "B"):
        candidates = [f"random_forest_{fs}", f"hist_gradient_boosting_{fs}"]
        best[fs] = max(candidates, key=lambda m: point[m]["validation"]["masked"]["macro_f1"])

    # --- Per style, by street, by history length -----------------------------------------
    per_style = {}
    for s in REPORT_SETS:
        per_style[s] = {}
        for family in FAMILIES:
            rows = masks[s] & (pred["label_actor_family"] == family).to_numpy()
            per_style[s][family] = {"rows": int(rows.sum()),
                                    "matches": int(pred.loc[rows, "id_match"].nunique()),
                                    **{m: evaluate(y[rows], probs[m]["masked"][rows]) for m in MAIN_MODELS}}
    eval_rows = masks["test"] | masks["heldout"]
    history = []
    bins = pd.cut(pred["prior_hands"], HISTORY_BINS, right=False)
    for interval, group in pred[eval_rows].groupby(bins[eval_rows], observed=True):
        rows = eval_rows & (bins == interval).to_numpy()
        entry = {"bin": f"{int(interval.left)}-{int(interval.right) - 1}", "rows": int(rows.sum())}
        for m in ("dummy_A", best["A"], best["B"]):
            entry[m] = {k: evaluate(y[rows], probs[m]["masked"][rows])[k] for k in ("macro_f1", "log_loss", "accuracy")}
        history.append(entry)
    by_street = {}
    for street in ("preflop", "flop", "turn", "river"):
        rows = eval_rows & (pred["obs_street"] == street).to_numpy()
        by_street[street] = {"rows": int(rows.sum()),
                             **{m: evaluate(y[rows], probs[m]["masked"][rows])["macro_f1"] for m in MAIN_MODELS}}

    # Predicted vs true class shares (checks the models do not collapse onto one class).
    share_rows = []
    for s in REPORT_SETS:
        true_share = np.bincount(y[masks[s]], minlength=5) / masks[s].sum()
        share_rows.append({"set": s, "source": "true", **{c: true_share[i] for i, c in enumerate(CLASS_ORDER)}})
        for m in MAIN_MODELS:
            p = np.bincount(probs[m]["masked"][masks[s]].argmax(axis=1), minlength=5) / masks[s].sum()
            share_rows.append({"set": s, "source": MODEL_LABELS[m], **{c: p[i] for i, c in enumerate(CLASS_ORDER)}})

    # --- Permutation importance (feature groups) ------------------------------------------
    permutation = {}
    if not args.skip_permutation:
        manifest = load_manifest(DEFAULT_MANIFEST_DIR / MAIN_MANIFEST_NAME)
        frame = load_dataset(manifest, args.raw)
        frame["eval_set"] = evaluation_set(frame)
        groups = {**FEATURE_GROUPS_A, "opponent_history": B_FEATURES}
        for fs in ("A", "B"):
            bundle = joblib.load(MODEL_DIR / f"{best[fs]}.joblib")
            features = feature_matrix(frame, fs, bundle["priors"])
            permutation[best[fs]] = {}
            for s in REPORT_SETS:
                rows_mask = (frame["eval_set"] == s).to_numpy()
                part = features[rows_mask]
                pred_rows = masks[s]
                assert np.array_equal(pred.loc[pred_rows, "row"].to_numpy(), frame.index[rows_mask].to_numpy())
                legal_s, y_s = legal[pred_rows], y[pred_rows]
                base_raw = class_probabilities(bundle["model"], bundle["preprocessor"].transform(part))
                assert np.allclose(base_raw, probs[best[fs]]["raw"][pred_rows]), "saved model differs from predictions"
                base = evaluate(y_s, mask_and_renormalize(base_raw, legal_s).masked)
                rng = np.random.default_rng(derive_seed(3000, "permutation", best[fs], s) % 2**32)
                results = {}
                for group, columns in groups.items():
                    columns = [c for c in columns if c in bundle["preprocessor"].columns]
                    if not columns:
                        continue
                    drops_f1, drops_ll = [], []
                    for _ in range(PERMUTATION_REPEATS):
                        shuffled = part.copy()
                        order = rng.permutation(len(part))
                        shuffled[columns] = part[columns].to_numpy()[order]
                        raw = class_probabilities(bundle["model"], bundle["preprocessor"].transform(shuffled))
                        m = evaluate(y_s, mask_and_renormalize(raw, legal_s).masked)
                        drops_f1.append(base["macro_f1"] - m["macro_f1"])
                        drops_ll.append(m["log_loss"] - base["log_loss"])
                    results[group] = {"columns": len(columns), "macro_f1_drop_mean": float(np.mean(drops_f1)),
                                      "macro_f1_drop_std": float(np.std(drops_f1)),
                                      "log_loss_increase_mean": float(np.mean(drops_ll))}
                permutation[best[fs]][s] = {"baseline_macro_f1": base["macro_f1"], "groups": results}

    # --- Figures ------------------------------------------------------------------------------
    figures = {}
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

    # 1. Action distribution by style family (rows used).
    fig, axes = new_figure(1, 1, (8, 3.4))
    ax = axes[0, 0]
    labels = list(FAMILIES)
    left = np.zeros(len(labels))
    for i, cls in enumerate(CLASS_ORDER):
        shares = np.array([np.mean(y[(pred["label_actor_family"] == f).to_numpy()] == i) for f in labels])
        ax.barh(labels, shares, left=left, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, label=cls, height=0.6)
        left += shares
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Action mix by opponent style (all 38,320 rows used)", loc="left", fontsize=11)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9)
    ax.invert_yaxis()
    figures["action_distribution"] = save(fig, "fig1_action_distribution.png")

    # 2. Model comparison with bootstrap CIs (macro-F1 and log-loss; test and held-out).
    fig, axes = new_figure(2, 2, (11, 6.2))
    order = list(MAIN_MODELS)[::-1]
    colors = {"dummy_A": COLOR_DUMMY}
    for m in MAIN_MODELS[1:]:
        colors[m] = COLOR_A if m.endswith("_A") else COLOR_B
    for r, metric in enumerate(("macro_f1", "log_loss")):
        for c, s in enumerate(REPORT_SETS):
            ax = axes[r, c]
            for k, m in enumerate(order):
                e = boot[s][m][metric]
                ax.errorbar(e["value"], k, xerr=[[e["value"] - e["ci_low"]], [e["ci_high"] - e["value"]]],
                            fmt="o", color=colors[m], ecolor=colors[m], elinewidth=2, capsize=3, markersize=8)
                ax.annotate(f"{e['value']:.3f}", (e["ci_high"], k), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK_2)
            ax.set_yticks(range(len(order)), [MODEL_LABELS[m] for m in order], fontsize=9)
            nice = "macro-F1 (higher is better)" if metric == "macro_f1" else "log-loss (lower is better)"
            ax.set_title(f"{s.replace('heldout', 'held-out')} set ({boot[s]['_matches']} matches): {nice}",
                         loc="left", fontsize=10)
    fig.suptitle("Legal-masked predictions, 95% match-level bootstrap intervals   "
                 "(blue = Feature Set A, orange = Feature Set B, gray = Dummy)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    figures["model_comparison"] = save(fig, "fig2_model_comparison.png")

    # 3. Confusion matrices (row-normalized) of the best A and B models.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = new_figure(2, 2, (10, 9))
    for r, s in enumerate(REPORT_SETS):
        for c, fs in enumerate(("A", "B")):
            ax = axes[r, c]
            ax.grid(False)
            cm = np.array(point[best[fs]][s]["masked"]["confusion"], dtype=float)
            norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7.5,
                            color="white" if norm[i, j] > 0.55 else INK)
            ax.set_xticks(range(5), CLASS_ORDER, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(5), CLASS_ORDER, fontsize=8)
            ax.set_xlabel("predicted (legal-masked)")
            ax.set_ylabel("true action")
            ax.set_title(f"{MODEL_LABELS[best[fs]]} - {s.replace('heldout', 'held-out')}", loc="left", fontsize=10)
    fig.suptitle("Confusion matrices, row-normalized (share of each true action; counts in brackets)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    figures["confusion"] = save(fig, "fig3_confusion_matrices.png")

    # 4. Per-class F1 (held-out and test): Dummy vs best A vs best B.
    fig, axes = new_figure(1, 2, (11, 3.8))
    width = 0.26
    for c, s in enumerate(REPORT_SETS):
        ax = axes[0, c]
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        x = np.arange(5)
        for k, (m, color) in enumerate((("dummy_A", COLOR_DUMMY), (best["A"], COLOR_A), (best["B"], COLOR_B))):
            f1 = [point[m][s]["masked"]["per_class"][cls]["f1"] for cls in CLASS_ORDER]
            ax.bar(x + (k - 1) * width, f1, width - 0.03, color=color, label=MODEL_LABELS[m])
        ax.set_xticks(x, CLASS_ORDER, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.set_title(f"Per-class F1 - {s.replace('heldout', 'held-out')} set", loc="left", fontsize=10)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    figures["per_class_f1"] = save(fig, "fig4_per_class_f1.png")

    # 5. Per-style macro-F1, A vs B (held-out and test).
    fig, axes = new_figure(1, 2, (11, 3.6))
    for c, s in enumerate(REPORT_SETS):
        ax = axes[0, c]
        yy = np.arange(len(FAMILIES))
        for k, (m, color) in enumerate((("dummy_A", COLOR_DUMMY), (best["A"], COLOR_A), (best["B"], COLOR_B))):
            vals = [per_style[s][f][m]["macro_f1"] for f in FAMILIES]
            ax.barh(yy + (k - 1) * 0.26, vals, 0.23, color=color, label=MODEL_LABELS[m])
        ax.set_yticks(yy, [f"{f}\n({per_style[s][f]['rows']} rows, {per_style[s][f]['matches']} matches)"
                           for f in FAMILIES], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("macro-F1 (legal-masked)")
        ax.set_title(f"By opponent style - {s.replace('heldout', 'held-out')} set", loc="left", fontsize=10)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    figures["per_style"] = save(fig, "fig5_per_style.png")

    # 6. Effect of history length: macro-F1 by number of earlier hands observed.
    fig, axes = new_figure(1, 2, (11, 3.6))
    xs = np.arange(len(history))
    for c, metric in enumerate(("macro_f1", "log_loss")):
        ax = axes[0, c]
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for m, color in (("dummy_A", COLOR_DUMMY), (best["A"], COLOR_A), (best["B"], COLOR_B)):
            ax.plot(xs, [h[m][metric] for h in history], color=color, linewidth=2, marker="o", markersize=7,
                    label=MODEL_LABELS[m])
        ax.set_xticks(xs, [f"{h['bin']}\n({h['rows']})" for h in history], fontsize=8)
        ax.set_xlabel("earlier hands of this opponent observed (rows)")
        ax.set_ylabel("macro-F1" if metric == "macro_f1" else "log-loss")
        ax.set_title(("macro-F1 (higher is better)" if metric == "macro_f1" else "log-loss (lower is better)")
                     + " - test + held-out", loc="left", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figures["history_effect"] = save(fig, "fig6_history_effect.png")

    # 7. Permutation importance by feature group.
    if permutation:
        fig, axes = new_figure(1, 2, (11, 3.6))
        for c, fs in enumerate(("A", "B")):
            ax = axes[0, c]
            res = permutation[best[fs]]["heldout"]["groups"]
            names = sorted(res, key=lambda g: res[g]["macro_f1_drop_mean"])
            vals = [res[g]["macro_f1_drop_mean"] for g in names]
            errs = [res[g]["macro_f1_drop_std"] for g in names]
            ax.barh(names, vals, xerr=errs, color=COLOR_A if fs == "A" else COLOR_B, ecolor=INK_2, height=0.6)
            ax.axvline(0, color=INK_2, linewidth=0.8)
            ax.set_xlabel("drop in macro-F1 when the group is shuffled")
            ax.set_title(f"{MODEL_LABELS[best[fs]]} - held-out set", loc="left", fontsize=10)
        fig.tight_layout()
        figures["permutation_importance"] = save(fig, "fig7_permutation_importance.png")

    # --- Report -----------------------------------------------------------------------------
    metrics_out = {"best_by_feature_set": best, "point": point, "bootstrap": boot, "paired_differences": diffs,
                   "per_style": per_style, "history_bins": history, "by_street": by_street,
                   "predicted_shares": share_rows, "permutation_importance": permutation,
                   "resamples": RESAMPLES, "figures": figures}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=float) + "\n",
                                              encoding="utf-8")

    md = ["# Phase 3 Stage 3: opponent-model results", "",
          "Generated by `scripts/evaluate_opponent_models.py`. **Primary metrics use legal-masked probabilities** "
          "(illegal actions set to 0, floor 1e-6, renormalized; prediction = most probable legal action). "
          "Raw-probability metrics are a diagnostic (section 6). Intervals are 95% match-level bootstrap "
          f"intervals ({RESAMPLES} resamples of whole matches). The test set has only "
          f"{boot['test']['_matches']} matches, so its intervals are rough; the held-out set has "
          f"{boot['heldout']['_matches']}.", ""]

    sel_rows = []
    for fs in ("A", "B"):
        for name in ("random_forest", "hist_gradient_boosting"):
            info = summary["feature_sets"][fs]["models"][name]
            sel_rows.append({"model": MODEL_LABELS[f"{name}_{fs}"], "selected settings": json.dumps(info["params"]),
                             "val macro-F1": info["validation"]["val_macro_f1"],
                             "val log-loss": info["validation"]["val_log_loss"],
                             "refit identical": info["refit_identical"]})
    md += ["## 1. Selected models (validation macro-F1, legal-masked)", "", md_table(pd.DataFrame(sel_rows)), "",
           f"Best per feature set (by validation macro-F1): A = {MODEL_LABELS[best['A']]}, "
           f"B = {MODEL_LABELS[best['B']]}.", ""]

    for s in REPORT_SETS:
        rows = []
        for m in MAIN_MODELS:
            b = boot[s][m]
            rows.append({"model": MODEL_LABELS[m], "accuracy": fmt_ci(b["accuracy"]), "macro-F1": fmt_ci(b["macro_f1"]),
                         "balanced acc.": fmt_ci(b["balanced_accuracy"]), "log-loss": fmt_ci(b["log_loss"])})
        md += [f"## 2{'a' if s == 'test' else 'b'}. {s.replace('heldout', 'Held-out')} set "
               f"({int(masks[s].sum())} rows, {boot[s]['_matches']} matches)", "", md_table(pd.DataFrame(rows)), ""]

    for s in REPORT_SETS:
        rows = [{"comparison": k, **{metric: fmt_diff(v[metric]) for metric in METRICS},
                 "P(macro-F1 diff > 0)": f"{v['macro_f1']['share_positive']:.3f}"}
                for k, v in diffs[s].items() if "shuffled" not in k]
        md += [f"## 3{'a' if s == 'test' else 'b'}. Paired differences - {s.replace('heldout', 'held-out')} set", "",
               "Point difference [95% paired bootstrap interval]. For log-loss, negative is better.", "",
               md_table(pd.DataFrame(rows)), ""]

    for s in ("validation",) + REPORT_SETS:
        rows = []
        for m in ("dummy_A", best["A"], best["B"]):
            for cls in CLASS_ORDER:
                pc = point[m][s]["masked"]["per_class"][cls]
                rows.append({"model": MODEL_LABELS[m], "class": cls, "precision": pc["precision"],
                             "recall": pc["recall"], "F1": pc["f1"], "support": int(pc["support"])})
        md += [f"## 4. Per-class precision, recall and F1 - {s.replace('heldout', 'held-out')} set", "",
               md_table(pd.DataFrame(rows)), ""]

    for s in REPORT_SETS:
        rows = []
        for f in FAMILIES:
            e = per_style[s][f]
            rows.append({"style": f, "rows": e["rows"], "matches": e["matches"],
                         **{f"{MODEL_LABELS[m]} macro-F1": e[m]["macro_f1"] for m in MAIN_MODELS},
                         **{f"{MODEL_LABELS[m]} log-loss": e[m]["log_loss"] for m in (best["A"], best["B"])}})
        md += [f"## 5. By opponent style - {s.replace('heldout', 'held-out')} set (point estimates)", "",
               md_table(pd.DataFrame(rows)), ""]

    rows = []
    for m in models:
        for s in SETS:
            rows.append({"model": MODEL_LABELS[m], "set": s,
                         "macro-F1 masked": point[m][s]["masked"]["macro_f1"],
                         "macro-F1 raw": point[m][s]["raw"]["macro_f1"],
                         "log-loss masked": point[m][s]["masked"]["log_loss"],
                         "log-loss raw": point[m][s]["raw"]["log_loss"],
                         "raw mass on illegal actions": point[m][s]["mean_illegal_mass_raw"]})
    md += ["## 6. Raw vs legal-masked probabilities, all sets (point estimates)", "",
           "`raw` = the model's own probabilities (prediction = most probable of all five). "
           "`masked` = primary. The train rows show how much each model fits its training data.", "",
           md_table(pd.DataFrame(rows)), ""]

    md += ["## 7. History length and street (test + held-out, point estimates)", "",
           md_table(pd.DataFrame([{"earlier hands": h["bin"], "rows": h["rows"],
                                   **{f"{MODEL_LABELS[m]} macro-F1": h[m]["macro_f1"] for m in ("dummy_A", best["A"], best["B"])},
                                   **{f"{MODEL_LABELS[m]} log-loss": h[m]["log_loss"] for m in (best["A"], best["B"])}}
                                  for h in history])), "",
           md_table(pd.DataFrame([{"street": k, "rows": v["rows"],
                                   **{MODEL_LABELS[m]: v[m] for m in MAIN_MODELS}} for k, v in by_street.items()])), ""]

    md += ["## 8. Predicted vs true action shares", "", md_table(pd.DataFrame(share_rows)), ""]

    if permutation:
        rows = []
        for m, per_set in permutation.items():
            for s, res in per_set.items():
                for g, v in res["groups"].items():
                    rows.append({"model": MODEL_LABELS[m], "set": s, "feature group": g, "columns": v["columns"],
                                 "macro-F1 drop": f"{v['macro_f1_drop_mean']:+.4f} ± {v['macro_f1_drop_std']:.4f}",
                                 "log-loss increase": f"{v['log_loss_increase_mean']:+.4f}"})
        md += ["## 9. Permutation importance by feature group", "",
               f"Each group's columns are shuffled together across rows ({PERMUTATION_REPEATS} repeats); the legal "
               "mask at prediction time still uses the true legal actions.", "", md_table(pd.DataFrame(rows)), ""]

    rows = []
    for s in REPORT_SETS:
        for k, v in diffs[s].items():
            if "shuffled" in k:
                rows.append({"set": s, "comparison": k, "macro-F1": fmt_diff(v["macro_f1"]),
                             "log-loss": fmt_diff(v["log_loss"])})
    md += ["## 10. Shuffled-label control (audit L13)", "",
           "Models trained with the selected settings on randomly permuted training labels. "
           "They should do no better than the Dummy baseline.", "", md_table(pd.DataFrame(rows)), ""]
    md += ["## Figures", "", *(f"- `{p}`" for p in figures.values()), "",
           f"Evaluation took {(time.perf_counter() - start) / 60:.1f} min.", ""]
    (RESULTS_DIR / "model_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
