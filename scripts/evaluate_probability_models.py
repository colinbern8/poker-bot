"""Probability-quality evaluation of the Phase 4 candidate model.

    python scripts/evaluate_probability_models.py

Compares the validation-log-loss winner (and the best HistGradientBoosting
setting by the same rule) with the Dummy baseline and the Stage 3
macro-F1-selected Feature Set B models. Nothing here is used to choose a
model: the choice was made on validation data by
scripts/select_probability_model.py.

Reports raw and legal-masked probabilities separately (the agent will use
the masked ones), reliability diagnostics, overfitting gaps, per-style and
per-class behaviour (including an investigation of Solid opponents and
the rare all-in class), with match-level bootstrap intervals on the test
(5 matches) and held-out (12 matches) sets.

Writes results/phase3/probability_model_report.md and
probability_metrics.json, and docs/figures/phase3/fig9-fig11.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hybridpoker.action_probabilities import CLASS_ORDER, mask_and_renormalize  # noqa: E402
from hybridpoker.calibration import (  # noqa: E402
    brier_score,
    reliability_table,
    calibration_in_the_large,
    class_reliability,
    one_vs_rest_ranking,
    top_label_reliability,
)
from hybridpoker.evaluation import evaluate, interval, match_stats, paired_difference, resample_weights  # noqa: E402
from hybridpoker.reporting import md_table  # noqa: E402
from hybridpoker.seeding import derive_seed  # noqa: E402

RESULTS = Path("results/phase3")
FIGURES = Path("docs/figures/phase3")
SETS = ("train", "validation", "test", "heldout")
REPORT_SETS = ("test", "heldout")
FAMILIES = ("calling_station", "maniac", "nit", "solid")
RESAMPLES = 2000

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRAY = "#8a8984"

# name -> (label, source file, column stem, color)
MODELS = {
    "dummy": ("Dummy (prior)", "predictions", "p_dummy_A", GRAY),
    "rf_mf1": ("RF-B, macro-F1 selected (Stage 3)", "predictions", "p_random_forest_B", SERIES[0]),
    "hgb_mf1": ("HGB-B, macro-F1 selected (Stage 3)", "predictions", "p_hist_gradient_boosting_B", SERIES[1]),
    "rf_ll": ("RF-B, log-loss selected (Phase 4 candidate)", "logloss", "p_random_forest_logloss_B", SERIES[2]),
    "hgb_ll": ("HGB-B, best log-loss setting", "logloss", "p_hist_gradient_boosting_logloss_B", SERIES[3]),
}
WINNER = "rf_ll"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(rows: int, cols: int, size):
    fig, axes = plt.subplots(rows, cols, figsize=size, facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        style(ax)
    return fig, axes


def brier_terms(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return np.sum((proba - np.eye(5)[y]) ** 2, axis=1)


def per_match_sums(values: np.ndarray, ids: np.ndarray, matches: tuple[str, ...]) -> np.ndarray:
    return np.array([values[ids == m].sum() for m in matches])


def main() -> None:
    base = pd.read_parquet("data/processed/phase3/predictions.parquet")
    extra = pd.read_parquet("data/processed/phase3/predictions_logloss.parquet")
    if not np.array_equal(base["row"].to_numpy(), extra["row"].to_numpy()):
        raise RuntimeError("Prediction files are not aligned row by row.")
    selection = json.loads((RESULTS / "probability_model_selection.json").read_text(encoding="utf-8"))
    sources = {"predictions": base, "logloss": extra}
    y = base["y"].to_numpy()
    legal = base[[f"legal_{c}" for c in CLASS_ORDER]].to_numpy(dtype=bool)
    sets = base["eval_set"].to_numpy()
    ids = base["id_match"].to_numpy()
    family = base["label_actor_family"].to_numpy()
    masks = {s: sets == s for s in SETS}
    evalrows = masks["test"] | masks["heldout"]

    probs = {}
    for key, (_, source, stem, _) in MODELS.items():
        raw = sources[source][[f"{stem}_{c}" for c in CLASS_ORDER]].to_numpy()
        m = mask_and_renormalize(raw, legal)
        probs[key] = {"raw": raw, "masked": m.masked, "illegal": m.illegal_mass}

    # 1. Headline metrics, raw and masked, every set.
    headline = []
    for key in MODELS:
        for s in SETS:
            r = masks[s]
            masked, raw = probs[key]["masked"][r], probs[key]["raw"][r]
            e = evaluate(y[r], masked)
            headline.append({
                "model": key, "set": s, "rows": int(r.sum()),
                "log_loss_masked": e["log_loss"], "log_loss_raw": evaluate(y[r], raw)["log_loss"],
                "brier_masked": brier_score(y[r], masked), "brier_raw": brier_score(y[r], raw),
                "ece_masked": top_label_reliability(y[r], masked)["ece"],
                "macro_f1_masked": e["macro_f1"], "accuracy_masked": e["accuracy"],
                "balanced_accuracy_masked": e["balanced_accuracy"],
                "raw_mass_on_illegal": float(probs[key]["illegal"][r].mean()),
            })
    headline_df = pd.DataFrame(headline)

    # 2. Bootstrap intervals and paired differences (log-loss, Brier, macro-F1).
    boot, paired = {}, {}
    for s in REPORT_SETS:
        r = masks[s]
        stats = {k: match_stats(y[r], probs[k]["masked"][r], ids[r]) for k in MODELS}
        matches = stats["dummy"].matches
        weights = resample_weights(len(matches), RESAMPLES, derive_seed(3000, "bootstrap_probability", s) % 2**32)
        counts = per_match_sums(np.ones(r.sum()), ids[r], matches)
        brier_sums = {k: per_match_sums(brier_terms(y[r], probs[k]["masked"][r]), ids[r], matches) for k in MODELS}
        boot[s] = {"matches": len(matches)}
        for k in MODELS:
            b = (weights @ brier_sums[k]) / (weights @ counts)
            ll = paired_difference(stats[k], stats[k], weights)  # zero-width; we only need point + CI below
            del ll
            ll_boot = (weights @ stats[k].loss_sum) / (weights @ stats[k].count)
            boot[s][k] = {"log_loss": {"value": float(stats[k].loss_sum.sum() / stats[k].count.sum()),
                                       **dict(zip(("ci_low", "ci_high"), interval(ll_boot)))},
                          "brier": {"value": float(brier_sums[k].sum() / counts.sum()),
                                    **dict(zip(("ci_low", "ci_high"), interval(b)))}}
        paired[s] = {}
        for other in ("dummy", "rf_mf1", "hgb_mf1", "hgb_ll"):
            d = paired_difference(stats[WINNER], stats[other], weights)
            bw = (weights @ brier_sums[WINNER]) / (weights @ counts) - (weights @ brier_sums[other]) / (weights @ counts)
            lo, hi = interval(bw)
            paired[s][other] = {"log_loss": d["log_loss"], "macro_f1": d["macro_f1"],
                                "brier": {"value": float((brier_sums[WINNER].sum() - brier_sums[other].sum()) / counts.sum()),
                                          "ci_low": lo, "ci_high": hi}}

    # 3. Calibration diagnostics.
    top_label = {k: {s: top_label_reliability(y[masks[s]], probs[k]["masked"][masks[s]]) for s in ("validation",)}
                 for k in MODELS}
    for k in MODELS:
        top_label[k]["test+heldout"] = top_label_reliability(y[evalrows], probs[k]["masked"][evalrows])
    class_cal = {s: class_reliability(y[r], probs[WINNER]["masked"][r]) for s, r in
                 (("validation", masks["validation"]), ("test+heldout", evalrows))}
    in_large = {k: {s: calibration_in_the_large(y[r], probs[k]["masked"][r]) for s, r in
                    (("validation", masks["validation"]), ("test", masks["test"]), ("heldout", masks["heldout"]))}
                for k in MODELS}
    ranking = {k: one_vs_rest_ranking(y[evalrows], probs[k]["masked"][evalrows]) for k in MODELS}

    # 4. Per style (log-loss), with a label-informed reference that is NOT deployable:
    # the training action frequencies of the acting player's style family (legal-masked).
    style_prior = {}
    for f in FAMILIES:
        rows = masks["train"] & (family == f)
        style_prior[f] = np.bincount(y[rows], minlength=5) / rows.sum()
    oracle_raw = np.stack([style_prior.get(f, np.full(5, 0.2)) for f in family])
    oracle = mask_and_renormalize(oracle_raw, legal).masked
    per_style = []
    for s in ("validation",) + REPORT_SETS:
        for f in FAMILIES:
            r = masks[s] & (family == f)
            row = {"set": s, "style": f, "rows": int(r.sum()), "matches": int(len(set(ids[r])))}
            for k in MODELS:
                row[k] = evaluate(y[r], probs[k]["masked"][r])["log_loss"]
            row["style_prior_reference"] = evaluate(y[r], oracle[r])["log_loss"]
            per_style.append(row)
    per_style_df = pd.DataFrame(per_style)

    # Solid: where does the loss come from? Mean -log p(true) by true class, and calibration in the large.
    solid_rows = evalrows & (family == "solid")
    solid_detail = []
    for c, cls in enumerate(CLASS_ORDER):
        r = solid_rows & (y == c)
        entry = {"true action": cls, "rows": int(r.sum())}
        for k in ("dummy", WINNER):
            entry[f"{k} mean -log p(true)"] = float(-np.log(np.maximum(probs[k]["masked"][r, c], 1e-15)).mean()) if r.any() else float("nan")
        entry[f"{WINNER} mean p(action) on solid rows"] = float(probs[WINNER]["masked"][solid_rows, c].mean())
        entry["observed share on solid rows"] = float(np.mean(y[solid_rows] == c))
        solid_detail.append(entry)

    # all-in: rows, where they occur, and how much probability the models give them.
    allin = []
    for s in ("train", "validation", "test", "heldout"):
        r = masks[s]
        positive = r & (y == 4)
        entry = {"set": s, "all_in rows": int(positive.sum()), "share": float(positive.sum() / r.sum())}
        for f in FAMILIES:
            entry[f] = int((positive & (family == f)).sum())
        allin.append(entry)
    allin_prob = []
    for k in MODELS:
        p = probs[k]["masked"][evalrows, 4]
        pos = y[evalrows] == 4
        allin_prob.append({"model": MODELS[k][0], "mean p(all_in) | all_in": float(p[pos].mean()),
                           "mean p(all_in) | other": float(p[~pos].mean()),
                           "AUC": ranking[k]["all_in"]["auc"], "average precision": ranking[k]["all_in"]["average_precision"],
                           "base rate": ranking[k]["all_in"]["base_rate"]})
    street = base["obs_street"].to_numpy()
    allin_street = pd.crosstab(street[(y == 4)], family[(y == 4)])

    # --- Figures ----------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    FIGURES.mkdir(parents=True, exist_ok=True)

    # fig9: top-label reliability (test + held-out), one panel per model.
    keys = ["dummy", "rf_mf1", "hgb_mf1", "rf_ll"]
    fig, axes = figure(1, 4, (14, 3.8))
    for ax, k in zip(axes.flat, keys):
        bins = top_label[k]["test+heldout"]["bins"]
        ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1)
        ax.plot([b["mean_predicted"] for b in bins], [b["observed"] for b in bins], color=MODELS[k][3],
                linewidth=2, marker="o", markersize=7)
        for b in bins:
            if b["rows"] >= 50:
                ax.annotate(str(b["rows"]), (b["mean_predicted"], b["observed"]), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=7, color=INK_2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence in predicted action")
        ax.set_title(f"{MODELS[k][0]}\nECE {top_label[k]['test+heldout']['ece']:.3f}", loc="left", fontsize=9)
    axes[0, 0].set_ylabel("share correct")
    fig.suptitle("Top-label reliability, test + held-out (legal-masked; dashed = perfect calibration; "
                 "labels = rows in bins with 50+ rows)", x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_reliability.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # fig10: per-class reliability of the candidate (test + held-out), quantile bins so that
    # rare actions (all_in probabilities are all below 0.1) still get several bins; each panel is
    # zoomed to its own probability range.
    quantile_tables = {}
    fig, axes = figure(1, 5, (16, 3.6))
    for c, (ax, cls) in enumerate(zip(axes.flat, CLASS_ORDER)):
        rows_c = evalrows & legal[:, c]  # only rows where this action was possible
        p = probs[WINNER]["masked"][rows_c, c]
        outcome = y[rows_c] == c
        k = 5 if cls == "all_in" else 10
        edges = np.unique(np.quantile(p, np.linspace(0, 1, k + 1)))
        table = reliability_table(p, outcome, edges)
        quantile_tables[cls] = table
        top = max(max(b["mean_predicted"] for b in table), max(b["observed"] for b in table)) * 1.12
        ax.plot([0, top], [0, top], color=INK_2, linestyle="--", linewidth=1)
        ax.plot([b["mean_predicted"] for b in table], [b["observed"] for b in table], color=SERIES[c],
                linewidth=2, marker="o", markersize=7)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel(f"predicted P({cls})")
        ax.set_title(f"{cls} ({int(outcome.sum())} of {int(rows_c.sum())} legal rows)", loc="left", fontsize=9)
    axes[0, 0].set_ylabel("observed frequency")
    fig.suptitle("Per-action reliability of the log-loss-selected RF-B, test + held-out "
                 "(rows where the action was legal; equal-count bins: 10 per action, 5 for all_in; zoomed)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_class_reliability.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # fig11: log-loss and Brier with bootstrap intervals.
    fig, axes = figure(2, 2, (12, 6))
    order = list(MODELS)[::-1]
    for r_i, metric in enumerate(("log_loss", "brier")):
        for c_i, s in enumerate(REPORT_SETS):
            ax = axes[r_i, c_i]
            ax.grid(axis="y", visible=False)
            for k_i, k in enumerate(order):
                e = boot[s][k][metric]
                ax.errorbar(e["value"], k_i, xerr=[[e["value"] - e["ci_low"]], [e["ci_high"] - e["value"]]], fmt="o",
                            color=MODELS[k][3], ecolor=MODELS[k][3], elinewidth=2, capsize=3, markersize=8)
                ax.annotate(f"{e['value']:.3f}", (e["ci_high"], k_i), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK_2)
            ax.set_yticks(range(len(order)), [MODELS[k][0] for k in order], fontsize=8)
            nice = "log-loss" if metric == "log_loss" else "Brier score"
            ax.set_title(f"{s.replace('heldout', 'held-out')} set ({boot[s]['matches']} matches): {nice} "
                         "(lower is better)", loc="left", fontsize=10)
    fig.suptitle("Probability quality, legal-masked, 95% match-level bootstrap intervals",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_probability_quality.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # --- Report -----------------------------------------------------------------------------
    win = selection["winner"]
    label = {k: v[0] for k, v in MODELS.items()}

    def ci(e):
        return f"{e['value']:.3f} [{e['ci_low']:.3f}, {e['ci_high']:.3f}]"

    def dci(e):
        return f"{e['value']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]"

    md = ["# Phase 4 probability model: selection and calibration", "",
          "**Decision context.** The Stage 3 classification models were selected by validation macro-F1 and were "
          "poorly calibrated. *After inspecting those results*, the project owner decided that the model feeding the "
          "Phase 4 agent is selected by validation log-loss instead. Stage 3 models and results are unchanged. "
          "Test and held-out data were not used for this choice; they are reported here for evaluation only.", "",
          "## 1. Selection (validation only)", "",
          f"- Rule: {selection['rule']}.",
          f"- Candidates: {selection['candidates']} Feature Set B settings.",
          f"- **Winner: Random Forest, grid #{win['grid_index']}: {json.dumps(win['params'])}**, seed {win['seed']}.",
          f"- Validation log-loss {win['val_log_loss']:.4f} (masked), macro-F1 {win['val_macro_f1']:.3f}, "
          f"accuracy {win['val_accuracy']:.3f}. Refit reproduced the grid value exactly.",
          f"- Best HistGradientBoosting B setting: grid #{selection['families']['hist_gradient_boosting']['grid_index']} "
          f"{json.dumps(selection['families']['hist_gradient_boosting']['params'])}, validation log-loss "
          f"{selection['families']['hist_gradient_boosting']['val_log_loss']:.4f}.",
          f"- RF minus HGB on validation (5 matches, paired bootstrap): "
          f"{dci(selection['validation_rf_minus_hgb']['log_loss'])} log-loss.",
          f"- Runner-up margin {selection['runner_up_margin_val_log_loss']:.4f}; the five lowest validation log-losses "
          "are all Random Forest settings within 0.002 of each other, so the choice is not a single lucky setting.", "",
          md_table(pd.DataFrame(selection["top5"]), "{:.4f}"), ""]

    def table_for(columns, sets_):
        part = headline_df[headline_df["set"].isin(sets_)].copy()
        part["model"] = part["model"].map(label)
        return md_table(part[["model", "set", *columns]], "{:.3f}")

    md += ["## 2. Probability quality on every set (point estimates)", "",
           "`masked` = legal-action-masked and renormalized (what the agent uses; primary). `raw` = the model's own "
           "probabilities over all five actions. ECE = top-label expected calibration error (10 bins).", "",
           table_for(["rows", "log_loss_masked", "log_loss_raw", "brier_masked", "brier_raw", "ece_masked",
                      "raw_mass_on_illegal"], SETS), "",
           "Classification metrics for the same models (legal-masked):", "",
           table_for(["macro_f1_masked", "balanced_accuracy_masked", "accuracy_masked"], SETS), ""]

    for s in REPORT_SETS:
        rows = [{"model": label[k], "log-loss": ci(boot[s][k]["log_loss"]), "Brier": ci(boot[s][k]["brier"])}
                for k in MODELS]
        md += [f"## 3{'a' if s == 'test' else 'b'}. {s.replace('heldout', 'Held-out')} set "
               f"({boot[s]['matches']} matches), 95% bootstrap intervals", "", md_table(pd.DataFrame(rows)), "",
               "Paired differences, candidate minus comparison (negative log-loss/Brier = candidate better):", "",
               md_table(pd.DataFrame([{"comparison": f"candidate - {label[o]}", "log-loss": dci(v["log_loss"]),
                                       "Brier": dci(v["brier"]), "macro-F1": dci(v["macro_f1"])}
                                      for o, v in paired[s].items()])), ""]

    rows = [{"model": label[k], "ECE validation": top_label[k]["validation"]["ece"],
             "ECE test+held-out": top_label[k]["test+heldout"]["ece"]} for k in MODELS]
    md += ["## 4. Calibration diagnostics", "", "### 4a. Top-label expected calibration error", "",
           md_table(pd.DataFrame(rows)), "",
           "Reliability bins of the candidate, test + held-out:", "",
           md_table(pd.DataFrame(top_label[WINNER]["test+heldout"]["bins"])), ""]
    rows = []
    for s in ("validation", "test+heldout"):
        for cls in CLASS_ORDER:
            t = class_cal[s][cls]
            rows.append({"set": s, "action": cls, "occurrences": t["positives"], "per-class ECE": t["ece"]})
    md += ["### 4b. Per-action calibration of the candidate", "",
           "Per-class ECE compares predicted P(action) with how often the action occurred, in 10 probability bins. "
           "With few occurrences (all_in) the bins are sparse and the numbers are rough.", "",
           md_table(pd.DataFrame(rows)), ""]
    md += ["Equal-count reliability bins of the candidate (test + held-out, rows where the action was legal) "
           "for raise_pot and all_in:", ""]
    for cls in ("raise_pot", "all_in"):
        md += [f"**{cls}**", "", md_table(pd.DataFrame(quantile_tables[cls]), "{:.4f}"), ""]
    rows = []
    for k in MODELS:
        for s in ("validation", "test", "heldout"):
            entry = {"model": label[k], "set": s}
            for cls in CLASS_ORDER:
                v = in_large[k][s][cls]
                entry[cls] = f"{v['mean_predicted']:.3f} / {v['observed']:.3f}"
            rows.append(entry)
    md += ["### 4c. Calibration in the large (mean predicted / observed share)", "", md_table(pd.DataFrame(rows)), ""]
    rows = [{"model": label[k], **{cls: (f"{ranking[k][cls]['auc']:.3f}" if ranking[k][cls]["auc"] is not None else "n/a")
                                   for cls in CLASS_ORDER}} for k in MODELS]
    md += ["### 4d. One-vs-rest AUC (test + held-out)", "",
           "How well each action's probability ranks rows where it happened above rows where it did not "
           "(0.5 = no ranking ability). The Dummy gives every row with the same legal actions the same probabilities.", "",
           md_table(pd.DataFrame(rows)), ""]

    rows = []
    for k in MODELS:
        h = headline_df[headline_df["model"] == k].set_index("set")
        rows.append({"model": label[k], "train log-loss": h.loc["train", "log_loss_masked"],
                     "validation": h.loc["validation", "log_loss_masked"], "test": h.loc["test", "log_loss_masked"],
                     "held-out": h.loc["heldout", "log_loss_masked"],
                     "gap validation - train": h.loc["validation", "log_loss_masked"] - h.loc["train", "log_loss_masked"],
                     "train macro-F1": h.loc["train", "macro_f1_masked"],
                     "validation macro-F1": h.loc["validation", "macro_f1_masked"]})
    md += ["## 5. Overfitting check", "",
           "Train vs unseen-data log-loss. A large gap means the model memorizes training rows.", "",
           md_table(pd.DataFrame(rows)), ""]

    show = per_style_df.copy()
    show = show.rename(columns={**label, "style_prior_reference": "style-prior reference (uses labels; not deployable)"})
    md += ["## 6. Per-style log-loss (legal-masked)", "",
           "The style-prior reference predicts each row from the training action frequencies of the acting player's "
           "*true* style family. It uses labels and cannot be deployed; it shows how much knowing the style alone "
           "would give. This investigation is descriptive only: nothing here changes the selected model.", "",
           md_table(show, "{:.3f}"), "",
           "### 6a. Solid opponents (test + held-out): where the loss comes from", "",
           md_table(pd.DataFrame(solid_detail), "{:.3f}"), ""]
    md += ["## 7. The rare all-in class", "", "Where all-in rows occur (rows used):", "",
           md_table(pd.DataFrame(allin), "{:.4f}"), "",
           "All-in rows by street and family (all sets):", "",
           md_table(allin_street.reset_index().rename(columns={"row_0": "street", "index": "street"})), "",
           "Probability given to all-in (test + held-out, legal-masked):", "",
           md_table(pd.DataFrame(allin_prob), "{:.4f}"), ""]
    md += ["## Figures", "", "- `docs/figures/phase3/fig9_reliability.png`",
           "- `docs/figures/phase3/fig10_class_reliability.png`",
           "- `docs/figures/phase3/fig11_probability_quality.png`", ""]
    (RESULTS / "probability_model_report.md").write_text("\n".join(md), encoding="utf-8")
    out = {"selection": selection, "headline": headline, "bootstrap": boot, "paired": paired,
           "top_label": top_label, "class_calibration": class_cal, "in_the_large": in_large, "ranking": ranking,
           "per_style": per_style, "solid_detail": solid_detail, "class_reliability_quantile": quantile_tables, "all_in": allin, "all_in_probability": allin_prob}
    (RESULTS / "probability_metrics.json").write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
