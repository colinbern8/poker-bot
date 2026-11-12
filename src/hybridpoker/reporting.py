"""Descriptive tables for the Phase 3 dataset reports.

These functions only *describe* a loaded dataset (counts, shares, legal-
action availability). They never change data or build model features.
Labels such as ``label_actor_family`` are used only for grouping in
reports.
"""

from __future__ import annotations

import pandas as pd

from hybridpoker.action_probabilities import CLASS_ORDER

EVALUATION_SET_ORDER = ("train", "validation", "test", "heldout")


def md_table(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """A GitHub-flavoured Markdown table (no extra dependency needed)."""

    def cell(value) -> str:
        if isinstance(value, float):
            return floatfmt.format(value)
        return str(value)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def evaluation_set(frame: pd.DataFrame) -> pd.Series:
    """Which evaluation set each row belongs to, or ``None`` if it is not used.

    Train, validation and test use every row. In held-out matches only the
    held-out player's (scored) rows count; the counterpart's rows are unused.
    """
    sets = frame["id_split"].where(frame["id_split"] != "heldout")
    heldout_scored = (frame["id_split"] == "heldout") & frame["id_scored"]
    return sets.mask(heldout_scored, "heldout")


def class_counts(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Rows and count of each action class, per group (or overall)."""
    groups = [("all", frame)] if by is None else list(frame.groupby(by, sort=False))
    rows = []
    for name, part in groups:
        counts = part["act_category"].value_counts()
        rows.append({"group": name, "rows": len(part), **{c: int(counts.get(c, 0)) for c in CLASS_ORDER}})
    return pd.DataFrame(rows)


def class_shares(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Like :func:`class_counts` but each class as a share of the group's rows."""
    counts = class_counts(frame, by)
    shares = counts.copy()
    for c in CLASS_ORDER:
        shares[c] = counts[c] / counts["rows"]
    return shares


def format_counts_with_shares(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Counts with shares in brackets, e.g. ``34 (0.9%)``, for report tables."""
    counts = class_counts(frame, by)
    out = counts[["group", "rows"]].copy()
    for c in CLASS_ORDER:
        out[c] = [f"{n} ({n / r:.1%})" for n, r in zip(counts[c], counts["rows"])]
    return out


def legal_conditioned(frame: pd.DataFrame, category: str, by: str | None = None) -> pd.DataFrame:
    """How often an action was available, and how often it was chosen when available.

    ``legal share`` is the fraction of decisions where the action was legal;
    ``chosen | legal`` is the fraction of those legal decisions where it
    was chosen. Choosing an illegal action is impossible, which is checked.
    """
    if category not in CLASS_ORDER:
        raise ValueError(f"Unknown action {category!r}.")
    legal_column = f"obs_legal_{category}"
    groups = [("all", frame)] if by is None else list(frame.groupby(by, sort=False))
    rows = []
    for name, part in groups:
        legal = part[legal_column].astype(bool)
        chosen = part["act_category"] == category
        if (chosen & ~legal).any():
            raise ValueError(f"{category} was chosen while illegal.")
        legal_rows = int(legal.sum())
        rows.append({
            "group": name,
            "rows": len(part),
            "legal rows": legal_rows,
            "legal share": legal_rows / len(part) if len(part) else float("nan"),
            "chosen": int(chosen.sum()),
            "chosen | legal": chosen.sum() / legal_rows if legal_rows else float("nan"),
        })
    return pd.DataFrame(rows)


def share_differences(shares: pd.DataFrame, reference: str = "train") -> pd.DataFrame:
    """Each group's class shares minus the reference group's, in percentage points."""
    ref = shares.loc[shares["group"] == reference, list(CLASS_ORDER)]
    if len(ref) != 1:
        raise ValueError(f"Reference group {reference!r} not found exactly once.")
    diffs = shares[["group"]].copy()
    for c in CLASS_ORDER:
        diffs[c] = (shares[c] - ref[c].iloc[0]) * 100
    diffs["max |diff| (pp)"] = diffs[list(CLASS_ORDER)].abs().max(axis=1)
    return diffs


def per_match_share_range(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Min and max share of each class across the matches in each group.

    Shows how much class balance varies between matches, which matters more
    than row-level differences because matches are the independent units.
    """
    rows = []
    for name, part in frame.groupby(group_column, sort=False):
        per_match = class_shares(part, "id_match")
        row = {"group": name, "matches": len(per_match)}
        for c in CLASS_ORDER:
            row[c] = f"{per_match[c].min():.1%}-{per_match[c].max():.1%}"
        rows.append(row)
    return pd.DataFrame(rows)
