"""The research results panel: figures come from the frozen Phase 4 summary, verified, or not at all.

The screen placement (collapsed, last on the page) is tested in ``test_demo_app.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybridpoker_demo import research_results as rr
from test_phase4_freeze_guard import RESULT_SHA256
from test_phase4_freeze_guard import normalized_sha256 as guard_sha256

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / rr.SUMMARY_RELATIVE_PATH
REPORT = ROOT / "results/phase4/stage4d_final_report.md"
TEXT_FIELDS = ("primary", "interval", "meaning", "derived", "scale", "baseline", "source")


def frozen() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def panel_text(panel: rr.ResearchPanel) -> str:
    return " ".join([*(getattr(panel, f) for f in TEXT_FIELDS), *panel.limitations])


def root_with_summary(tmp_path: Path, content: bytes | None) -> Path:
    """A stand-in repository root holding ``content`` as the summary file (None: no file)."""
    target = tmp_path / rr.SUMMARY_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    if content is not None:
        target.write_bytes(content)
    return tmp_path


# --- integrity ---------------------------------------------------------------------------------------


def test_the_pinned_hash_is_the_freeze_guards_hash_for_the_same_file() -> None:
    assert rr.SUMMARY_SHA256 == RESULT_SHA256[SUMMARY.name]
    assert rr.normalized_sha256(SUMMARY) == guard_sha256(SUMMARY) == rr.SUMMARY_SHA256


def test_loading_reads_but_never_changes_the_frozen_file() -> None:
    before = (SUMMARY.read_bytes(), SUMMARY.stat().st_mtime_ns)
    rr.load_research_summary(ROOT)
    rr.research_panel(ROOT)
    assert (SUMMARY.read_bytes(), SUMMARY.stat().st_mtime_ns) == before


@pytest.mark.parametrize("content", [
    None,                                                                   # missing
    SUMMARY.read_bytes().replace(b'"mean": 84.75625', b'"mean": 94.75625'),  # one figure changed
    SUMMARY.read_bytes() + b" ",                                            # any other change
    b"{}",
], ids=["missing", "figure-changed", "trailing-byte", "empty-json"])
def test_failed_verification_shows_no_figures(tmp_path, content) -> None:
    assert content != SUMMARY.read_bytes()  # each case really differs from the frozen file
    panel = rr.research_panel(root_with_summary(tmp_path, content))
    assert not panel.available
    assert panel.message == rr.UNAVAILABLE and panel.reason
    assert panel_text(panel).strip() == ""  # nothing but the message
    assert not any(ch.isdigit() for ch in panel.message)


def test_windows_line_endings_still_verify(tmp_path) -> None:
    crlf = SUMMARY.read_bytes().replace(b"\n", b"\r\n")
    panel = rr.research_panel(root_with_summary(tmp_path, crlf))
    assert panel.available and panel == rr.research_panel(ROOT)


@pytest.mark.parametrize("change", [
    lambda d: d["endpoints"].pop(rr.PRIMARY_ENDPOINT),
    lambda d: d["endpoints"][rr.PRIMARY_ENDPOINT].update(role="secondary"),
    lambda d: d["endpoints"][rr.PRIMARY_ENDPOINT]["estimate"].update(mean=90.0),   # no longer ML minus baseline
    lambda d: d["endpoints"][rr.PRIMARY_ENDPOINT]["estimate"].update(low=90.0),    # interval misses the estimate
    lambda d: d["all_in"]["ev_ml"].update(hands=15999),                            # agents played different hands
    lambda d: d["all_in"].pop("probability"),                                      # an agent missing
    lambda d: d["coverage"]["ev_ml"].update(chosen_low_share_of_decisions=1.5),
    lambda d: d.update(deals="many"),
], ids=["no-primary", "not-primary", "inconsistent-mean", "interval", "hands", "agents", "share", "type"])
def test_inconsistent_figures_are_rejected(change) -> None:
    data = frozen()
    rr.summary_from_data(data)  # the frozen data itself is consistent
    change(data)
    with pytest.raises(rr.ResearchUnavailable):
        rr.summary_from_data(data)


# --- figures and wording ---------------------------------------------------------------------------------


def test_every_figure_is_read_from_the_frozen_summary() -> None:
    s, d = rr.load_research_summary(ROOT), frozen()
    estimate = d["endpoints"][rr.PRIMARY_ENDPOINT]["estimate"]
    assert (s.primary_bb100, s.ci_low, s.ci_high) == (estimate["mean"], estimate["low"], estimate["high"])
    assert s.baseline_bb100 == d["bb100_groups"]["ev_situation"]["all 16"]["mean"]
    assert s.derived_percent == estimate["mean"] / s.baseline_bb100
    assert s.agents == len(d["all_in"]) and s.hands_per_agent == d["all_in"]["ev_ml"]["hands"]
    assert s.total_hands == sum(a["hands"] for a in d["all_in"].values())
    assert s.opponents == len(d["opponents"])
    assert s.held_out_opponents == d["bb100_groups"]["ev_ml"]["held-out"]["units"] // d["deals"]
    assert s.ml_all_in_share == d["all_in"]["ev_ml"]["all_in_action_share"]
    assert s.ml_low_coverage_share == d["coverage"]["ev_ml"]["chosen_low_share_of_decisions"]
    assert s.ml_low_coverage_all_in_share == d["coverage"]["ev_ml"]["chosen_low_share_of_all_ins"]


def test_the_panel_shows_the_approved_figures() -> None:
    """The figures approved for the presentation, as the frozen file produces them."""
    p = rr.research_panel(ROOT)
    assert p.available
    assert p.primary == "+84.8 bb/100"
    assert p.interval == "95% confidence interval [+51.6, +117.9]"
    assert p.derived.startswith("Derived figure: about 11.2% higher average profitability than the baseline")
    assert "not a separately tested result" in p.derived
    assert p.scale.startswith("64,000 simulated hands: 4 agents × 16,000 hands each.")
    assert "big blinds won per 100 hands" in p.meaning and "statistically significant" in p.meaning


def test_figures_agree_with_the_frozen_report() -> None:
    """The panel's rounded figures are the ones printed in the frozen final report."""
    report = REPORT.read_text(encoding="utf-8")
    p, s = rr.research_panel(ROOT), rr.load_research_summary(ROOT)
    assert f"{p.primary}, {p.interval.replace('95% confidence interval', '95% CI')}" in report
    section = report.split("## 4.")[1].split("\n## ")[0]
    ml_row = next(line for line in section.splitlines() if line.startswith("| ML EV |"))
    cells = [c.strip() for c in ml_row.strip().strip("|").split("|")]
    assert cells[2] == f"{s.ml_low_coverage_share:.1%}" and cells[4] == f"{s.ml_low_coverage_all_in_share:.1%}"
    all_in_row = next(line for line in report.splitlines()
                      if line.startswith("| ML EV |") and f"/ {s.hands_per_agent}" in line)
    assert all_in_row.split("|")[2].strip() == f"{s.ml_all_in_share:.1%}"


def test_baseline_and_limitations_are_stated_plainly() -> None:
    p = rr.research_panel(ROOT)
    assert "same one-step expected-value decision rule" in p.baseline
    assert "situation-based frequencies" in p.baseline and "instead of from the trained Random Forest" in p.baseline
    assert "human" not in p.baseline.lower() and "average player" not in p.baseline.lower()
    assert len(p.limitations) == 4
    scripted, humans, one_step, coverage = p.limitations
    assert "scripted bots" in scripted
    assert "human players" in humans
    assert "one-step estimated value" in one_step
    assert f"fewer than {rr.LOW_COVERAGE_THRESHOLD} similar training examples" in coverage


def test_each_percentage_names_what_it_is_a_share_of() -> None:
    """The all-in share counts only the all-in action; the two coverage shares have distinct denominators:
    all of the ML agent's decisions, and only the decisions where it chose the all-in action."""
    s, p = rr.load_research_summary(ROOT), rr.research_panel(ROOT)
    _, _, one_step, coverage = p.limitations
    assert f"it chose the all-in action in {s.ml_all_in_share:.1%} of its decisions" in one_step
    assert f"in {s.ml_low_coverage_share:.1%} of all the ML agent's decisions" in coverage
    assert f"in {s.ml_low_coverage_all_in_share:.1%} of the decisions where it chose the all-in action" in coverage
    assert "of its all-ins" not in coverage and "went all-in" not in one_step
    # The all-in share is the all-in *action* (category), not every action that put the whole stack in.
    d = frozen()["all_in"]["ev_ml"]
    assert s.ml_all_in_share == d["all_in_action_share"] != d["all_in_any_share"]


def test_secondary_endpoints_are_left_out() -> None:
    d, text = frozen(), panel_text(rr.research_panel(ROOT))
    for name, endpoint in d["endpoints"].items():
        if endpoint["role"] != "primary":
            assert f"{endpoint['estimate']['mean']:+.1f}" not in text, name
    assert "Global Frequency" not in text and "ProbabilityAgent" not in text
