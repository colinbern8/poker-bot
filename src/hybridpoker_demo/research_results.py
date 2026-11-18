"""The read-only research results panel: the frozen Phase 4 evaluation, verified before it is shown.

Every number comes from ``results/phase4/stage4d_final_summary.json``, the
frozen output of the Phase 4 final evaluation. Nothing here recomputes,
rounds differently from the report, or stores a research result:

* **Integrity first.** The file is read only if its SHA-256 (with CRLF line
  endings normalized to LF, exactly as ``tests/test_phase4_freeze_guard.py``
  hashes it) equals the pinned value. That value is the freeze guard's own
  entry for this file; a test checks that the two stay identical.
* **Consistency.** The extracted values must agree with each other (for
  example, the primary difference must equal the difference of the two
  agents' win rates, and every agent must have played the same number of
  hands). Any missing key or disagreement counts as a failed verification.
* **Fail closed.** If verification fails, :func:`research_panel` returns an
  "unavailable" panel that contains no figures at all.

The one derived figure, the percentage, is the primary difference divided by
the baseline's average win rate, and it is always labelled as derived.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybridpoker.state_coverage import LOW_COVERAGE_THRESHOLD

SUMMARY_RELATIVE_PATH = Path("results/phase4/stage4d_final_summary.json")
# The freeze guard's pinned hash for the summary file (tests/test_phase4_freeze_guard.py).
SUMMARY_SHA256 = "abd2c8b0ed8d7c59210473d0e5d68c16a648788936b423868bb9ab9aa9673864"

PRIMARY_ENDPOINT = "ML EV minus Situation-Aware Frequency EV"
ML_AGENT, BASELINE_AGENT = "ev_ml", "ev_situation"
ALL_OPPONENTS, HELD_OUT = "all 16", "held-out"
SEATS_PER_DEAL = 2  # each deal is played twice with the seats swapped
TITLE = "Research results (Phase 4 evaluation, simulated)"
UNAVAILABLE = ("Research results are unavailable: the frozen results file could not be verified, "
               "so no figures are shown.")


class ResearchUnavailable(RuntimeError):
    """The frozen results file is missing, altered, or inconsistent."""


def normalized_sha256(path: Path) -> str:
    """SHA-256 with CRLF normalized to LF (the freeze guard's definition, so a Windows checkout also passes)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def summary_path(root: Path | None = None) -> Path:
    if root is None:
        from hybridpoker_demo.resources import repo_root

        root = repo_root()
    return root / SUMMARY_RELATIVE_PATH


@dataclass(frozen=True)
class ResearchSummary:
    """The figures the panel shows, exactly as recorded in the frozen summary."""

    primary_bb100: float      # ML EV minus the baseline, bb/100
    ci_low: float             # 95% confidence interval of the primary difference
    ci_high: float
    baseline_bb100: float     # the baseline's average win rate over all 16 opponents
    derived_percent: float    # primary_bb100 / baseline_bb100 (derived, not a tested endpoint)
    agents: int
    hands_per_agent: int
    total_hands: int
    opponents: int
    held_out_opponents: int
    ml_all_in_share: float           # share of the ML agent's decisions that were all-in
    ml_low_coverage_share: float     # share of its decisions whose chosen action was low-coverage
    ml_low_coverage_all_in_share: float  # the same, among its all-ins
    low_coverage_threshold: int


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6)


def summary_from_data(data: dict[str, Any]) -> ResearchSummary:
    """Extract and cross-check the panel's figures; raise ResearchUnavailable if anything is off."""
    try:
        endpoint = data["endpoints"][PRIMARY_ENDPOINT]
        estimate = endpoint["estimate"]
        ml = data["bb100_groups"][ML_AGENT]
        baseline = data["bb100_groups"][BASELINE_AGENT]
        deals, opponents = int(data["deals"]), len(data["opponents"])
        hands = {agent: int(record["hands"]) for agent, record in data["all_in"].items()}
        coverage = data["coverage"][ML_AGENT]
        summary = ResearchSummary(
            primary_bb100=float(estimate["mean"]), ci_low=float(estimate["low"]), ci_high=float(estimate["high"]),
            baseline_bb100=float(baseline[ALL_OPPONENTS]["mean"]),
            derived_percent=float(estimate["mean"]) / float(baseline[ALL_OPPONENTS]["mean"]),
            agents=len(hands), hands_per_agent=hands[ML_AGENT], total_hands=sum(hands.values()),
            opponents=opponents, held_out_opponents=int(ml[HELD_OUT]["units"]) // deals,
            ml_all_in_share=float(data["all_in"][ML_AGENT]["all_in_action_share"]),
            ml_low_coverage_share=float(coverage["chosen_low_share_of_decisions"]),
            ml_low_coverage_all_in_share=float(coverage["chosen_low_share_of_all_ins"]),
            low_coverage_threshold=LOW_COVERAGE_THRESHOLD,
        )
        checks = [
            endpoint["role"] == "primary",
            set(hands) == set(data["bb100"]) and {ML_AGENT, BASELINE_AGENT} <= set(hands),
            len(set(hands.values())) == 1,                                  # every agent played the same hands
            summary.hands_per_agent == opponents * deals * SEATS_PER_DEAL,  # 16 opponents x 500 deals x 2 seats
            _close(summary.primary_bb100, float(ml[ALL_OPPONENTS]["mean"]) - summary.baseline_bb100),
            summary.ci_low <= summary.primary_bb100 <= summary.ci_high,
            int(ml[HELD_OUT]["units"]) % deals == 0 and 0 < summary.held_out_opponents < opponents,
            summary.baseline_bb100 > 0,
            all(0.0 <= share <= 1.0 for share in (summary.ml_all_in_share, summary.ml_low_coverage_share,
                                                  summary.ml_low_coverage_all_in_share)),
        ]
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ResearchUnavailable(f"unexpected summary structure: {type(error).__name__}") from None
    if not all(checks):
        raise ResearchUnavailable("the summary's figures are not consistent with each other")
    return summary


def load_research_summary(root: Path | None = None) -> ResearchSummary:
    """Read the frozen summary, verify its SHA-256, and extract the figures (read-only)."""
    path = summary_path(root)
    if not path.is_file():
        raise ResearchUnavailable(f"{SUMMARY_RELATIVE_PATH} is missing")
    if normalized_sha256(path) != SUMMARY_SHA256:
        raise ResearchUnavailable(f"{SUMMARY_RELATIVE_PATH} does not match the frozen SHA-256")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise ResearchUnavailable(f"{SUMMARY_RELATIVE_PATH} is not valid JSON") from None
    return summary_from_data(data)


# --- Display text ---------------------------------------------------------------------------------


def _signed(value: float) -> str:
    """One decimal with a sign, as in the frozen report ("+84.8")."""
    return f"{value:+.1f}"


def _percent(share: float) -> str:
    return f"{share:.1%}"


@dataclass(frozen=True)
class ResearchPanel:
    """What the collapsed panel shows: either verified figures or a plain "unavailable" message."""

    available: bool
    primary: str = ""             # "+84.8 bb/100"
    interval: str = ""            # "95% confidence interval [+51.6, +117.9]"
    meaning: str = ""             # bb/100 in plain English and what the result means
    derived: str = ""             # the derived percentage, labelled as derived
    scale: str = ""               # hands, agents, opponents
    baseline: str = ""            # what the baseline is
    limitations: tuple[str, ...] = ()
    source: str = ""
    message: str = ""             # the unavailable message
    reason: str = ""              # why verification failed (for the developer log; no figures)


def research_panel(root: Path | None = None) -> ResearchPanel:
    """The panel's text. Never raises: a failed verification gives an unavailable panel with no figures."""
    try:
        s = load_research_summary(root)
    except ResearchUnavailable as error:
        return ResearchPanel(available=False, message=UNAVAILABLE, reason=str(error))
    primary = _signed(s.primary_bb100)
    above_zero = s.ci_low > 0
    meaning = ("bb/100 means big blinds won per 100 hands. Over the simulated games, the Hybrid ML Agent won "
               f"{primary.lstrip('+')} more big blinds per 100 hands than the pre-specified statistical baseline. "
               + ("The whole 95% confidence interval is above zero, so this advantage is statistically "
                  "significant in the simulation." if above_zero else
                  "The 95% confidence interval includes zero, so this difference is not statistically "
                  "significant."))
    derived = (f"Derived figure: about {s.derived_percent:.1%} higher average profitability than the baseline "
               f"({primary.lstrip('+')} ÷ the baseline's average of {s.baseline_bb100:.1f} bb/100). "
               "This percentage is calculated from the results above; it was not a separately tested result.")
    scale = (f"{s.total_hands:,} simulated hands: {s.agents} agents × {s.hands_per_agent:,} hands each. Every agent "
             f"played the same deals against the same {s.opponents} scripted opponents "
             f"({s.held_out_opponents} of them held out from training).")
    baseline = ("The baseline, Situation-Aware Frequency EV, was chosen before the evaluation. It uses the same "
                "one-step expected-value decision rule as the Hybrid ML Agent, but estimates the opponent's "
                "responses from situation-based frequencies in the training data (how often players folded, "
                "called or raised in each betting situation) instead of from the trained Random Forest. It does "
                "not use the opponent's history or other details of the hand.")
    limitations = (
        "The opponents were scripted bots, not people; the held-out bots are variations of the training styles.",
        "It has not been shown that these results carry over to human players.",
        "The AI looks only one decision ahead (one-step estimated value): it does not plan later betting rounds, "
        f"and it chose the all-in action in {_percent(s.ml_all_in_share)} of its decisions.",
        f"Some decisions rely on limited training examples: in {_percent(s.ml_low_coverage_share)} of all the ML "
        f"agent's decisions, and in {_percent(s.ml_low_coverage_all_in_share)} of the decisions where it chose "
        f"the all-in action, the chosen action led to a situation with fewer than {s.low_coverage_threshold} "
        "similar training examples.",
    )
    source = (f"Source: {SUMMARY_RELATIVE_PATH.as_posix()} (frozen Phase 4 final evaluation; SHA-256 verified). "
              "Demonstration games are not research evidence.")
    return ResearchPanel(available=True, primary=f"{primary} bb/100",
                         interval=f"95% confidence interval [{_signed(s.ci_low)}, {_signed(s.ci_high)}]",
                         meaning=meaning, derived=derived, scale=scale, baseline=baseline,
                         limitations=limitations, source=source)
