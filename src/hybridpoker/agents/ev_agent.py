"""The EV agents of Phase 4: one decision procedure, two opponent models.

Both agents are the same :class:`EVAgent`. They differ **only** in the
opponent probability model plugged in:

* :class:`FrequencyResponseModel` (the **Frequency EV Agent**, control):
  the global training-split action frequencies (``frequency_prior.json``),
  legal-masked in each hypothetical state. It is deliberately simple: it
  ignores the betting situation and the opponent's history, so it may be
  weaker than a context-aware frequency baseline would be.
* :class:`MLResponseModel` (the **ML EV Agent**, treatment): the frozen
  Phase 3 Random Forest (Feature Set B, setting #22) with live opponent
  history from :class:`~hybridpoker.opponent_tracker.OpponentTracker`.

Decision procedure, identical for both:

1. Equity against a random hand, with the same function and sample count
   (500) as the ProbabilityAgent.
2. For every legal bet/raise, the opponent's hypothetical observation after
   it (:mod:`hybridpoker.counterfactual`, public information only), and the
   opponent model's legal-masked response probabilities, in one batched call.
3. For every bet, the hero's equity **if called**: against the opponent's
   estimated continuing range, the strongest ``P(call) + P(re-raise)``
   fraction of its unblocked hands (:mod:`hybridpoker.range_equity`),
   estimated as the step-1 equity plus a low-noise correction from one
   shared simulation. The same function and sample count are used by both
   agents; only the predicted probabilities differ.
4. EV of every legal action with the approved equations (:mod:`hybridpoker.ev`).
5. The highest EV wins. Ties (within 1e-9 chips) go to the action committing
   **fewer additional chips**, then to the earlier action in ``CLASS_ORDER``.

Every decision is recorded in ``agent.diagnostics``: equity, all action EVs,
opponent probabilities (raw and masked), training coverage of each
hypothetical state, and timing (equity vs model inference). Coverage is
diagnostic only and never changes a decision.
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hybridpoker.action_probabilities import CLASS_ORDER, MaskedProbabilities, legal_mask, mask_and_renormalize
from hybridpoker.actions import RAISE_CATEGORIES, ActionCategory
from hybridpoker.agents.base import Agent, Decision
from hybridpoker.agents.strength import hand_strength
from hybridpoker.counterfactual import hypothetical_opponent_observation
from hybridpoker.ev import ActionValue, ResponseProbabilities, action_value
from hybridpoker.hand_log import observation_columns
from hybridpoker.observation import Observation, PublicHandSummary
from hybridpoker.opponent_history import HistoryPriors
from hybridpoker.opponent_predictor import PHASE4_MODEL_PATH, OpponentPredictor
from hybridpoker.opponent_tracker import OpponentTracker
from hybridpoker.range_equity import RANGE_EQUITY_SAMPLES, opponent_ranking, range_adjustments, top_count
from hybridpoker.state_coverage import COVERAGE_PATH, LOW_COVERAGE_THRESHOLD, coverage_key, load_coverage

FREQUENCY_PRIOR_PATH = Path("results/phase3/frequency_prior.json")
FROZEN_MODEL_SHA256 = "5d4c8235e142a2fdd23a2b370def8183aabfad72b7998120cbe4fb711dda7b31"
EQUITY_SAMPLES = 500
TIE_TOLERANCE = 1e-9

EquityFunction = Callable[[Observation, int, random.Random], float]
# (observation, equity vs a random hand, continuing fractions, samples, rng) -> equity if called, per fraction
CalledEquityFunction = Callable[[Observation, float, list, int, random.Random], list]


def called_equities(observation: Observation, equity: float, fractions: list, samples: int,
                    rng: random.Random) -> list:
    """Hero equity if each bet is called: against the strongest ``fraction`` of hands.

    The equity against a random hand (the same estimate the ProbabilityAgent uses) is
    corrected by the low-noise difference "range minus all hands" from
    :func:`hybridpoker.range_equity.range_adjustments` (one shared run for all bets).
    A fraction of 0 (the opponent never continues) gives the called branch no weight,
    so the plain equity is returned; a range covering every hand needs no correction.
    """
    positive = sorted({f for f in fractions if f > 0.0})
    if not positive:
        return [equity for _ in fractions]
    ranking = opponent_ranking(tuple(observation.hole_cards), tuple(observation.board_cards))
    if all(top_count(ranking, f) >= ranking.size for f in positive):
        return [equity for _ in fractions]
    adjustments = range_adjustments(observation.hole_cards, observation.board_cards, positive, samples,
                                    seed=rng.getrandbits(63))
    return [equity if f <= 0.0 else min(1.0, max(0.0, equity + adjustments[f])) for f in fractions]


# --- Opponent probability models ---------------------------------------------------------


class ResponseModel(ABC):
    """Predicts the opponent's next action in hypothetical states (legal-masked)."""

    name: str = "response_model"

    @abstractmethod
    def predict(self, observations: list[Observation], tracker: OpponentTracker | None) -> MaskedProbabilities:
        """Raw and legal-masked probabilities in ``CLASS_ORDER``, one row per observation."""

    @property
    def priors(self) -> HistoryPriors | None:
        """History priors for an opponent tracker (None if history is not used)."""
        return None

    def config(self) -> dict[str, Any]:
        return {"name": self.name}


class FrequencyResponseModel(ResponseModel):
    """Global training-split action frequencies, legal-masked per state (the control)."""

    name = "frequency_prior"

    def __init__(self, shares) -> None:
        shares = np.asarray(shares, dtype=float)
        if shares.shape != (len(CLASS_ORDER),) or (shares < 0).any() or not np.isclose(shares.sum(), 1.0):
            raise ValueError("The frequency prior must be 5 non-negative shares summing to 1.")
        self.shares = shares

    @classmethod
    def load(cls, path: Path = FREQUENCY_PRIOR_PATH) -> FrequencyResponseModel:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["class_order"] != list(CLASS_ORDER):
            raise ValueError("Frequency prior uses a different class order.")
        return cls(data["shares"])

    def predict(self, observations, tracker) -> MaskedProbabilities:
        frame = pd.DataFrame([observation_columns(o) for o in observations])
        raw = np.tile(self.shares, (len(observations), 1))
        return mask_and_renormalize(raw, legal_mask(frame))

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "shares": [round(float(s), 12) for s in self.shares]}


class MLResponseModel(ResponseModel):
    """The frozen Phase 3 opponent model with the opponent's live public history."""

    name = "random_forest_B_22"

    def __init__(self, predictor: OpponentPredictor, model_sha256: str | None = None) -> None:
        self.predictor = predictor
        self.model_sha256 = model_sha256

    @classmethod
    def load(cls, path: Path = PHASE4_MODEL_PATH, expected_sha256: str = FROZEN_MODEL_SHA256) -> MLResponseModel:
        return cls(OpponentPredictor.load(path, expected_sha256=expected_sha256), model_sha256=expected_sha256)

    @property
    def priors(self) -> HistoryPriors | None:
        return self.predictor.priors

    def predict(self, observations, tracker) -> MaskedProbabilities:
        if tracker is None:
            raise ValueError("The ML model needs an opponent tracker for its history features.")
        return self.predictor.predict(observations, tracker.history_features())

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "model_sha256": self.model_sha256}


# --- Diagnostics -----------------------------------------------------------------------------


@dataclass
class DecisionDiagnostics:
    """Everything that went into one decision (kept in memory, for analysis)."""

    street: str
    position: str
    equity: float
    action_evs: dict[str, float]
    chosen: str
    reason: str
    opponent_probabilities: dict[str, list[float]] = field(default_factory=dict)  # masked, per raise category
    opponent_raw_probabilities: dict[str, list[float]] = field(default_factory=dict)
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)  # per raise category
    continue_fraction: dict[str, float] = field(default_factory=dict)  # P(call) + P(re-raise), per raise
    called_equity: dict[str, float] = field(default_factory=dict)  # equity vs the continuing range, per raise
    hypothetical_obs: dict[str, dict[str, Any]] = field(default_factory=dict)  # obs_ columns, per raise category
    opponent_history_hands: int = 0
    equity_seconds: float = 0.0
    model_seconds: float = 0.0
    range_seconds: float = 0.0
    total_seconds: float = 0.0


# --- The agent ----------------------------------------------------------------------------------


class EVAgent(Agent):
    """One-step expected-value agent with a pluggable opponent probability model."""

    def __init__(
        self,
        name: str,
        response_model: ResponseModel,
        equity_samples: int = EQUITY_SAMPLES,
        equity_fn: EquityFunction = hand_strength,
        coverage: dict[str, Any] | None = None,
        called_equity_fn: CalledEquityFunction = called_equities,
        range_samples: int = RANGE_EQUITY_SAMPLES,
    ) -> None:
        super().__init__(name)
        self.response_model = response_model
        self.equity_samples = equity_samples
        self.equity_fn = equity_fn
        self.called_equity_fn = called_equity_fn
        self.range_samples = range_samples
        self.coverage_counts = (coverage or {}).get("counts", {})
        self.low_coverage_threshold = (coverage or {}).get("low_coverage_threshold", LOW_COVERAGE_THRESHOLD)
        priors = response_model.priors
        self.tracker = OpponentTracker(priors) if priors is not None else None
        self.diagnostics: list[DecisionDiagnostics] = []

    def config(self) -> dict[str, Any]:
        return {**super().config(), "response_model": self.response_model.config(),
                "equity_samples": self.equity_samples, "range_samples": self.range_samples,
                "tie_break": "fewest additional chips, then CLASS_ORDER",
                "decision_rule": "one-step EV (hybridpoker.ev), called branch vs continuing range"}

    def observe_hand_end(self, summary: PublicHandSummary) -> None:
        if self.tracker is not None:
            self.tracker.observe_hand_end(summary)

    @staticmethod
    def choose(values: list[ActionValue], chips: dict[ActionCategory, int]) -> ActionValue:
        """Highest EV; ties go to fewer additional chips, then to the earlier class."""
        best = max(v.ev for v in values)
        tied = [v for v in values if v.ev >= best - TIE_TOLERANCE]
        return min(tied, key=lambda v: (chips[v.category], CLASS_ORDER.index(v.category.value)))

    def act(self, observation: Observation, rng: random.Random) -> Decision:
        start = time.perf_counter()
        equity = self.equity_fn(observation, self.equity_samples, rng)
        equity_done = time.perf_counter()

        raises = [a for a in observation.legal_actions if a.category in RAISE_CATEGORIES]
        hypotheticals = [hypothetical_opponent_observation(observation, a) for a in raises]
        if any(h is None for h in hypotheticals):
            raise RuntimeError("A bet or raise must leave the opponent a decision.")
        model_start = time.perf_counter()
        predicted = self.response_model.predict(hypotheticals, self.tracker) if raises else None
        model_done = time.perf_counter()

        responses = {a.category: ResponseProbabilities.from_action_probabilities(predicted.masked[i])
                     for i, a in enumerate(raises)}
        range_start = time.perf_counter()
        fractions = {c: min(1.0, r.call + r.raise_) for c, r in responses.items()}
        estimates = self.called_equity_fn(observation, equity, list(fractions.values()), self.range_samples, rng)
        called = dict(zip(fractions, estimates))
        range_done = time.perf_counter()
        values = [action_value(observation, a, equity, responses.get(a.category), called.get(a.category))
                  for a in observation.legal_actions]
        chips = {a.category: a.chips_added for a in observation.legal_actions}
        chosen = self.choose(values, chips)
        reason = f"{chosen.explanation} (best of: " + ", ".join(
            f"{v.category.value} {v.ev:+.2f}" for v in values) + ")"

        diag = DecisionDiagnostics(
            street=observation.street, position=observation.position, equity=equity,
            action_evs={v.category.value: v.ev for v in values}, chosen=chosen.category.value, reason=reason,
            opponent_history_hands=(int(round(np.expm1(self.tracker.history_features()["b_log_prior_hands"])))
                                    if self.tracker is not None else 0),
            equity_seconds=equity_done - start, model_seconds=model_done - model_start,
            range_seconds=range_done - range_start,
        )
        for i, (action, hypothetical) in enumerate(zip(raises, hypotheticals)):
            cols = observation_columns(hypothetical)
            key = coverage_key(cols)
            count = int(self.coverage_counts.get(key, 0))
            name = action.category.value
            diag.opponent_probabilities[name] = predicted.masked[i].tolist()
            diag.opponent_raw_probabilities[name] = predicted.raw[i].tolist()
            diag.coverage[name] = {"key": key, "training_rows": count, "low": count < self.low_coverage_threshold}
            diag.hypothetical_obs[name] = cols
            diag.continue_fraction[name] = fractions[action.category]
            diag.called_equity[name] = called[action.category]
        diag.total_seconds = time.perf_counter() - start
        self.diagnostics.append(diag)
        return Decision(chosen.category, reason, equity)


def make_frequency_ev_agent(name: str = "ev_frequency", equity_samples: int = EQUITY_SAMPLES) -> EVAgent:
    """The Frequency EV Agent (control)."""
    return EVAgent(name, FrequencyResponseModel.load(), equity_samples, coverage=load_coverage(COVERAGE_PATH))


def make_ml_ev_agent(name: str = "ev_ml", equity_samples: int = EQUITY_SAMPLES,
                     model_path: Path = PHASE4_MODEL_PATH) -> EVAgent:
    """The ML EV Agent (treatment): frozen Random Forest B #22, verified by SHA-256."""
    return EVAgent(name, MLResponseModel.load(model_path), equity_samples, coverage=load_coverage(COVERAGE_PATH))
