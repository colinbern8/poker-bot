"""A heads-up match between a human and one AI, driven one human action at a time.

**Rules** (the Phase 4 game, unchanged): heads-up No-Limit Hold'em, blinds
1/2, both players start **every hand with 200 chips**, the five abstract
action categories (only legal ones are accepted), and the button alternates
every hand (the human is on the button in hands 0, 2, 4, ...). The running
score is kept separately from the per-hand stacks.

**How a hand is played.** Every operation replays the current hand from the
start with the unchanged ``play_hand`` (see :mod:`hybridpoker_demo.recording`):
the human's committed actions are played back, the AI's committed decisions
are replayed exactly, and the replay stops at the human's next turn or at the
end of the hand.

**State safety.**

* Every operation first computes the new state on the side and commits it
  only after everything succeeded (a few plain assignments). If anything
  fails - an illegal or stale input, an AI error, an error while updating the
  AI's opponent tracker - the controller is left exactly as before, the AI's
  diagnostics list is truncated back, and the tracker is restored, so the
  same input can be retried.
* Inputs carry a :class:`Ticket` naming the match, hand and decision they
  answer. A ticket from an earlier turn, another match or a finished hand is
  rejected, so a double click cannot act twice.
* The AI is told about a finished hand (``observe_hand_end``) exactly once,
  in the same all-or-nothing step that finishes the hand, and the controller
  checks that its tracker recorded one row per human action of that hand.

**Information shown.** During a hand the human sees only :class:`HumanTurn`
(their own observation). The AI's reasoning (:class:`AIDecisionReport`) is
exposed only in :class:`CompletedHand`, after the hand. The AI's hole cards
are shown only if the hand reached a showdown; mucked cards stay hidden.
"""

from __future__ import annotations

import copy
import math
import secrets
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pokerkit import Card

from hybridpoker.actions import ActionCategory, LegalAction
from hybridpoker.agents.base import Agent
from hybridpoker.game import BIG_BLIND, SEAT_POSITIONS
from hybridpoker.observation import Observation, PublicAction
from hybridpoker.simulator import HandResult, play_hand, public_hand_summary, public_history
from hybridpoker_demo.human import HumanInputRequired, ScriptedHumanAgent
from hybridpoker_demo.recording import RecordedDecision, RecordingAgent, ReplayDivergence
from hybridpoker_demo.seeds import HandSetup, check_seed, hand_setup, new_match_seed


class ControllerError(RuntimeError):
    """An input the controller rejected. Its state is unchanged."""


class StaleTicket(ControllerError):
    """The input answers a turn that is no longer current (or belongs to another match)."""


class WrongPhase(ControllerError):
    """The operation is not allowed now (e.g. acting after the hand has finished)."""


class IllegalHumanAction(ControllerError):
    """The chosen action category is unknown or not legal in this situation."""


class TrackerMismatch(RuntimeError):
    """The AI's opponent tracker did not record one row per human action."""


class Phase(str, Enum):
    HUMAN_TO_ACT = "human_to_act"
    HAND_COMPLETE = "hand_complete"


@dataclass(frozen=True)
class Ticket:
    """Identifies the turn an input answers."""

    match_id: str
    hand_index: int
    decision_index: int


@dataclass(frozen=True)
class HumanTurn:
    """What the human may see while it is their turn: their own observation only."""

    ticket: Ticket
    hand_index: int
    observation: Observation

    @property
    def legal_categories(self) -> tuple[ActionCategory, ...]:
        return self.observation.legal_categories


@dataclass(frozen=True)
class AIDecisionReport:
    """One AI decision of a finished hand, for the explanation shown afterwards.

    ``diagnostics`` is the agent's ``DecisionDiagnostics`` (equity, EV of each
    action, the opponent model's predicted responses, coverage, timings); it
    holds no hole cards.
    """

    street: str
    position: str
    category: ActionCategory
    bet_to: int
    chips_added: int
    to_call: int  # chips the AI had to call before acting
    legal_actions: tuple[LegalAction, ...]  # the AI's options with their chip amounts (public)
    reason: str
    equity: float | None
    diagnostics: Any | None


@dataclass(frozen=True)
class CompletedHand:
    """A finished hand, as shown to the human afterwards."""

    hand_index: int
    human_position: str
    human_hole_cards: tuple[Card, ...]
    ai_hole_cards: tuple[Card, ...] | None  # only if the hand reached a showdown
    board_cards: tuple[Card, ...]
    history: tuple[PublicAction, ...]
    went_to_showdown: bool
    human_net: int
    ai_decisions: tuple[AIDecisionReport, ...]

    @property
    def ai_net(self) -> int:
        return -self.human_net


@dataclass(frozen=True)
class Scoreboard:
    """Cumulative result of the match (separate from the 200-chip stacks of each hand)."""

    hands_completed: int
    human_net_chips: int

    @property
    def ai_net_chips(self) -> int:
        return -self.human_net_chips

    @property
    def human_net_bb(self) -> float:
        return self.human_net_chips / BIG_BLIND

    @property
    def human_bb_per_100(self) -> float | None:
        return None if self.hands_completed == 0 else 100 * self.human_net_bb / self.hands_completed


@dataclass(frozen=True)
class OpponentProfile:
    """What the AI's opponent tracker currently believes about the human (read-only copy).

    These are the **model inputs** (Feature Set B): rates from the human's
    actions in finished hands, smoothed toward the population rates
    (``population``) with weight ``alpha`` (10), so with few hands they stay
    near the population values. Keys are the ``b_`` feature names, e.g.
    ``b_vpip``, ``b_pfr``, ``b_face_fold``.
    """

    hands_observed: int
    decisions_observed: int
    rates: dict[str, float]
    population: dict[str, float]
    alpha: float


@dataclass(frozen=True)
class _Replay:
    """The outcome of replaying the current hand (computed on the side, not yet committed)."""

    records: tuple[RecordedDecision, ...]
    pending: Observation | None
    result: HandResult | None


class HeadsUpController:
    """One human-versus-AI match. Create a **fresh** AI agent for every match."""

    def __init__(self, ai: Agent, match_seed: int | None = None) -> None:
        tracker = getattr(ai, "tracker", None)
        if getattr(ai, "diagnostics", None) or (tracker is not None and tracker.observed_rows()):
            raise ValueError("The AI has already played; create a fresh agent for every match.")
        self._ai = ai
        self.match_seed = check_seed(new_match_seed() if match_seed is None else match_seed)
        self.match_id = secrets.token_hex(8)
        self._completed: list[CompletedHand] = []
        self._results: list[HandResult] = []  # full results incl. hidden cards: for audits and tests only
        self._human_net = 0
        self._begin_hand(0)

    # --- read-only views -------------------------------------------------------------------------

    @property
    def phase(self) -> Phase:
        return Phase.HUMAN_TO_ACT if isinstance(self._view, HumanTurn) else Phase.HAND_COMPLETE

    @property
    def view(self) -> HumanTurn | CompletedHand:
        """The current turn, or the hand that has just finished."""
        return self._view

    @property
    def hand_index(self) -> int:
        return self._hand_index

    @property
    def completed_hands(self) -> tuple[CompletedHand, ...]:
        return tuple(self._completed)

    @property
    def scoreboard(self) -> Scoreboard:
        return Scoreboard(len(self._completed), self._human_net)

    def opponent_profile(self) -> OpponentProfile | None:
        """A copy of the AI's current opponent-history features, or None if the AI keeps none.

        The tracker is updated only when a hand finishes, so this reflects
        finished hands only. Reading it changes nothing the AI uses (the
        tracker may cache the computed features, which are deterministic).
        """
        tracker = getattr(self._ai, "tracker", None)
        if tracker is None:
            return None
        features = {name: float(value) for name, value in tracker.history_features().items()}
        priors = tracker.priors
        population = {f"b_open_{name}": float(rate) for name, rate in priors.open_rates.items()}
        population.update({f"b_face_{name}": float(rate) for name, rate in priors.face_rates.items()})
        population.update({"b_vpip": float(priors.vpip), "b_pfr": float(priors.pfr)})
        return OpponentProfile(
            hands_observed=int(round(math.expm1(features.pop("b_log_prior_hands")))),
            decisions_observed=int(round(math.expm1(features.pop("b_log_prior_decisions")))),
            rates=features,
            population=population,
            alpha=float(priors.alpha),
        )

    # --- operations ----------------------------------------------------------------------------

    def submit(self, ticket: Ticket, category: ActionCategory | str) -> HumanTurn | CompletedHand:
        """Play the human's action for the turn named by ``ticket``."""
        view = self._view
        if not isinstance(view, HumanTurn):
            raise WrongPhase("The hand has finished; start the next hand first.")
        if ticket != view.ticket:
            raise StaleTicket("This input answers a turn that is no longer current.")
        try:
            chosen = ActionCategory(category)
        except ValueError:
            raise IllegalHumanAction(f"Unknown action {category!r}.") from None
        if view.observation.legal(chosen) is None:
            raise IllegalHumanAction(
                f"{chosen.value} is not legal now; legal: {[c.value for c in view.legal_categories]}.")

        actions = (*self._human_actions, chosen)
        with self._rollback_ai_on_error():
            replay = self._replay(self._hand_index, actions, self._records)
            if replay.pending is not None:
                new_view = self._human_turn(self._hand_index, replay.pending)
                self._human_actions, self._records, self._view = actions, replay.records, new_view
            else:
                self._finish_hand(self._hand_index, actions, replay)
        return self._view

    def next_hand(self) -> HumanTurn | CompletedHand:
        """Deal the next hand (only after the current one has finished)."""
        if not isinstance(self._view, CompletedHand):
            raise WrongPhase("The current hand has not finished.")
        self._begin_hand(self._hand_index + 1)
        return self._view

    # --- internals -----------------------------------------------------------------------------

    @contextmanager
    def _rollback_ai_on_error(self) -> Iterator[None]:
        """Undo every change to the AI's diagnostics and tracker if the operation fails.

        The controller's own state needs no undo: it is assigned only as the
        last step of a successful operation.
        """
        diagnostics = getattr(self._ai, "diagnostics", None)
        kept = len(diagnostics) if diagnostics is not None else 0
        tracker = getattr(self._ai, "tracker", None)
        saved = copy.deepcopy(tracker)
        try:
            yield
        except BaseException:
            if diagnostics is not None:
                del diagnostics[kept:]
            if tracker is not None:
                self._ai.tracker = saved
            raise

    def _begin_hand(self, hand_index: int) -> None:
        with self._rollback_ai_on_error():
            replay = self._replay(hand_index, (), ())
            if replay.pending is not None:
                new_view = self._human_turn(hand_index, replay.pending)
                self._hand_index, self._human_actions, self._records, self._view = \
                    hand_index, (), replay.records, new_view
            else:  # the hand ended before the human had to act (the AI folded on the button)
                self._finish_hand(hand_index, (), replay)

    def _human_turn(self, hand_index: int, observation: Observation) -> HumanTurn:
        ticket = Ticket(self.match_id, hand_index, len(observation.history))
        return HumanTurn(ticket, hand_index, observation)

    def _replay(self, hand_index: int, actions: Sequence[ActionCategory],
                records: Sequence[RecordedDecision]) -> _Replay:
        """Replay the hand on the side (the caller rolls the AI back if this fails)."""
        setup = hand_setup(self.match_seed, hand_index)
        human = ScriptedHumanAgent(actions)
        recorder = RecordingAgent(self._ai, records)
        agents: list[Agent] = [human, human]
        agents[setup.ai_seat] = recorder
        try:
            result = play_hand(agents, setup.deal_seed, setup.agent_seeds)
            pending = None
        except HumanInputRequired as stop:
            result, pending = None, stop.observation
        recorder.assert_fully_replayed()
        if human.used != len(actions):
            raise ReplayDivergence(f"The replay used {human.used} of {len(actions)} human actions.")
        return _Replay(recorder.records, pending, result)

    def _finish_hand(self, hand_index: int, actions: tuple[ActionCategory, ...], replay: _Replay) -> None:
        """Tell the AI about the hand exactly once, then commit (the caller rolls the AI back on failure)."""
        setup = hand_setup(self.match_seed, hand_index)
        result = replay.result
        completed = self._completed_hand(setup, result, replay.records)
        summary = public_hand_summary(result, setup.ai_seat, hand_index, 0, 1)
        human_actions = sum(1 for d in result.decisions if d.seat == setup.human_seat)

        tracker = getattr(self._ai, "tracker", None)
        rows_before = tracker.observed_rows() if tracker is not None else 0
        self._ai.observe_hand_end(summary)
        if tracker is not None and self._ai.tracker.observed_rows() - rows_before != human_actions:
            raise TrackerMismatch("The opponent tracker did not record one row per human action.")
        # Commit (plain assignments only).
        self._hand_index, self._human_actions, self._records = hand_index, actions, replay.records
        self._completed.append(completed)
        self._results.append(result)
        self._human_net += completed.human_net
        self._view = completed

    @staticmethod
    def _completed_hand(setup: HandSetup, result: HandResult,
                        records: tuple[RecordedDecision, ...]) -> CompletedHand:
        ai_moves = [d for d in result.decisions if d.seat == setup.ai_seat]
        if len(ai_moves) != len(records):
            raise ReplayDivergence("The AI records do not match the AI decisions of the hand.")
        reports = tuple(
            AIDecisionReport(move.observation.street, move.observation.position, move.action.category,
                             move.action.bet_to, move.action.chips_added, move.observation.to_call,
                             move.observation.legal_actions, record.decision.reason, record.decision.equity,
                             record.diagnostics)
            for move, record in zip(ai_moves, records))
        return CompletedHand(
            hand_index=setup.hand_index,
            human_position=SEAT_POSITIONS[setup.human_seat],
            human_hole_cards=result.hole_cards[setup.human_seat],
            ai_hole_cards=result.hole_cards[setup.ai_seat] if result.went_to_showdown else None,
            board_cards=result.board_cards,
            history=public_history(result),
            went_to_showdown=result.went_to_showdown,
            human_net=result.payoffs[setup.human_seat],
            ai_decisions=reports,
        )
