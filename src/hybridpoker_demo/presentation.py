"""Display helpers for the demonstration: plain functions, no Streamlit.

Everything the screen shows is built here from the controller's public
views, so it can be unit-tested and cannot reach hidden information:

* cards as text with suit symbols (hearts and diamonds red);
* action labels with the exact chip amounts of the legal action;
* the public action log, the hand result and showdown hand names;
* the announcement of the AI's latest action (from the recorded history);
* the **after-hand explanation**, built only from the AI's recorded
  ``DecisionDiagnostics`` (nothing is estimated or invented here): a short
  default explanation (:func:`default_explanation`) and the full numbers for
  the "Detailed analysis" section (:func:`explain_decision`);
* the opponent profile, keeping three things apart: what the human was
  **observed** to do, the **model's inputs** (smoothed rates), and the
  **population** averages the smoothing starts from.

Chip conventions follow the engine: "bet/raise to X" is the player's total
bet on the street after the action; "call X" is the chips added.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pokerkit import Card, StandardHighHand

from hybridpoker.action_probabilities import CLASS_ORDER
from hybridpoker.actions import RAISE_CATEGORIES, ActionCategory, LegalAction
from hybridpoker.agents.ev_agent import TIE_TOLERANCE
from hybridpoker.game import BIG_BLIND
from hybridpoker.observation import PublicAction
from hybridpoker.opponent_tracker import decisions_from_history
from hybridpoker.state_coverage import LOW_COVERAGE_THRESHOLD
from hybridpoker_demo.controller import AIDecisionReport, CompletedHand, OpponentProfile

SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RED_SUITS = frozenset("hd")
HIDDEN_CARDS = ":gray[**■ ■**]"
POSITION_NAMES = {"button": "Button", "big_blind": "Big blind"}
STREET_NAMES = {"preflop": "Preflop", "flop": "Flop", "turn": "Turn", "river": "River"}
SIZE_NAMES = {ActionCategory.RAISE_HALF_POT: "½ pot", ActionCategory.RAISE_POT: "pot"}


# --- Cards ---------------------------------------------------------------------------------------


def card_text(card: Card) -> str:
    """``A♠``, ``10♥``."""
    rank = "10" if card.rank.value == "T" else card.rank.value
    return f"{rank}{SUIT_SYMBOLS[card.suit.value]}"


def card_markdown(card: Card) -> str:
    """Bold card text; red for hearts and diamonds (Streamlit colour syntax)."""
    text = f"**{card_text(card)}**"
    return f":red[{text}]" if card.suit.value in RED_SUITS else text


def cards_markdown(cards: Iterable[Card], empty: str = "—") -> str:
    shown = [card_markdown(c) for c in cards]
    return "  ".join(shown) if shown else empty


def hand_name(hole: Sequence[Card], board: Sequence[Card]) -> str:
    """The best five-card hand, e.g. "Two pair" (PokerKit's evaluation)."""
    return StandardHighHand.from_game(list(hole), list(board)).entry.label.value


# --- Actions -------------------------------------------------------------------------------------


def action_text(category: ActionCategory, bet_to: int, chips_added: int, to_call: int, street: str,
                is_all_in: bool = False) -> str:
    """A player-facing label with chip amounts, e.g. "Call 4", "Bet 6 (½ pot)", "Raise to 24 (pot)".

    A call that puts the player all-in reads "Call all-in (198)" (the chips it adds).
    """
    category = ActionCategory(category)
    if category is ActionCategory.FOLD:
        return "Fold"
    if category is ActionCategory.CHECK_CALL:
        if to_call == 0:
            return "Check"
        return f"Call all-in ({chips_added})" if is_all_in else f"Call {chips_added}"
    if category is ActionCategory.ALL_IN:
        return f"All-in ({bet_to})"
    verb = "Bet" if to_call == 0 and street != "preflop" else "Raise to"
    text = f"{verb} {bet_to} ({SIZE_NAMES[category]})"
    return text + (" — all-in" if is_all_in else "")


def legal_action_label(action: LegalAction, to_call: int, street: str) -> str:
    return action_text(action.category, action.bet_to, action.chips_added, to_call, street, action.is_all_in)


def chips(amount: int) -> str:
    return f"{amount} chip" + ("" if amount == 1 else "s")


def signed(amount: int | float) -> str:
    """A signed amount with a typographic minus, e.g. "+4", "−400", "+0"."""
    text = f"{amount:+d}" if isinstance(amount, int) else f"{amount:+g}"
    return text.replace("-", "−")


def ai_action_announcement(action: PublicAction, to_call: int) -> str:
    """One recorded AI action in plain words, e.g. "AI raises to 12 chips".

    Amounts follow the engine: a call names the chips added; a bet, raise or
    all-in names the AI's total bet on the street after the action.
    """
    category = ActionCategory(action.category)
    if category is ActionCategory.FOLD:
        return "AI folds"
    if category is ActionCategory.CHECK_CALL:
        if to_call == 0:
            return "AI checks"
        return f"AI calls all-in for {chips(action.chips_added)}" if action.is_all_in else \
            f"AI calls {chips(action.chips_added)}"
    if action.is_all_in:
        return f"AI goes all-in for {chips(action.bet_to)}"
    if to_call == 0 and action.street != "preflop":
        return f"AI bets {chips(action.bet_to)}"
    return f"AI raises to {chips(action.bet_to)}"


def latest_ai_action(history: Sequence[PublicAction], ai_position: str) -> tuple[str, str] | None:
    """The street name and announcement of the AI's most recent action in the hand, if any.

    Built from the public history only, so it always agrees with the action history.
    """
    latest = None
    for action, decision in zip(history, decisions_from_history(tuple(history))):
        if action.position == ai_position:
            latest = (STREET_NAMES[action.street], ai_action_announcement(action, decision.to_call))
    return latest


def history_lines(history: Sequence[PublicAction], human_position: str) -> list[str]:
    """The public action log, e.g. "Preflop · You (Button): Raise to 6 (pot)"."""
    lines = []
    for action, decision in zip(history, decisions_from_history(tuple(history))):
        who = "You" if action.position == human_position else "AI"
        label = action_text(action.category, action.bet_to, action.chips_added, decision.to_call, action.street,
                            action.is_all_in)
        lines.append(f"{STREET_NAMES[action.street]} · {who} ({POSITION_NAMES[action.position]}): {label}")
    return lines


# --- Results -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HandResultText:
    headline: str
    detail: str
    outcome: str  # "win", "loss" or "split"


def chips_and_bb(chips: int) -> str:
    return f"{chips:+d} chips ({chips / BIG_BLIND:+g} bb)"


def hand_result(hand: CompletedHand) -> HandResultText:
    if hand.human_net > 0:
        headline, outcome = f"You won {hand.human_net} chips ({hand.human_net / BIG_BLIND:g} bb)", "win"
    elif hand.human_net < 0:
        headline, outcome = f"The AI won {-hand.human_net} chips ({-hand.human_net / BIG_BLIND:g} bb)", "loss"
    else:
        headline, outcome = "Split pot", "split"
    if hand.went_to_showdown:
        yours = hand_name(hand.human_hole_cards, hand.board_cards)
        theirs = hand_name(hand.ai_hole_cards, hand.board_cards)
        detail = f"Showdown: you had {yours}, the AI had {theirs}."
    else:
        folder = "You" if hand.history[-1].position == hand.human_position else "The AI"
        detail = f"{folder} folded. The AI's cards stay hidden."
    return HandResultText(headline, detail, outcome)


# --- Explanation of the AI's decisions (only from recorded diagnostics) ----------------------------


@dataclass(frozen=True)
class EVRow:
    action: str
    ev: float
    chosen: bool


@dataclass(frozen=True)
class ResponseRow:
    """The opponent model's predicted response to one AI bet size (a model prediction)."""

    action: str
    fold: float
    call: float
    raise_: float
    continue_fraction: float
    equity_if_called: float
    training_rows: int
    low_coverage: bool


@dataclass(frozen=True)
class DecisionExplanation:
    street: str
    action: str
    equity: float
    ev_rows: tuple[EVRow, ...]
    responses: tuple[ResponseRow, ...]
    warnings: tuple[str, ...]
    summary: str
    history_hands: int
    seconds: float


def _ai_action_labels(report: AIDecisionReport) -> dict[str, str]:
    """Labels with chip amounts for every action the AI could take (its real legal actions)."""
    labels = {a.category.value: legal_action_label(a, report.to_call, report.street) for a in report.legal_actions}
    missing = set(report.diagnostics.action_evs) - set(labels)
    if missing:
        raise ValueError(f"Diagnostics mention actions that were not legal: {sorted(missing)}.")
    return labels


def explain_decision(report: AIDecisionReport) -> DecisionExplanation:
    """The explanation of one AI decision. Every number comes from ``report.diagnostics``."""
    d = report.diagnostics
    if d is None:
        raise ValueError("This AI decision has no recorded diagnostics.")
    labels = _ai_action_labels(report)
    chosen = d.chosen
    best = max(d.action_evs.values())
    ev_rows = tuple(EVRow(labels[name], float(ev), name == chosen) for name, ev in d.action_evs.items())

    responses = []
    for name, probs in d.opponent_probabilities.items():
        p = dict(zip(CLASS_ORDER, probs))
        coverage = d.coverage[name]
        responses.append(ResponseRow(
            action=labels[name], fold=p["fold"], call=p["check_call"],
            raise_=p["raise_half_pot"] + p["raise_pot"] + p["all_in"],
            continue_fraction=float(d.continue_fraction[name]), equity_if_called=float(d.called_equity[name]),
            training_rows=int(coverage["training_rows"]), low_coverage=bool(coverage["low"])))

    warnings = []
    if chosen in d.coverage and d.coverage[chosen]["low"]:
        warnings.append(
            f"The situation after this bet resembles only {d.coverage[chosen]['training_rows']} training "
            f"examples (fewer than {LOW_COVERAGE_THRESHOLD}), so the predicted response is less certain.")
    other_low = [labels[n] for n, c in d.coverage.items() if c["low"] and n != chosen]
    if other_low:
        warnings.append(f"Predictions for these other options also rest on fewer than {LOW_COVERAGE_THRESHOLD} "
                        "similar training examples: " + ", ".join(other_low) + ".")

    tied = [n for n, ev in d.action_evs.items() if ev >= best - TIE_TOLERANCE]
    parts = [f"On the {report.street}, the AI estimated {d.equity:.0%} equity against a random hand."]
    rank = "had the highest expected value" if len(tied) == 1 else \
        "tied for the highest expected value (ties go to the action risking fewer chips)"
    parts.append(f"{labels[chosen]} {rank}: {d.action_evs[chosen]:+.1f} chips, "
                 f"of {len(d.action_evs)} legal actions.")
    if chosen in d.opponent_probabilities:
        row = next(r for r in responses if r.action == labels[chosen])
        parts.append(f"The model predicted you would fold {row.fold:.0%} of the time; if you continued, the AI's "
                     f"equity against your likely continuing hands was {row.equity_if_called:.0%}.")
    elif d.opponent_probabilities:
        bets = [(n, ev) for n, ev in d.action_evs.items() if n in d.opponent_probabilities]
        name, ev = max(bets, key=lambda item: item[1])
        parts.append(f"Its best bet, {labels[name]}, was worth {ev:+.1f} chips.")
    return DecisionExplanation(
        street=report.street, action=labels[chosen], equity=float(d.equity), ev_rows=ev_rows,
        responses=tuple(responses), warnings=tuple(warnings), summary=" ".join(parts),
        history_hands=int(d.opponent_history_hands), seconds=float(d.total_seconds))


# --- The default explanation of one AI decision (readable in about ten seconds) ------------------------

NEAR_TIE_CHIPS = 0.5  # wording only: a smaller lead is described as "only slightly ahead"
MAX_VALUE_DECIMALS = 3  # values are shown with 1 decimal, more only if needed to tell them apart
MISMATCH_REASON = "The recorded analysis does not match this action, so no explanation is shown."


@dataclass(frozen=True)
class DefaultExplanation:
    """The short explanation shown for every AI decision; the rest is under "Detailed analysis".

    ``status`` is "ok" (complete, consistent diagnostics), "no_diagnostics",
    "missing_values" (not every legal action has a recorded value) or
    "mismatch" (the diagnostics disagree with the action played). Only "ok"
    has a comparison; only when ``details`` is true can the full detailed
    analysis (:func:`explain_decision`) be built from the diagnostics.
    """

    headline: str
    reason: str
    comparison: str | None = None
    response: str | None = None
    warning: str | None = None
    status: str = "ok"
    details: bool = False


def _bets(to_call: int, street: str) -> bool:
    """True if a raise category is a bet (nothing to call after the flop), as in :func:`action_text`."""
    return to_call == 0 and street != "preflop"


def _short_name(action: LegalAction, to_call: int, street: str) -> str:
    """A short action name for the value comparison, e.g. "Call", "Bet ½ pot", "All-in"."""
    category = action.category
    if category is ActionCategory.FOLD:
        return "Fold"
    if category is ActionCategory.CHECK_CALL:
        return "Check" if to_call == 0 else "Call"
    if category is ActionCategory.ALL_IN:
        return "All-in"
    name = f"{'Bet' if _bets(to_call, street) else 'Raise'} {SIZE_NAMES[category]}"
    return name + (" (all-in)" if action.is_all_in else "")


def _gerund(action: LegalAction, to_call: int, street: str) -> str:
    """The action as an "-ing" phrase, e.g. "calling", "betting half the pot", "going all-in"."""
    category = action.category
    if category is ActionCategory.FOLD:
        return "folding"
    if category is ActionCategory.CHECK_CALL:
        return "checking" if to_call == 0 else "calling"
    if category is ActionCategory.ALL_IN:
        return "going all-in"
    size = "half the pot" if category is ActionCategory.RAISE_HALF_POT else "the pot"
    phrase = f"{'betting' if _bets(to_call, street) else 'raising'} {size}"
    return phrase + (" (all-in)" if action.is_all_in else "")


def _bet_noun(action: LegalAction, to_call: int, street: str) -> str:
    if action.category is ActionCategory.ALL_IN or action.is_all_in:
        return "all-in"
    return "bet" if _bets(to_call, street) else "raise"


def _value_text(value: float, decimals: int) -> str:
    """An estimated value in chips, e.g. "+1.3 chips", "−2.0 chips", "0 chips" (exactly zero only)."""
    if value == 0:
        return "0 chips"
    return f"{value:+.{decimals}f}".replace("-", "−") + " chips"


def _comparison_decimals(values: Sequence[float], tied: bool) -> int:
    """The fewest decimals (1 to 3) at which unequal values look unequal and no non-zero value looks like 0."""
    for decimals in range(1, MAX_VALUE_DECIMALS + 1):
        shown = [f"{v:.{decimals}f}" for v in values]
        distinct = tied or len(set(shown)) == len(shown)
        if distinct and all(v == 0 or float(s) != 0 for v, s in zip(values, shown)):
            return decimals
    return MAX_VALUE_DECIMALS


def _percent(p: float) -> str:
    if 0 < p < 0.005:
        return "less than 1%"
    if 0.995 < p < 1:
        return "more than 99%"
    return f"about {p:.0%}" if 0 < p < 1 else f"{p:.0%}"


def _headline(report: AIDecisionReport) -> str:
    """The action actually played, worded like the table announcement, e.g. "AI calls 4 chips."."""
    legal = next((a for a in report.legal_actions if a.category is report.category), None)
    action = PublicAction(report.street, report.position, report.category, report.bet_to, report.chips_added,
                          legal.is_all_in if legal is not None else False)
    return ai_action_announcement(action, report.to_call) + "."


def default_explanation(report: AIDecisionReport) -> DefaultExplanation:
    """The short default explanation of one AI decision, built only from its recorded diagnostics.

    It names the action, says why it was estimated to be best, and compares its
    estimated value with the next-best legal alternative. For a bet, raise or
    all-in it adds the model's predicted fold rate and a low-coverage warning,
    but only if they were recorded for that exact action. Nothing is inferred:
    missing or inconsistent diagnostics give a plain notice instead.
    """
    headline = _headline(report)
    d = report.diagnostics
    if d is None:
        return DefaultExplanation(headline, "No recorded analysis is available for this decision.",
                                  status="no_diagnostics")
    legal = {a.category.value: a for a in report.legal_actions}
    chosen = report.category.value
    values = {name: float(ev) for name, ev in d.action_evs.items()}
    if d.chosen != chosen or chosen not in legal or not set(values) <= set(legal):
        return DefaultExplanation(headline, MISMATCH_REASON, status="mismatch")
    if set(values) != set(legal):
        return DefaultExplanation(headline, "The AI's estimated values were not fully recorded for this decision, "
                                            "so no comparison is shown.", status="missing_values")
    others = [(name, ev) for name, ev in values.items() if name != chosen]
    if any(ev > values[chosen] + TIE_TOLERANCE for _, ev in others):
        return DefaultExplanation(headline, MISMATCH_REASON, status="mismatch")

    to_call, street = report.to_call, report.street
    lead = _gerund(legal[chosen], to_call, street).capitalize()
    if not others:
        reason = "This was the AI's only legal option."
        decimals = _comparison_decimals([values[chosen]], tied=True)
        comparison = f"Estimated value: {_short_name(legal[chosen], to_call, street)} " \
                     f"{_value_text(values[chosen], decimals)}"
    else:
        other, other_value = max(others, key=lambda item: item[1])  # the first of equals, in recorded order
        gap = values[chosen] - other_value
        tied = gap <= TIE_TOLERANCE  # the agent's own tie rule
        alternative = _gerund(legal[other], to_call, street)
        if tied:
            rule = "on a tie the AI takes the option that adds fewer chips" \
                if legal[chosen].chips_added < legal[other].chips_added else "the AI's fixed tie rule chose it"
            reason = f"{lead} tied with {alternative} for the highest estimated value; {rule}."
        elif gap < NEAR_TIE_CHIPS:
            reason = f"{lead} had the highest estimated value, only slightly ahead of {alternative}."
        else:
            reason = f"{lead} had the highest estimated value of the {len(legal)} available options."
        decimals = _comparison_decimals([values[chosen], other_value], tied)
        shown = [_value_text(values[chosen], decimals), _value_text(other_value, decimals)]
        comparison = (f"Estimated value: {_short_name(legal[chosen], to_call, street)} {shown[0]} | "
                      f"{_short_name(legal[other], to_call, street)} {shown[1]}")
        if not tied and shown[0] == shown[1]:
            comparison += f" (a difference under {10 ** -MAX_VALUE_DECIMALS:g} chips)"

    response = warning = None
    if report.category in RAISE_CATEGORIES:  # predictions exist only for the AI's own bets
        noun = _bet_noun(legal[chosen], to_call, street)
        probabilities = d.opponent_probabilities.get(chosen)
        if probabilities is not None and len(probabilities) == len(CLASS_ORDER):
            fold = float(probabilities[CLASS_ORDER.index("fold")])
            response = f"The model estimated you would fold {_percent(fold)} of the time to this {noun}."
        if (d.coverage.get(chosen) or {}).get("low") is True:
            warning = f"Few similar training examples for this {noun}, so the predicted response is less certain."
    details = all(name in legal and name in d.coverage and name in d.continue_fraction and name in d.called_equity
                  for name in d.opponent_probabilities)
    return DefaultExplanation(headline, reason, comparison, response, warning, status="ok", details=details)


# --- Opponent profile: observed behaviour versus model inputs ---------------------------------------


@dataclass(frozen=True)
class ProfileRow:
    behaviour: str
    observed: str        # what the human was seen to do, as "rate (count of n)"
    model_input: float   # the smoothed rate the model uses
    population: float    # the population average the smoothing starts from


def _observed_counts(hands: Sequence[CompletedHand]) -> dict[str, tuple[int, int]]:
    """Counts of the human's own actions in finished hands, with the same definitions as the tracker."""
    counts = {key: [0, 0] for key in ("vpip", "pfr", "face_fold", "face_call", "face_raise", "open_check",
                                      "open_bet")}
    for hand in hands:
        mine = [(a, d) for a, d in zip(hand.history, decisions_from_history(hand.history))
                if a.position == hand.human_position]
        preflop = [(a, d) for a, d in mine if a.street == "preflop"]
        if preflop:
            voluntary = any(a.category in RAISE_CATEGORIES or (d.to_call > 0 and a.category is ActionCategory.CHECK_CALL)
                            for a, d in preflop)
            counts["vpip"][0] += voluntary
            counts["pfr"][0] += any(a.category in RAISE_CATEGORIES for a, _ in preflop)
            counts["vpip"][1] += 1
            counts["pfr"][1] += 1
        for action, decision in mine:
            group = "face" if decision.to_call > 0 else "open"
            kind = ("raise" if group == "face" else "bet") if action.category in RAISE_CATEGORIES else \
                ("fold" if action.category is ActionCategory.FOLD else ("call" if group == "face" else "check"))
            for key in counts:
                if key.startswith(group + "_"):
                    counts[key][1] += 1
            counts[f"{group}_{kind}"][0] += 1
    return {key: (value[0], value[1]) for key, value in counts.items()}


def _rate(count: int, total: int) -> str:
    return "—" if total == 0 else f"{count / total:.0%} ({count} of {total})"


def profile_rows(profile: OpponentProfile, hands: Sequence[CompletedHand]) -> list[ProfileRow]:
    observed = _observed_counts(hands)
    r, pop = profile.rates, profile.population

    def summed(source: dict[str, float], prefix: str) -> float:
        return sum(source[f"{prefix}_{size}"] for size in ("half", "pot", "allin"))

    return [
        ProfileRow("Plays the hand voluntarily preflop (VPIP)", _rate(*observed["vpip"]), r["b_vpip"], pop["b_vpip"]),
        ProfileRow("Raises preflop (PFR)", _rate(*observed["pfr"]), r["b_pfr"], pop["b_pfr"]),
        ProfileRow("Folds when facing a bet", _rate(*observed["face_fold"]), r["b_face_fold"], pop["b_face_fold"]),
        ProfileRow("Calls when facing a bet", _rate(*observed["face_call"]), r["b_face_call"], pop["b_face_call"]),
        ProfileRow("Raises when facing a bet", _rate(*observed["face_raise"]), summed(r, "b_face"),
                   summed(pop, "b_face")),
        ProfileRow("Checks when not facing a bet", _rate(*observed["open_check"]), r["b_open_check"],
                   pop["b_open_check"]),
        ProfileRow("Bets when not facing a bet", _rate(*observed["open_bet"]), summed(r, "b_open"),
                   summed(pop, "b_open")),
    ]
