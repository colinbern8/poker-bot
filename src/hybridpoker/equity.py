"""Hand equity calculations.

*Equity* is the share of the pot a hand wins on average if the hand is
played to showdown with no further betting. Winning outright is worth 1;
an n-way split pot is worth 1/n; losing is worth 0.

Two methods are provided:

* :func:`estimate_equity` -- seeded Monte Carlo simulation. Works on any
  street and against any number of opponents.
* :func:`exact_river_equity` -- exact enumeration on the river against a
  single opponent. Used as a ground-truth check for the simulation.

Assumption used by both: every opponent holds a uniformly random hand
from the cards we cannot see. No opponent modeling is applied here; that
is the job of later phases.

Hand strength is evaluated with PokerKit's ``StandardHighHand``.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from pokerkit import Card, Deck, StandardHighHand

HOLE_CARD_COUNT = 2
FULL_BOARD_COUNT = 5
VALID_BOARD_COUNTS = (0, 3, 4, 5)  # preflop, flop, turn, river
MAX_OPPONENTS = 9  # a 10-handed table

# Fixed card order, so seeded sampling always sees the same deck order.
_STANDARD_DECK: tuple[Card, ...] = tuple(Deck.STANDARD)
_STANDARD_DECK_SET = frozenset(_STANDARD_DECK)

CardsInput = str | Iterable[Card]


@dataclass(frozen=True)
class EquityResult:
    """Outcome of an equity calculation for the hero's hand.

    Attributes:
        equity: Average share of the pot won (0.0 to 1.0). Split pots
            count fractionally, e.g. a 3-way tie is worth 1/3.
        win_rate: Fraction of outcomes the hero wins outright.
        tie_rate: Fraction of outcomes the hero splits the pot.
        loss_rate: Fraction of outcomes the hero wins nothing.
        standard_error: Monte Carlo uncertainty of ``equity``, computed
            from the per-outcome pot shares. 0.0 for exact results and
            NaN when only one sample was drawn.
        sample_count: Number of outcomes evaluated.
        opponent_count: Number of opponents.
        exact: True if every possible outcome was enumerated.
    """

    equity: float
    win_rate: float
    tie_rate: float
    loss_rate: float
    standard_error: float
    sample_count: int
    opponent_count: int
    exact: bool


def parse_cards(cards: CardsInput) -> tuple[Card, ...]:
    """Convert text such as ``"AsKh"`` (or Card objects) to a tuple of Cards.

    Ranks are ``23456789TJQKA`` and suits are ``cdhs``, e.g. ``"Td"`` is
    the ten of diamonds. Unknown cards (``"??"``) are rejected.
    """
    if isinstance(cards, str):
        try:
            parsed = tuple(Card.parse(cards))
        except ValueError as error:
            raise ValueError(f"Invalid card text {cards!r}: {error}") from error
    else:
        parsed = tuple(cards)

    for card in parsed:
        if card not in _STANDARD_DECK_SET:
            raise ValueError(f"{card!r} is not a card from a standard 52-card deck.")
    return parsed


def _validate_hand(
    hole_cards: CardsInput, board_cards: CardsInput
) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    hole = parse_cards(hole_cards)
    board = parse_cards(board_cards)

    if len(hole) != HOLE_CARD_COUNT:
        raise ValueError(f"Expected {HOLE_CARD_COUNT} hole cards, got {len(hole)}.")
    if len(board) not in VALID_BOARD_COUNTS:
        raise ValueError(
            f"Board must have 0, 3, 4, or 5 cards, got {len(board)}."
        )
    known = hole + board
    if len(set(known)) != len(known):
        raise ValueError(f"Duplicate cards: {' '.join(map(repr, known))}.")
    return hole, board


def _validate_positive_int(name: str, value: object, maximum: int | None = None) -> int:
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}.")
    return value


def hero_pot_share(
    hero_hole: tuple[Card, ...],
    opponent_holes: Iterable[tuple[Card, ...]],
    board: tuple[Card, ...],
) -> float:
    """Return the hero's share of the pot at showdown on a complete board.

    1.0 for an outright win, 1/n for an n-way tie for best hand, 0.0 for
    a loss.
    """
    hero_hand = StandardHighHand.from_game(hero_hole, board)
    tied_players = 1  # the hero
    for opponent_hole in opponent_holes:
        opponent_hand = StandardHighHand.from_game(opponent_hole, board)
        if opponent_hand > hero_hand:
            return 0.0
        if opponent_hand == hero_hand:
            tied_players += 1
    return 1.0 / tied_players


def summarize_pot_shares(
    shares: list[float], opponent_count: int, exact: bool
) -> EquityResult:
    """Turn a list of per-outcome pot shares into an EquityResult.

    The standard error is the sample standard deviation of the fractional
    shares (not a binary win/loss indicator) divided by sqrt(n).
    """
    n = len(shares)
    mean = sum(shares) / n
    wins = sum(1 for share in shares if share == 1.0)
    losses = sum(1 for share in shares if share == 0.0)
    ties = n - wins - losses

    if exact or n < 2:
        standard_error = 0.0 if exact else math.nan
    else:
        # Sample standard deviation of the fractional shares, divided by sqrt(n).
        variance = sum((share - mean) ** 2 for share in shares) / (n - 1)
        standard_error = math.sqrt(variance / n)

    return EquityResult(
        equity=mean,
        win_rate=wins / n,
        tie_rate=ties / n,
        loss_rate=losses / n,
        standard_error=standard_error,
        sample_count=n,
        opponent_count=opponent_count,
        exact=exact,
    )


def estimate_equity(
    hole_cards: CardsInput,
    board_cards: CardsInput = (),
    *,
    opponent_count: int = 1,
    sample_count: int = 1000,
    seed: int,
) -> EquityResult:
    """Estimate the hero's equity with a seeded Monte Carlo simulation.

    Each simulation deals random hole cards to every opponent and random
    cards to complete the board, all from the cards the hero cannot see.
    It then records the hero's share of the pot (1, 1/n, or 0). Equity is
    the average share across all simulations.

    Args:
        hole_cards: The hero's two hole cards, e.g. ``"AsKh"``.
        board_cards: 0, 3, 4, or 5 community cards, e.g. ``"Td9d2c"``.
        opponent_count: Number of opponents, each with a random hand.
        sample_count: Number of simulated outcomes. More samples give a
            smaller standard error (it shrinks with 1/sqrt(sample_count)).
        seed: Random seed. The same inputs and seed always give the same
            result.
    """
    hole, board = _validate_hand(hole_cards, board_cards)
    _validate_positive_int("opponent_count", opponent_count, MAX_OPPONENTS)
    _validate_positive_int("sample_count", sample_count)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}.")

    known = set(hole + board)
    unseen = [card for card in _STANDARD_DECK if card not in known]
    missing_board_count = FULL_BOARD_COUNT - len(board)
    cards_per_sample = HOLE_CARD_COUNT * opponent_count + missing_board_count

    rng = random.Random(seed)
    shares: list[float] = []
    for _ in range(sample_count):
        dealt = rng.sample(unseen, cards_per_sample)
        opponent_holes = [
            tuple(dealt[i : i + HOLE_CARD_COUNT])
            for i in range(0, HOLE_CARD_COUNT * opponent_count, HOLE_CARD_COUNT)
        ]
        full_board = board + tuple(dealt[HOLE_CARD_COUNT * opponent_count :])
        shares.append(hero_pot_share(hole, opponent_holes, full_board))

    return summarize_pot_shares(shares, opponent_count, exact=False)


def exact_river_equity(hole_cards: CardsInput, board_cards: CardsInput) -> EquityResult:
    """Compute the hero's exact river equity against one random opponent.

    With all five board cards known, the opponent can hold one of
    C(45, 2) = 990 two-card combinations. Each is equally likely, so the
    exact equity is the average pot share over all of them.

    Limited to one opponent: with two opponents there are about 450,000
    combinations, which is too slow for routine use.
    """
    hole, board = _validate_hand(hole_cards, board_cards)
    if len(board) != FULL_BOARD_COUNT:
        raise ValueError(
            f"Exact river equity needs a full 5-card board, got {len(board)} cards."
        )

    known = set(hole + board)
    unseen = [card for card in _STANDARD_DECK if card not in known]
    shares = [
        hero_pot_share(hole, [opponent_hole], board)
        for opponent_hole in itertools.combinations(unseen, HOLE_CARD_COUNT)
    ]
    return summarize_pot_shares(shares, opponent_count=1, exact=True)
