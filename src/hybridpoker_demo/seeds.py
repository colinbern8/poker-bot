"""Seeds and seating for demonstration matches.

One match seed determines every deck and every AI random stream, so a match
can be replayed exactly from its seed and the human's actions:

* normal play draws a fresh seed from the operating system (``secrets``);
* rehearsals pass a fixed seed to get the same cards every time.

The streams are derived under their own label (``"phase5-demo"``), so they
never coincide with the seeds of the Phase 2-4 experiments.

The human sits on the button in even-numbered hands (0, 2, ...) and in the
big blind in odd-numbered hands, so the button alternates every hand.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from hybridpoker.game import BIG_BLIND_SEAT, BUTTON_SEAT, other_seat
from hybridpoker.seeding import derive_seed

SEED_LABEL = "phase5-demo"
MAX_SEED = 2**63 - 1


def new_match_seed() -> int:
    """A fresh random match seed from the operating system."""
    return secrets.randbits(63)


def check_seed(seed: int) -> int:
    """Return ``seed`` if it is a valid match seed (an integer in [0, 2**63 - 1])."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"A match seed must be an integer from 0 to {MAX_SEED}, got {seed!r}.")
    return seed


@dataclass(frozen=True)
class HandSetup:
    """Seeds and seats for one hand of a match."""

    hand_index: int
    deal_seed: int
    agent_seeds: tuple[int, int]
    human_seat: int

    @property
    def ai_seat(self) -> int:
        return other_seat(self.human_seat)


def hand_setup(match_seed: int, hand_index: int) -> HandSetup:
    """The deck seed, the per-seat agent seeds and the human's seat for one hand."""
    if hand_index < 0:
        raise ValueError("hand_index must be non-negative.")
    return HandSetup(
        hand_index=hand_index,
        deal_seed=derive_seed(SEED_LABEL, match_seed, "deal", hand_index),
        agent_seeds=(derive_seed(SEED_LABEL, match_seed, "agent", hand_index, 0),
                     derive_seed(SEED_LABEL, match_seed, "agent", hand_index, 1)),
        human_seat=BUTTON_SEAT if hand_index % 2 == 0 else BIG_BLIND_SEAT,
    )
