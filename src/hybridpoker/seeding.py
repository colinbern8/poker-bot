"""Deterministic seed derivation for reproducible simulations.

One master seed controls a whole run. Every random stream (each deal's
deck, each agent's choices) gets its own seed derived from the master seed
and a label, e.g. ``derive_seed(42, "deal", 17)``. The derivation uses
SHA-256, so it is identical on every platform and Python version, and
adding or removing one stream never shifts the others.
"""

from __future__ import annotations

import hashlib


def derive_seed(*parts: object) -> int:
    """Return a 63-bit integer seed derived from ``parts``."""
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1
