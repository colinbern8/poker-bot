"""Play the matches listed in a dataset manifest and write their logs.

Each match is played with Phase 2's ``run_match`` (seat-swapped, paired
deal seeds) and written with ``write_run`` (atomic; complete only when
``metadata.json`` exists). The match number, split and manifest hash go
into the run's ``metadata.json`` so the loader can verify every run.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

from hybridpoker.agents.strength import RANK_ORDER, preflop_equity
from hybridpoker.dataset_manifest import MatchSpec, configs_by_name, manifest_hash, run_dir_name
from hybridpoker.hand_log import write_run
from hybridpoker.match import run_match


def preflop_classes() -> list[str]:
    """All 169 starting-hand classes, e.g. ``AA``, ``AKs``, ``AKo``."""
    ranks = RANK_ORDER[::-1]
    classes = []
    for i, high in enumerate(ranks):
        classes.append(high * 2)
        for low in ranks[i + 1 :]:
            classes.extend((f"{high}{low}s", f"{high}{low}o"))
    return classes


def warm_preflop_cache(sample_counts: list[int]) -> float:
    """Fill the preflop equity cache; returns the seconds it took.

    Cached values depend only on the class and sample count (never on play
    order), so warming the cache does not change any results.
    """
    start = time.perf_counter()
    for samples, hand_class in itertools.product(sorted(set(sample_counts)), preflop_classes()):
        preflop_equity(hand_class, samples)
    return time.perf_counter() - start


def generate_match(spec: MatchSpec, manifest: dict[str, Any], out_root: Path) -> dict[str, Any]:
    """Play one manifest match and write its logs to ``out_root/match_NN``."""
    configs = configs_by_name(manifest)
    hero = configs[spec.hero].make_agent()
    villain = configs[spec.villain].make_agent()

    start = time.perf_counter()
    match = run_match(hero, villain, spec.deal_count, spec.seed, seat_swap=manifest["seat_swap"])
    elapsed = time.perf_counter() - start

    decisions = sum(len(h.result.decisions) for h in match.hands)
    extra = {
        "manifest_name": manifest["name"],
        "manifest_hash": manifest_hash(manifest),
        "match_number": spec.number,
        "split": spec.split,
        "scored_role": spec.scored_role,
        "timing": {"seconds": elapsed, "decisions": decisions, "hands": len(match.hands)},
    }
    write_run(match, hero, villain, Path(out_root) / run_dir_name(spec), extra=extra)
    return {"number": spec.number, "seconds": elapsed, "decisions": decisions, "hands": len(match.hands)}
