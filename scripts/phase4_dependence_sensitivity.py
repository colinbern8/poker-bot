"""Sensitivity analysis: dependence between deal pairs within the same match.

    python scripts/phase4_dependence_sensitivity.py              # final evaluation (after the run)
    python scripts/phase4_dependence_sensitivity.py --dev 100    # development records (validation)

**Status.** Requested by the project owner after the freeze of commit d50af2b
and written **before any final hand was played**. It is not part of the
frozen code (it reads the saved match records only) and it **does not
change the registered primary result**. It is a clearly labelled secondary
check of one assumption of the registered interval: that the 500 deal pairs
of a match are independent.

**Why dependence is possible.** Within a match the ML agent learns the
opponent's history, so its play in later deals depends on earlier deals.
(Stacks reset every hand, so chips do not carry over.) If paired differences
were correlated over consecutive deals, the registered standard error could
be too small.

**Methods (fixed in advance), for the primary and both secondary comparisons,**
applied to each opponent's series of paired differences in deal order:

1. *Batch means:* 10 consecutive batches of equal size (50 deal pairs in the
   final run). SE_o = SD(batch means) / sqrt(10).
2. *Newey-West (HAC) standard error* with Bartlett weights and lag 10.
3. *Circular moving-block bootstrap* within each opponent, block length 10
   deal pairs (50 blocks per final match), 10,000 replicates, seed
   derive_seed(4000, "bootstrap", "dependence"); 95% percentile interval.
   (A block length of 25 was first tried on the development data only: with
   4 blocks per 100-deal match the interval was too narrow, a known
   small-sample bias, so 10 was fixed before any final hand.)
4. *Autocorrelation* at lags 1-5, averaged over opponents.

For 1 and 2, opponents are combined exactly as in the registered analysis
(equal weight, SE = sqrt(sum SE_o^2) / 16, 95% normal interval).

**Reading rule (fixed in advance).** The registered interval and decision stand
as reported. If a dependence-robust interval disagrees with it about
whether 0 is excluded, that is reported as a limitation of the primary
result, not as a replacement for it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

from hybridpoker.phase4_eval import net_by_deal, to_bb100
from hybridpoker.seeding import derive_seed

BATCHES = 10
HAC_LAG = 10
BLOCK_LENGTH = 10
REPLICATES = 10_000
Z = NormalDist().inv_cdf(0.975)
COMPARISONS = (("ev_ml", "ev_situation", "primary"), ("ev_ml", "ev_global", "secondary"),
               ("ev_ml", "probability", "secondary"))


def batch_means_se(x: np.ndarray, batches: int = BATCHES) -> float:
    if len(x) % batches:
        raise ValueError("The series length must be a multiple of the batch count.")
    means = x.reshape(batches, -1).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(batches))


def newey_west_se(x: np.ndarray, lag: int = HAC_LAG) -> float:
    n = len(x)
    e = x - x.mean()
    long_run = e @ e / n
    for k in range(1, lag + 1):
        long_run += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(math.sqrt(max(long_run, 0.0) / n))


def autocorrelation(x: np.ndarray, k: int) -> float:
    e = x - x.mean()
    return float((e[k:] @ e[:-k]) / (e @ e))


def block_bootstrap(series: list[np.ndarray], block: int, replicates: int, seed: int) -> tuple[float, float]:
    """Equal-weight mean; circular blocks resampled within each opponent; 95% percentile interval."""
    rng = np.random.default_rng(seed)
    total = np.zeros(replicates)
    for x in series:
        n = len(x)
        blocks = math.ceil(n / block)
        starts = rng.integers(0, n, size=(replicates, blocks))
        index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(replicates, -1)[:, :n] % n
        total += x[index].mean(axis=1)
    total /= len(series)
    low, high = np.percentile(total, [2.5, 97.5])
    return float(low), float(high)


def combine(means: list[float], ses: list[float]) -> dict[str, float]:
    k = len(means)
    mean = sum(means) / k
    se = math.sqrt(sum(s ** 2 for s in ses)) / k
    return {"mean": mean, "se": se, "low": mean - Z * se, "high": mean + Z * se}


def analyse(results: list[dict], deals: int, opponents: list[str], seed: int) -> dict:
    by = {(r["hero"], r["opponent"]): r for r in results}
    out = {}
    for treatment, control, role in COMPARISONS:
        series = [to_bb100(net_by_deal(by[(treatment, o)]["hands"], deals))
                  - to_bb100(net_by_deal(by[(control, o)]["hands"], deals)) for o in opponents]
        means = [float(x.mean()) for x in series]
        iid = combine(means, [float(x.std(ddof=1) / math.sqrt(len(x))) for x in series])
        batch = combine(means, [batch_means_se(x) for x in series])
        hac = combine(means, [newey_west_se(x) for x in series])
        boot = block_bootstrap(series, BLOCK_LENGTH, REPLICATES, seed)
        out[f"{treatment} - {control}"] = {
            "role": role, "registered_iid": iid, "batch_means": batch, "newey_west": hac,
            "block_bootstrap_95": boot,
            "autocorrelation": {k: float(np.mean([autocorrelation(x, k) for x in series])) for k in range(1, 6)},
            "se_ratio_batch": batch["se"] / iid["se"], "se_ratio_hac": hac["se"] / iid["se"],
            "zero_excluded": {"registered": iid["low"] > 0 or iid["high"] < 0,
                              "batch_means": batch["low"] > 0 or batch["high"] < 0,
                              "newey_west": hac["low"] > 0 or hac["high"] < 0,
                              "block_bootstrap": boot[0] > 0 or boot[1] < 0}}
    return out


def markdown(result: dict, title: str) -> str:
    lines = [f"# {title}", "",
             "Secondary sensitivity check requested after the freeze and written before any final hand was "
             "played (`scripts/phase4_dependence_sensitivity.py`). It does **not** change the registered "
             "primary result; see the script's header for the methods and the reading rule.", "",
             "| comparison | role | registered (i.i.d.) | batch means (10 batches) | Newey-West (lag 10) | "
             "block bootstrap (10) | SE ratio batch / HAC | lag-1 to lag-5 autocorrelation |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in result.items():
        def f(e):
            return f"{e['mean']:+.1f} [{e['low']:+.1f}, {e['high']:+.1f}]"
        ac = ", ".join(f"{v:+.3f}" for v in r["autocorrelation"].values())
        lines.append(f"| {name} | {r['role']} | {f(r['registered_iid'])} | {f(r['batch_means'])} | "
                     f"{f(r['newey_west'])} | [{r['block_bootstrap_95'][0]:+.1f}, {r['block_bootstrap_95'][1]:+.1f}] | "
                     f"{r['se_ratio_batch']:.2f} / {r['se_ratio_hac']:.2f} | {ac} |")
    lines += ["", "Does each interval exclude 0?", "",
              "| comparison | registered | batch means | Newey-West | block bootstrap |", "|---|---|---|---|---|"]
    for name, r in result.items():
        z = r["zero_excluded"]
        lines.append(f"| {name} | {'yes' if z['registered'] else 'no'} | {'yes' if z['batch_means'] else 'no'} | "
                     f"{'yes' if z['newey_west'] else 'no'} | {'yes' if z['block_bootstrap'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev", type=int, help="analyse development records of this deal count instead")
    args = parser.parse_args()
    out_dir = Path("results/phase4")
    if args.dev:
        data = json.loads(gzip.decompress(Path(f"data/processed/phase4_dev/records_d{args.dev}.json.gz").read_bytes()))
        results, deals = data["results"], data["deals"]
        opponents = [o["name"] for o in data["opponents"]]
        seed, title, stem = derive_seed(5000, "bootstrap", "dependence"), "Dependence sensitivity (development run)", \
            "stage4c_dev_dependence_sensitivity"
    else:
        from hybridpoker.phase4_final import FINAL_DIR, MANIFEST_PATH, collect, jobs_from_manifest, load_manifest

        manifest = load_manifest(MANIFEST_PATH)
        main_jobs, replay_jobs = jobs_from_manifest(manifest)
        results = collect(main_jobs, replay_jobs, FINAL_DIR)["results"]
        deals, opponents = manifest["deals_per_match"], [o["name"] for o in manifest["opponents"]]
        seed, title, stem = derive_seed(4000, "bootstrap", "dependence"), "Dependence sensitivity (final evaluation)", \
            "stage4d_dependence_sensitivity"
    result = analyse(results, deals, opponents, seed)
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / f"{stem}.md").write_text(markdown(result, title), encoding="utf-8", newline="\n")
    print(markdown(result, title))
    return 0


if __name__ == "__main__":
    sys.exit(main())
