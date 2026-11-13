"""Tests for the dependence sensitivity check (scripts/phase4_dependence_sensitivity.py)."""

import importlib.util
import math

import numpy as np

spec = importlib.util.spec_from_file_location("phase4_dependence", "scripts/phase4_dependence_sensitivity.py")
dep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dep)


def ar1(rng, n, rho):
    x = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(n):
        x[i] = (rho * x[i - 1] if i else 0.0) + e[i]
    return x


def test_independent_series_give_the_iid_standard_error_on_average() -> None:
    rng = np.random.default_rng(0)
    ratios = [(dep.batch_means_se(x) / (x.std(ddof=1) / math.sqrt(500)),
               dep.newey_west_se(x) / (x.std(ddof=1) / math.sqrt(500)))
              for x in (rng.normal(0, 1, 500) for _ in range(200))]
    batch, hac = np.mean(ratios, axis=0)
    assert 0.9 < batch < 1.1 and 0.9 < hac < 1.1


def test_positively_correlated_series_get_wider_standard_errors() -> None:
    rng = np.random.default_rng(1)
    ratios = [(dep.batch_means_se(x) / (x.std(ddof=1) / math.sqrt(500)),
               dep.newey_west_se(x) / (x.std(ddof=1) / math.sqrt(500)))
              for x in (ar1(rng, 500, 0.5) for _ in range(200))]
    batch, hac = np.mean(ratios, axis=0)
    assert batch > 1.5 and hac > 1.45  # theory: sqrt(3) = 1.73


def test_autocorrelation_and_batch_errors() -> None:
    x = np.array([1.0, -1.0] * 50)
    assert math.isclose(dep.autocorrelation(x, 1), -0.99, abs_tol=1e-9)
    try:
        dep.batch_means_se(np.ones(101))
    except ValueError:
        pass
    else:
        raise AssertionError("uneven batches must be rejected")


def test_block_bootstrap_matches_the_normal_interval_for_iid_data() -> None:
    rng = np.random.default_rng(2)
    xs = [rng.normal(5, 100, 500) for _ in range(16)]
    low, high = dep.block_bootstrap(xs, dep.BLOCK_LENGTH, 4000, 3)
    se = math.sqrt(sum(x.var(ddof=1) / 500 for x in xs)) / 16
    assert abs((high - low) / 2 - 1.96 * se) < 0.15 * 1.96 * se


def test_analysis_combines_opponents_like_the_registered_interval() -> None:
    rng = np.random.default_rng(4)
    opponents = ["a", "b"]
    results = []
    for o in opponents:
        for hero in ("ev_ml", "ev_situation", "ev_global", "probability"):
            nets = rng.integers(-40, 40, 2 * 20)
            results.append({"hero": hero, "opponent": o,
                            "hands": [{"deal_index": i // 2, "hero_net": int(n)} for i, n in enumerate(nets)]})
    out = dep.analyse(results, 20, opponents, seed=5)
    primary = out["ev_ml - ev_situation"]
    assert primary["role"] == "primary"
    assert math.isclose(primary["batch_means"]["mean"], primary["registered_iid"]["mean"])
    assert set(primary["zero_excluded"]) == {"registered", "batch_means", "newey_west", "block_bootstrap"}
