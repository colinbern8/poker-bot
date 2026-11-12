"""Tests for hybridpoker.opponent_model (features, preprocessing, models, classes)."""

import numpy as np
import pandas as pd
import pytest

from hybridpoker.action_probabilities import CLASS_ORDER
from hybridpoker.features import A_FEATURES
from hybridpoker.opponent_history import B_FEATURES
from hybridpoker.opponent_model import (
    FeaturePreprocessor,
    class_probabilities,
    encode_targets,
    feature_matrix,
    grid,
    make_model,
    missing_training_classes,
    model_seed,
)


def test_encode_targets_uses_fixed_class_order() -> None:
    actions = pd.Series(["fold", "check_call", "raise_half_pot", "raise_pot", "all_in", "fold"])
    assert encode_targets(actions).tolist() == [0, 1, 2, 3, 4, 0]
    assert CLASS_ORDER[4] == "all_in"
    with pytest.raises(ValueError):
        encode_targets(pd.Series(["fold", "shove"]))


def test_grids_have_planned_sizes_and_fixed_order() -> None:
    assert grid("dummy") == [{}]
    assert len(grid("random_forest")) == 36
    assert len(grid("hist_gradient_boosting")) == 48
    assert grid("random_forest") == grid("random_forest")
    assert grid("random_forest")[0] == {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
                                        "max_features": "sqrt", "class_weight": None}


def test_models_have_fixed_seeds_and_no_internal_early_stopping() -> None:
    hgb = make_model("hist_gradient_boosting", {"max_iter": 10}, seed=7)
    assert hgb.early_stopping is False and hgb.random_state == 7
    rf = make_model("random_forest", {"n_estimators": 5}, seed=9)
    assert rf.random_state == 9 and rf.n_jobs == 1  # multi-threaded prediction is not bit-reproducible
    assert make_model("dummy", {}, seed=0).strategy == "prior"
    with pytest.raises(ValueError):
        make_model("svm", {}, seed=0)
    assert model_seed("random_forest", "B") == model_seed("random_forest", "B") < 2**32
    assert model_seed("random_forest", "A") != model_seed("random_forest", "B")


def test_preprocessor_is_fitted_on_training_data_only() -> None:
    train = pd.DataFrame({"a_x": [1.0, 2.0, 3.0], "a_constant": [1.0, 1.0, 1.0], "b_y": [0.1, 0.2, 0.3]})
    test = pd.DataFrame({"b_y": [0.5], "a_constant": [7.0], "a_x": [9.0]})  # varies only in test
    prep = FeaturePreprocessor().fit(train)
    assert prep.dropped_constant == ["a_constant"]
    assert prep.columns == ["a_x", "b_y"]
    np.testing.assert_array_equal(prep.transform(test), [[9.0, 0.5]])  # training column order kept


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_preprocessor_rejects_missing_or_infinite_values(bad) -> None:
    with pytest.raises(ValueError):
        FeaturePreprocessor().fit(pd.DataFrame({"a_x": [1.0, bad]}))
    prep = FeaturePreprocessor().fit(pd.DataFrame({"a_x": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        prep.transform(pd.DataFrame({"a_x": [bad]}))


def test_preprocessor_rejects_forbidden_names_and_missing_columns() -> None:
    with pytest.raises(ValueError):
        FeaturePreprocessor().fit(pd.DataFrame({"label_actor_agent": [1.0, 2.0]}))
    prep = FeaturePreprocessor().fit(pd.DataFrame({"a_x": [1.0, 2.0], "a_z": [0.0, 1.0]}))
    with pytest.raises(ValueError):
        prep.transform(pd.DataFrame({"a_x": [1.0]}))
    with pytest.raises(RuntimeError):
        FeaturePreprocessor().transform(pd.DataFrame({"a_x": [1.0]}))


def test_missing_training_class_gets_zero_probability_not_a_new_task() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    y = np.tile([0, 1, 2, 3], 15)  # all_in never appears in training
    model = make_model("random_forest", {"n_estimators": 10}, seed=1).fit(x, y)
    proba = class_probabilities(model, x[:5])
    assert proba.shape == (5, 5)
    assert (proba[:, 4] == 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert missing_training_classes(model) == ["all_in"]


@pytest.mark.parametrize("name", ["random_forest", "hist_gradient_boosting"])
def test_training_is_reproducible(name) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] > 0).astype(int) + (x[:, 1] > 1).astype(int) * 2
    params = {"n_estimators": 20} if name == "random_forest" else {"max_iter": 20}
    first = class_probabilities(make_model(name, params, seed=5).fit(x, y), x)
    second = class_probabilities(make_model(name, params, seed=5).fit(x, y), x)
    np.testing.assert_array_equal(first, second)


def test_feature_matrix_sets(tiny_dataset) -> None:
    frame = tiny_dataset.frame
    a = feature_matrix(frame, "A", None)
    b = feature_matrix(frame, "B", tiny_dataset.priors)
    assert list(a.columns) == list(A_FEATURES)
    assert list(b.columns) == list(A_FEATURES) + list(B_FEATURES)
    with pytest.raises(ValueError):
        feature_matrix(frame, "B", None)
    with pytest.raises(ValueError):
        feature_matrix(frame, "C", None)
