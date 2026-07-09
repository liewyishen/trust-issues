"""
Tests for src/train.py

Uses small synthetic DataFrames throughout -- never the real CSV -- and does
not chase realistic AUC numbers. What's locked down here is the CONTRACT:
- the packaged .pkl round-trips through joblib and carries the feature
  contract alongside the model (the whole point of not dumping a bare model),
- train-derived categories are what val/test are encoded against, and an
  unseen category at inference time degrades gracefully instead of crashing,
- early stopping's valid_sets never includes test,
- MLflow runs are created without touching the repo's real ./mlruns.

Run:  pytest tests/test_train.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib
import mlflow

import src.train as train_module
from src.features import add_features, CATEGORICAL, FEATURES, TARGET
from src.train import train_and_save, train_lgb, load_model_artifact, _x, _xy


def _make_split(n, purposes, homes, states, emp_lengths, rng):
    return pd.DataFrame({
        "revenue": rng.uniform(20_000, 150_000, n),
        "dti_n": rng.uniform(0, 40, n),
        "loan_amnt": rng.uniform(1_000, 35_000, n),
        "fico_n": rng.uniform(620, 820, n),
        "emp_length": rng.choice(emp_lengths, n),
        "purpose": rng.choice(purposes, n),
        "home_ownership_n": rng.choice(homes, n),
        "addr_state": rng.choice(states, n),
        "Default": rng.integers(0, 2, n),
    })


@pytest.fixture
def synthetic_splits() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    purposes = ["debt_consolidation", "credit_card", "other"]
    homes = ["MORTGAGE", "RENT", "OWN"]
    emp_lengths = ["< 1 year", "5 years", "10+ years", "NI"]
    train_states = ["CA", "TX", "NY"]

    train_df = _make_split(300, purposes, homes, train_states, emp_lengths, rng)
    val_df = _make_split(80, purposes, homes, train_states, emp_lengths, rng)
    # Test includes "other_unseen", a purpose value never seen in train --
    # this is what exercises the unseen-category-at-inference-time path.
    # Uses "purpose" rather than "addr_state" deliberately: "purpose" is
    # always in CATEGORICAL (features.py's CATEGORICAL_BASE), regardless of
    # features.py's INCLUDE_ADDR_STATE toggle, so this test's behavior
    # doesn't depend on that switch's current setting the way an
    # addr_state-based unseen-category fixture would.
    test_df = _make_split(80, purposes + ["other_unseen"], homes, train_states, emp_lengths, rng)

    return {"train": train_df, "val": val_df, "test": test_df}


@pytest.fixture(autouse=True)
def _isolated_mlflow_tracking(tmp_path):
    """
    Point MLflow at a throwaway SQLite database AND a throwaway artifact
    directory for every test in this file.

    The tracking URI alone isn't enough to isolate a test run: MLflow's
    artifact store is a separate concept from the tracking backend, and
    defaults to a local ./mlruns directory relative to the current working
    directory regardless of which tracking backend (file store, SQLite,
    ...) is configured. Without also giving each test its own experiment
    with an explicit artifact_location under tmp_path, tests would still
    leak the joblib artifact into the repo's real ./mlruns even while using
    an isolated SQLite database for run metadata.

    Uses the same SQLite backend as production (src/train.py), not a file
    store -- this project no longer relies on MLFLOW_ALLOW_FILE_STORE
    anywhere, tests included.
    """
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow_test.db'}")
    experiment_id = mlflow.create_experiment(
        "test_train", artifact_location=f"file://{tmp_path / 'mlflow_artifacts'}"
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    yield


# ---------------------------------------------------------------------------
# 1 & 2. The packaged .pkl round-trips through joblib, and its category_maps
# match what train was actually encoded with.
# ---------------------------------------------------------------------------
def test_saved_model_round_trips_with_full_contract(synthetic_splits, tmp_path):
    model_path = train_and_save(
        synthetic_splits,
        model_dir=tmp_path / "models",
        num_boost_round=20,
        early_stopping_rounds=5,
    )

    assert model_path.exists()

    artifact = joblib.load(model_path)
    for key in ("model", "features", "categorical", "category_maps",
                "best_iteration", "params", "trained_at"):
        assert key in artifact

    assert artifact["features"] == FEATURES
    assert artifact["categorical"] == CATEGORICAL

    expected_categories = {
        c: sorted(pd.Categorical(add_features(synthetic_splits["train"])[c]).categories)
        for c in CATEGORICAL
    }
    loaded_categories = {c: sorted(artifact["category_maps"][c]) for c in CATEGORICAL}
    assert loaded_categories == expected_categories

    # "other_unseen" only appears in test, never in train -- it must NOT be
    # part of the train-derived category map that was packaged.
    assert "other_unseen" not in artifact["category_maps"]["purpose"]


# ---------------------------------------------------------------------------
# 3. A purpose value unseen in train ("other_unseen", present only in test)
#    does not crash prediction.
# ---------------------------------------------------------------------------
def test_unseen_test_category_does_not_crash_prediction(synthetic_splits):
    booster, best_iteration, metrics, category_maps = train_lgb(
        synthetic_splits, use_spw=False, num_boost_round=20, early_stopping_rounds=5,
    )
    assert "other_unseen" not in category_maps["purpose"]
    assert 0.0 <= metrics["test"]["auc"] <= 1.0
    assert 0.0 <= metrics["test"]["pr_auc"] <= 1.0


# ---------------------------------------------------------------------------
# 4. Early stopping's valid_sets never includes test.
# ---------------------------------------------------------------------------
def test_early_stopping_valid_sets_excludes_test(synthetic_splits, monkeypatch):
    captured = {}
    original_train = train_module.lgb.train

    def spy(params, train_set, **kwargs):
        captured["valid_names"] = kwargs.get("valid_names")
        return original_train(params, train_set, **kwargs)

    monkeypatch.setattr(train_module.lgb, "train", spy)

    train_module.train_lgb(
        synthetic_splits, use_spw=False, num_boost_round=20, early_stopping_rounds=5,
    )

    assert captured["valid_names"] == ["train", "val"]
    assert "test" not in captured["valid_names"]


# ---------------------------------------------------------------------------
# 5. MLflow runs are created (baseline + LightGBM), isolated from the real
#    ./mlruns via the autouse fixture above.
# ---------------------------------------------------------------------------
def test_train_and_save_creates_two_mlflow_runs(synthetic_splits, tmp_path):
    train_and_save(
        synthetic_splits,
        model_dir=tmp_path / "models",
        num_boost_round=20,
        early_stopping_rounds=5,
    )

    runs = mlflow.search_runs(search_all_experiments=True)
    assert len(runs) >= 2
    run_names = set(runs["tags.mlflow.runName"])
    assert "lr_baseline" in run_names
    assert "lgbm_production" in run_names


# ---------------------------------------------------------------------------
# 6. (W1) load_model_artifact enforces the packaged feature contract against
# features.py's live FEATURES/CATEGORICAL. The self-describing artifact is now
# self-ENFORCING: a model trained under a different feature set than the one
# inference will encode with (via _xy/_to_lgb_frame) fails closed, instead of
# silently scoring against a misaligned column set.
# ---------------------------------------------------------------------------
def test_load_model_artifact_accepts_matching_contract(synthetic_splits, tmp_path):
    model_path = train_and_save(
        synthetic_splits, model_dir=tmp_path / "models",
        num_boost_round=20, early_stopping_rounds=5,
    )
    artifact = load_model_artifact(model_path)   # must not raise
    assert artifact["features"] == FEATURES
    assert artifact["categorical"] == CATEGORICAL


def _dump_artifact(tmp_path, *, features, categorical):
    """A minimal packaged-model dict with a chosen contract. load_model_artifact
    validates the contract before touching the model, so model=None is fine."""
    art = {
        "model": None, "features": features, "categorical": categorical,
        "category_maps": {}, "best_iteration": 1, "params": {}, "trained_at": "x",
    }
    path = tmp_path / "artifact.pkl"
    joblib.dump(art, path)
    return path


def test_load_model_artifact_rejects_feature_contract_mismatch(tmp_path):
    """A model packaged under an extra feature fails closed -- the scenario where
    FEATURES changed between train time and score time."""
    path = _dump_artifact(tmp_path, features=FEATURES + ["ghost_feature"], categorical=CATEGORICAL)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_model_artifact(path)


def test_load_model_artifact_rejects_categorical_mismatch(tmp_path):
    """Same guard on the categorical contract -- e.g. INCLUDE_ADDR_STATE flipped
    on between train and score, so addr_state is now expected but the model
    never saw it."""
    path = _dump_artifact(tmp_path, features=FEATURES, categorical=CATEGORICAL + ["addr_state"])
    with pytest.raises(ValueError, match="contract mismatch"):
        load_model_artifact(path)


# ---------------------------------------------------------------------------
# 7. _x encodes a LABEL-FREE frame. _xy cannot: it indexes engineered[TARGET],
# so it raises KeyError on any frame without a `Default` column -- which is
# every live serving request. _x is the same feature-engineering path with the
# target extraction removed, so src/explain.py can share ONE encoding path with
# training instead of writing a second one.
# ---------------------------------------------------------------------------
def test_x_encodes_a_frame_with_no_target_column(synthetic_splits):
    """The whole point of _x: no `Default` column, no raise, FEATURES out."""
    unlabeled = synthetic_splits["test"].drop(columns=[TARGET])
    assert TARGET not in unlabeled.columns

    X = _x(unlabeled)                       # must not raise

    assert list(X.columns) == FEATURES
    assert len(X) == len(unlabeled)
    assert TARGET not in X.columns


def test_xy_still_requires_the_target(synthetic_splits):
    """The other side of the guard: _xy is unchanged and still raises on the
    same frame. If _xy ever stops needing the target, _x has no reason to
    exist and this test says so."""
    unlabeled = synthetic_splits["test"].drop(columns=[TARGET])
    with pytest.raises(KeyError):
        _xy(unlabeled)


def test_x_matches_xy_s_features_on_a_labeled_frame(synthetic_splits):
    """_x is not a SECOND encoding path -- on a frame both accept, they produce
    the identical X. This is the assertion that would fail if the two ever
    drifted apart."""
    labeled = synthetic_splits["test"]
    X_only = _x(labeled)
    X_from_xy, _y = _xy(labeled)
    pd.testing.assert_frame_equal(X_only, X_from_xy)
