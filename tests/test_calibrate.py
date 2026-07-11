"""
Tests for src/calibrate.py

Doesn't chase the real dataset's Brier/mean_pred numbers -- those only mean
something at real scale. What's locked down here is the CALIBRATION
CONTRACT: fitting on Calib and applying to Test genuinely pulls mean
predicted probability toward the actual rate, does so without reordering
predictions (AUC is preserved), and the calibrator round-trips through
joblib the same way train.py's model artifact does.

To get a meaningfully miscalibrated raw model without real data, the model
artifact used here is trained WITH scale_pos_weight (via model_io.py's own
train_lgb(..., use_spw=True)) -- the exact mechanism notebooks/analysis.ipynb
Cell 25 and model_io.py's run_spw_ablation() identify as inflating mean
predicted probability without improving ranking. That gives calibration
something real to correct.

Run:  pytest tests/test_calibrate.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib

from src.features import CATEGORICAL, FEATURES
from src.model_io import train_lgb
from src.calibrate import calibrate_model, apply_calibration, load_calibrator


def _make_split(n, purposes, homes, states, emp_lengths, rng, base_default_rate=0.15):
    fico = rng.uniform(620, 820, n)
    dti = rng.uniform(0, 40, n)

    # Genuine (if simplistic) signal -- higher DTI / lower FICO -> higher
    # default probability -- so early stopping has something real to chase
    # instead of plateauing after a handful of rounds. That gives
    # scale_pos_weight's mean_pred-inflating effect (see module docstring)
    # room to actually show up, the same way it does over the notebook's
    # 274 real boosting rounds.
    z = 0.05 * (dti - 20) - 0.03 * (fico - 700)
    p_default = 0.5 / (1 + np.exp(-z)) + 0.5 * base_default_rate
    default = rng.binomial(1, p_default)

    return pd.DataFrame({
        "revenue": rng.uniform(20_000, 150_000, n),
        "dti_n": dti,
        "loan_amnt": rng.uniform(1_000, 35_000, n),
        "fico_n": fico,
        "emp_length": rng.choice(emp_lengths, n),
        "purpose": rng.choice(purposes, n),
        "home_ownership_n": rng.choice(homes, n),
        "addr_state": rng.choice(states, n),
        "Default": default,
    })


@pytest.fixture
def synthetic_splits() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    purposes = ["debt_consolidation", "credit_card", "other"]
    homes = ["MORTGAGE", "RENT", "OWN"]
    emp_lengths = ["< 1 year", "5 years", "10+ years", "NI"]
    states = ["CA", "TX", "NY"]

    return {
        "train": _make_split(800, purposes, homes, states, emp_lengths, rng),
        "val": _make_split(200, purposes, homes, states, emp_lengths, rng),
        # Calib and Test are both sized generously (relative to this
        # synthetic scale) on purpose: isotonic regression fit on too few
        # Calib points produces a coarse, few-step staircase, which then
        # introduces artificial ties (and a bigger AUC shift) among Test
        # points that had genuinely different raw scores -- an artifact of
        # small-sample calibration, not of the calibration LOGIC being
        # tested here. More Calib/Test data keeps that noise small enough
        # for the AUC-preservation assertion to be meaningful.
        "calib": _make_split(500, purposes, homes, states, emp_lengths, rng),
        "test": _make_split(300, purposes, homes, states, emp_lengths, rng),
    }


def _make_model_artifact(splits):
    """
    A deliberately miscalibrated model: scale_pos_weight=True skews mean
    predicted probability away from the actual default rate without
    improving ranking (see module docstring) -- exactly the failure mode
    calibration is supposed to correct.
    """
    booster, best_iteration, _metrics, category_maps = train_lgb(
        splits, use_spw=True, num_boost_round=100, early_stopping_rounds=15,
    )
    return {
        "model": booster,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "category_maps": category_maps,
        "best_iteration": best_iteration,
        "params": {},
        "trained_at": "test-fixture",
    }


@pytest.fixture
def model_path(synthetic_splits, tmp_path):
    artifact = _make_model_artifact(synthetic_splits)
    path = tmp_path / "model.pkl"
    joblib.dump(artifact, path)
    return path


# ---------------------------------------------------------------------------
# 1. Calibration pulls mean_pred closer to the actual test default rate.
# ---------------------------------------------------------------------------
def test_calibration_moves_mean_pred_closer_to_actual(synthetic_splits, model_path, tmp_path):
    iso, metrics = calibrate_model(
        model_path=model_path,
        splits=synthetic_splits,
        calibrator_path=tmp_path / "calibrator.pkl",
    )
    actual = metrics["actual_test_default_rate"]
    raw_gap = abs(metrics["raw"]["mean_pred"] - actual)
    cal_gap = abs(metrics["calibrated"]["mean_pred"] - actual)
    assert cal_gap < raw_gap


# ---------------------------------------------------------------------------
# 2. Isotonic is monotonic: AUC barely moves.
# ---------------------------------------------------------------------------
def test_calibration_preserves_auc(synthetic_splits, model_path, tmp_path):
    iso, metrics = calibrate_model(
        model_path=model_path,
        splits=synthetic_splits,
        calibrator_path=tmp_path / "calibrator.pkl",
    )
    assert abs(metrics["raw"]["auc"] - metrics["calibrated"]["auc"]) < 0.01


# ---------------------------------------------------------------------------
# 3. apply_calibration always returns valid probabilities.
# ---------------------------------------------------------------------------
def test_apply_calibration_output_in_unit_interval(synthetic_splits, model_path, tmp_path):
    iso, _metrics = calibrate_model(
        model_path=model_path,
        splits=synthetic_splits,
        calibrator_path=tmp_path / "calibrator.pkl",
    )
    raw_probs = np.array([-1.0, 0.0, 0.3, 0.7, 1.0, 2.0])
    calibrated = apply_calibration(iso, raw_probs)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


# ---------------------------------------------------------------------------
# 4. The saved calibrator round-trips through joblib and reproduces the
#    same transform as the original.
# ---------------------------------------------------------------------------
def test_calibrator_persists_and_reloads(synthetic_splits, model_path, tmp_path):
    calibrator_path = tmp_path / "calibrator.pkl"
    iso, _metrics = calibrate_model(
        model_path=model_path,
        splits=synthetic_splits,
        calibrator_path=calibrator_path,
    )

    assert calibrator_path.exists()
    reloaded = load_calibrator(calibrator_path)

    probe = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    np.testing.assert_allclose(
        apply_calibration(iso, probe), apply_calibration(reloaded, probe),
    )


# ---------------------------------------------------------------------------
# 5. (W1) calibrate_model is a real inference consumer of the packaged model,
# so it must fail closed when the model's feature contract no longer matches
# features.py's live one -- not score through a misaligned column set.
# ---------------------------------------------------------------------------
def test_calibrate_model_fails_closed_on_contract_mismatch(synthetic_splits, tmp_path):
    bad = {
        "model": None, "features": FEATURES + ["ghost"], "categorical": CATEGORICAL,
        "category_maps": {}, "best_iteration": 1, "params": {}, "trained_at": "x",
    }
    bad_path = tmp_path / "bad_model.pkl"
    joblib.dump(bad, bad_path)
    with pytest.raises(ValueError, match="contract mismatch"):
        calibrate_model(
            model_path=bad_path, splits=synthetic_splits,
            calibrator_path=tmp_path / "cal.pkl",
        )


# ---------------------------------------------------------------------------
# 6. (W2) load_calibrator enforces the calibrator<->model binding it records,
# instead of writing it and discarding it on load. Applying a calibrator to a
# different (e.g. retrained) model than it was fit against now fails closed.
# ---------------------------------------------------------------------------
def test_load_calibrator_enforces_model_binding(tmp_path):
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.0, 1.0])
    cal_art = {
        "calibrator": iso, "model_path": "models/lgbm_model.pkl",
        "model_trained_at": "model-A", "trained_at": "cal-time",
    }
    path = tmp_path / "cal.pkl"
    joblib.dump(cal_art, path)

    # matching model -> returns the calibrator
    assert load_calibrator(path, model_artifact={"trained_at": "model-A"}) is not None
    # stale model (different trained_at) -> fails closed
    with pytest.raises(ValueError, match="[Ss]tale calibrator"):
        load_calibrator(path, model_artifact={"trained_at": "model-B"})
    # no model given -> lenient bare load (inspection / tests)
    assert load_calibrator(path) is not None


def test_calibrator_binding_survives_real_round_trip(synthetic_splits, model_path, tmp_path):
    """End to end: a calibrator produced by calibrate_model records the model's
    trained_at, and load_calibrator then rejects a model whose trained_at differs
    -- the retrain-and-forget-to-recalibrate failure the packaging note warns
    about, now actually guarded."""
    cal_path = tmp_path / "calibrator.pkl"
    calibrate_model(model_path=model_path, splits=synthetic_splits, calibrator_path=cal_path)

    model_artifact = joblib.load(model_path)          # trained_at == "test-fixture"
    assert load_calibrator(cal_path, model_artifact=model_artifact) is not None

    stale = dict(model_artifact)
    stale["trained_at"] = "some-other-model"
    with pytest.raises(ValueError, match="[Ss]tale calibrator"):
        load_calibrator(cal_path, model_artifact=stale)
