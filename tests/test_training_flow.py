"""
Lightweight smoke tests for pipelines/training_flow.py.

A Metaflow FlowSpec is awkward to exercise end-to-end in a unit test (it drives
its own CLI + subprocess-per-step machinery, and a real run needs the 167 MB
CSV), so this file does NOT run the flow. It instead verifies the two things
that CAN be checked cheaply and deterministically:

  1. The module imports without error, is a proper FlowSpec, and exposes the
     expected linear step methods (import wiring + sys.path shim are sound).
  2. The pure helper functions that pull scalars out of the src functions'
     return dicts -- the actual MLflow-logging logic -- do the right thing.

Full end-to-end coverage is the `uv run python pipelines/training_flow.py run`
smoke run, not a pytest.
"""

from __future__ import annotations

import math

import pandas as pd
from metaflow import FlowSpec

from pipelines.training_flow import (
    TrainingFlow,
    calib_scalars,
    eval_scalars,
    fairness_scalars,
)


# --- Structure -----------------------------------------------------------------

def test_flow_is_a_flowspec():
    assert issubclass(TrainingFlow, FlowSpec)


def test_flow_has_expected_linear_steps():
    # The pipeline is load -> train -> calibrate -> evaluate -> fairness, framed
    # by Metaflow's mandatory start/end.
    for name in ["start", "train", "calibrate", "evaluate", "fairness", "end"]:
        assert callable(getattr(TrainingFlow, name)), f"missing step: {name}"


# --- calib_scalars -------------------------------------------------------------

def test_calib_scalars_flattens_nested_dict():
    metrics = {
        "actual_test_default_rate": 0.2323,
        "raw": {"brier": 0.1717, "mean_pred": 0.1705, "auc": 0.6660},
        "calibrated": {"brier": 0.1692, "mean_pred": 0.1915, "auc": 0.6654},
    }
    out = calib_scalars(metrics)
    assert out == {
        "brier_raw": 0.1717,
        "brier_cal": 0.1692,
        "mean_pred_raw": 0.1705,
        "mean_pred_cal": 0.1915,
        "auc_raw": 0.6660,
        "auc_cal": 0.6654,
        "actual_test_default_rate": 0.2323,
    }
    assert all(isinstance(v, float) for v in out.values())


# --- eval_scalars --------------------------------------------------------------

def test_eval_scalars_drops_val_profit_curve():
    result = {
        "best_threshold": 0.25,
        "val_profit_curve": pd.DataFrame({"threshold": [0.1, 0.2], "val_profit": [1.0, 2.0]}),
        "threshold": 0.25,
        "test_profit": -1.0e8,
        "naive_profit": -2.848e8,
        "naive_threshold": 0.50,
        "improvement_over_naive": 1.848e8,
        "approval_rate": 0.78,
        "bad_rate_approved": 0.189,
        "bad_rate_rejected": 0.385,
    }
    out = eval_scalars(result)
    # The DataFrame must not leak into the scalar summary.
    assert "val_profit_curve" not in out
    assert set(out) == {
        "best_threshold", "test_profit", "naive_profit", "improvement_over_naive",
        "approval_rate", "bad_rate_approved", "bad_rate_rejected",
    }
    assert out["best_threshold"] == 0.25
    assert out["approval_rate"] == 0.78
    assert all(isinstance(v, float) for v in out.values())


# --- fairness_scalars ----------------------------------------------------------

def _fairness_result(verdicts):
    return {
        "fair_df": pd.DataFrame({"addr_state": ["CA"], "y_true": [0], "p": [0.1]}),
        "layer1": pd.DataFrame({"state": [f"S{i}" for i in range(len(verdicts))],
                                "verdict": verdicts}),
        "layer2": pd.DataFrame({"state": ["MS"], "eo": [0.7]}),
        "layer3": {"auc_with_state": 0.6689, "auc_no_state": 0.6654,
                   "auc_cost": -0.0035, "states": pd.DataFrame({"state": ["MS"]})},
    }


def test_fairness_scalars_counts_confirmed_states():
    result = _fairness_result([
        "confirmed geographic-proxy risk",
        "confirmed proxy",
        "clear",
        "inconclusive",
    ])
    out = fairness_scalars(result)
    assert out["n_confirmed_proxy_states"] == 2.0
    assert out["auc_with_state"] == 0.6689
    assert out["auc_no_state"] == 0.6654
    assert out["auc_cost"] == -0.0035
    assert all(isinstance(v, float) for v in out.values())


def test_fairness_scalars_zero_confirmed():
    result = _fairness_result(["clear", "clear", "inconclusive"])
    out = fairness_scalars(result)
    assert out["n_confirmed_proxy_states"] == 0.0


def test_fairness_scalars_handles_nan_verdict():
    # A NaN verdict cell must not blow up the confirmed-count (na=False).
    result = _fairness_result(["confirmed proxy", float("nan"), "clear"])
    out = fairness_scalars(result)
    assert out["n_confirmed_proxy_states"] == 1.0


# --- scalars are MLflow-loggable ----------------------------------------------

def test_extracted_scalars_are_finite_floats():
    metrics = {
        "actual_test_default_rate": 0.23,
        "raw": {"brier": 0.17, "mean_pred": 0.17, "auc": 0.66},
        "calibrated": {"brier": 0.16, "mean_pred": 0.19, "auc": 0.66},
    }
    for v in calib_scalars(metrics).values():
        assert math.isfinite(v)
