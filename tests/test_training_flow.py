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
import pytest
from metaflow import FlowSpec

from pipelines.training_flow import (
    TrainingFlow,
    calib_scalars,
    eval_scalars,
    fairness_scalars,
    feature_date_columns,
    guarded_importance,
    importance_metrics,
)
from src.features import FEATURES


# --- Structure -----------------------------------------------------------------

def test_flow_is_a_flowspec():
    assert issubclass(TrainingFlow, FlowSpec)


def test_flow_has_expected_linear_steps():
    # The pipeline is load -> train -> calibrate -> evaluate -> fairness ->
    # explain, framed by Metaflow's mandatory start/end.
    for name in ["start", "train", "calibrate", "evaluate", "fairness", "explain", "end"]:
        assert callable(getattr(TrainingFlow, name)), f"missing step: {name}"


def test_flow_has_explain_step_between_fairness_and_end():
    """explain is a real inserted node on the DAG -- fairness -> explain -> end,
    not an orphan method. Read from Metaflow's own graph, so a broken
    self.next() rewiring fails HERE rather than only at a full flow run."""
    from metaflow.graph import FlowGraph

    nodes = FlowGraph(TrainingFlow).nodes
    assert nodes["fairness"].out_funcs == ["explain"]
    assert nodes["explain"].out_funcs == ["end"]


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


# --- feature_date_columns (temporal-consistency sentinel arming) ---------------

def test_feature_date_columns_empty_on_cleaned_frame():
    """The cleaned dataset's shape: issue_d + issue_year + features, no other
    date columns -- the sentinel must report 'nothing to check', not a false
    positive on issue_d/issue_year themselves."""
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2016-01-01"]),
        "issue_year": [2016],
        "fico_n": [700.0],
    })
    assert feature_date_columns(df) == []


def test_feature_date_columns_detects_lc_naming_convention():
    """A *_d column self-arms the sentinel even when it arrives as raw string."""
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2016-01-01"]),
        "last_credit_pull_d": ["2016-06-01"],
    })
    assert feature_date_columns(df) == ["last_credit_pull_d"]


def test_feature_date_columns_detects_datetime_dtype():
    """A datetime column self-arms the sentinel even without the *_d name."""
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2016-01-01"]),
        "some_event_time": pd.to_datetime(["2016-06-01"]),
    })
    assert feature_date_columns(df) == ["some_event_time"]


# --- scalars are MLflow-loggable ----------------------------------------------

def test_extracted_scalars_are_finite_floats():
    metrics = {
        "actual_test_default_rate": 0.23,
        "raw": {"brier": 0.17, "mean_pred": 0.17, "auc": 0.66},
        "calibrated": {"brier": 0.16, "mean_pred": 0.19, "auc": 0.66},
    }
    for v in calib_scalars(metrics).values():
        assert math.isfinite(v)


# --- importance_metrics (pure extractor, like calib_scalars) -------------------

def test_importance_metrics_keys_and_finite_floats():
    """global_importance()'s frame -> one shap_importance_<feature> metric per
    FEATURE, every value a finite float, the feature->value mapping preserved."""
    values = {feat: 0.1 * (i + 1) for i, feat in enumerate(FEATURES)}
    imp = pd.DataFrame({"feature": list(values), "mean_abs_shap": list(values.values())})

    out = importance_metrics(imp)

    assert set(out) == {f"shap_importance_{f}" for f in FEATURES}
    assert len(out) == len(FEATURES) == 8
    assert all(isinstance(v, float) and math.isfinite(v) for v in out.values())
    for feat, val in values.items():
        assert out[f"shap_importance_{feat}"] == val


# --- guarded_importance: fail-OPEN over ALL THREE allowlisted types ------------
# _EXPLAIN_SKIP_ERRORS is a positive allowlist of three types (ImportError,
# MemoryError, OSError). Pinning all three -- not just one -- documents the
# allowlist itself: three passing types is the contract, one is only a sample of
# it. If a type is ever dropped from the allowlist, its test flips from
# skip-reason to raised-exception and fails here.

def test_guarded_importance_returns_metrics_on_success():
    imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": [0.1] * len(FEATURES)})
    metrics, reason = guarded_importance(lambda: imp)
    assert reason is None
    assert len(metrics) == len(FEATURES)


def test_guarded_importance_skips_on_importerror_not_raises():
    """A missing/moved shap (ImportError) becomes a SKIP reason, never a raise --
    the flow must survive an unavailable explainer, not die for one."""
    def boom():
        raise ImportError("No module named 'shap'")

    metrics, reason = guarded_importance(boom)
    assert metrics is None
    assert reason is not None and reason.startswith("ImportError")


def test_guarded_importance_skips_on_memoryerror():
    """The 4000 x trees SHAP matrix not fitting (MemoryError) is a resource
    failure, also fail-open -- the model is intact, the explainer just cannot
    run here right now."""
    def boom():
        raise MemoryError()

    metrics, reason = guarded_importance(boom)
    assert metrics is None
    assert reason is not None and reason.startswith("MemoryError")


def test_guarded_importance_skips_on_oserror():
    """An artifact file unreadable this instant (OSError) is fail-open too. Raised
    here as FileNotFoundError -- an OSError SUBCLASS -- to pin that the allowlist
    catches the whole OSError family, which is how the real failure (a transient
    I/O race; a genuinely missing artifact would already have hard-stopped
    calibrate) actually surfaces."""
    def boom():
        raise FileNotFoundError("models/lgbm_model.pkl unreadable mid-run")

    metrics, reason = guarded_importance(boom)
    assert metrics is None
    assert reason is not None and reason.startswith("FileNotFoundError")


# --- guarded_importance: fail-CLOSED boundary ----------------------------------

def test_guarded_importance_does_not_swallow_additivity_valueerror():
    """The inner additivity guard's ValueError (src/explain.py:221) is NOT in
    _EXPLAIN_SKIP_ERRORS, so it propagates and hard-stops the flow: a
    wrong-but-numeric ranking must be neither logged nor hidden."""
    def boom():
        raise ValueError("Additivity check failed -- refusing to explain.")

    with pytest.raises(ValueError, match="Additivity check failed"):
        guarded_importance(boom)


def test_guarded_importance_does_not_swallow_typeerror():
    """A class that could be a shap API drift OR an ordinary bug (TypeError)
    propagates rather than silently skipping -- the fail-open allowlist admits
    only unambiguous resource/availability failures, never a maybe-a-bug."""
    def boom():
        raise TypeError("unexpected shap return contract")

    with pytest.raises(TypeError):
        guarded_importance(boom)
