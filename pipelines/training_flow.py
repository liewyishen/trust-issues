"""
End-to-end LendingClub credit-default training pipeline (Metaflow).

Chains src/'s modeling layer into one linear FlowSpec and unifies the key
metrics from train / calibrate / evaluate / fairness / explain into a SINGLE
MLflow run (the production LightGBM run), so one model execution produces one
complete experiment archive instead of numbers scattered across the terminal.

No src/ module is modified. Every step calls the existing entrypoints -- which
already accept splits=/model_path= and return metrics dicts -- and the MLflow
logging is done here, in the pipeline layer.

MLflow unification strategy (Option A -- recency lookup)
-------------------------------------------------------
train_and_save() opens its OWN two runs (lr_baseline + lgbm_production) and
closes them. We deliberately do NOT wrap it in an outer run: train_and_save's
start_run() calls omit nested=True, so an already-active outer run would make
MLflow raise "Run already active". Instead, right after training we locate THIS
execution's lgbm_production run by recency (the most recent run with that name
in the lc_default_risk experiment) and, in the calibrate/evaluate/fairness
steps, reopen it by run_id to APPEND their metrics onto it.

Every downstream metric describes the production model, so the lgbm_production
run becomes the full archive (train AUC + calibration Brier + profit + fairness
+ global SHAP importance) while the lr_baseline run stays as the untouched
comparison point.

Recency is sufficient for single-user local runs. A concurrent / multi-user
setup would instead have train_and_save return its run_id directly (a src
change, out of scope for this pipeline layer).

Run it:
  uv run python pipelines/training_flow.py run
Browse the unified run:
  mlflow ui --backend-store-uri sqlite:///mlflow.db   (experiment: lc_default_risk)
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

# pipelines/ lives beside src/, not inside it. When Metaflow launches this file
# as a script -- and re-launches it in a fresh subprocess for every step --
# sys.path[0] is pipelines/, not the repo root, so `import src...` would fail.
# Put the repo root on the path before importing anything from src. This runs at
# module top level, so each per-step subprocess re-applies it on import.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from metaflow import FlowSpec, step  # noqa: E402

EXPERIMENT_NAME = "lc_default_risk"
# The same absolute sqlite path src/train.py pins, so pipeline runs and any
# direct src.train runs write to one database regardless of the caller's cwd.
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


# --- MLflow plumbing -----------------------------------------------------------

def configure_mlflow() -> None:
    """
    Point MLflow at the repo's sqlite backend and the pipeline experiment.

    Called at the top of EVERY step that touches MLflow: Metaflow runs each step
    in its own subprocess, and MLflow's tracking URI / active experiment are
    process-global state that does not survive across those subprocess
    boundaries. Setting it once in start() would not carry into train().
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def find_latest_run_id(run_name: str, experiment_name: str = EXPERIMENT_NAME) -> str:
    """
    Return the run_id of the most recent run named `run_name` in the experiment.

    This is the "Option A" recency lookup used to locate the lgbm_production run
    that train_and_save() just created (it hands back a model Path, not a
    run_id). Ordered by start_time DESC so the run from THIS execution wins.
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise RuntimeError(
            f"No MLflow run named {run_name!r} found in experiment "
            f"{experiment_name!r} -- did the train step run and set the experiment?"
        )
    return str(runs.iloc[0]["run_id"])


def log_metrics_to_run(run_id: str, metrics: dict[str, float]) -> None:
    """
    Append metrics to an existing (already-finished) run by reopening it by id.

    MLflow lets a finished run be reactivated via start_run(run_id=...); logging
    appends and the context manager re-closes it. Non-finite values (e.g. a
    bad_rate over an empty approved set) are skipped rather than logged as NaN.
    """
    with mlflow.start_run(run_id=run_id):
        for key, value in metrics.items():
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                continue
            mlflow.log_metric(key, float(value))


def log_params_to_run(run_id: str, params: dict[str, object]) -> None:
    """
    Append params to an existing run by reopening it by id -- the sibling of
    log_metrics_to_run() for the CONFIGURATION behind a metric (which sample,
    which seed, which scale), so a logged number stays interpretable and
    reproducible rather than a bare float of unknown provenance.

    MLflow lets a finished run be reactivated via start_run(run_id=...); logging
    appends and the context manager re-closes it. Reopening a second time in the
    same step (params here, metrics next) is the pattern the calibrate / evaluate
    / fairness steps already use to append onto this run.
    """
    with mlflow.start_run(run_id=run_id):
        for key, value in params.items():
            mlflow.log_param(key, value)


# --- Leakage-sentinel helpers ----------------------------------------------------

def feature_date_columns(df: pd.DataFrame) -> list[str]:
    """
    Candidate feature-timestamp columns for the temporal-consistency sentinel.

    A column qualifies if it is datetime-typed or follows LendingClub's `*_d`
    date-column naming convention -- except issue_d itself, which is the
    decision date every feature timestamp gets compared against (issue_year is
    an int and matches neither rule).

    On the cleaned dataset this returns [] -- the known post-origination date
    fields were removed upstream, and prove_forbidden_absent() proves it every
    run. That empty list is what makes the start step's SKIPPED log honest:
    the sentinel states WHY it has nothing to check, and self-arms the moment
    a future join reintroduces any date column. Pure function, unit-testable
    without a flow run.
    """
    return [
        c for c in df.columns
        if c != "issue_d"
        and (pd.api.types.is_datetime64_any_dtype(df[c]) or c.endswith("_d"))
    ]


# --- Scalar extraction helpers -------------------------------------------------
# These pull ONLY flat scalars out of the src functions' return dicts, so that
# (a) MLflow receives clean float metrics and (b) large objects
# (val_profit_curve, fair_df, the layer DataFrames) never get pickled onto self
# as Metaflow artifacts. Pure functions, so they're unit-testable without a run.

def calib_scalars(metrics: dict) -> dict[str, float]:
    """Flatten calibrate_model()'s nested raw/calibrated metrics into scalars."""
    return {
        "brier_raw": float(metrics["raw"]["brier"]),
        "brier_cal": float(metrics["calibrated"]["brier"]),
        "mean_pred_raw": float(metrics["raw"]["mean_pred"]),
        "mean_pred_cal": float(metrics["calibrated"]["mean_pred"]),
        "auc_raw": float(metrics["raw"]["auc"]),
        "auc_cal": float(metrics["calibrated"]["auc"]),
        "actual_test_default_rate": float(metrics["actual_test_default_rate"]),
    }


def eval_scalars(result: dict) -> dict[str, float]:
    """
    Pull the scalar profit/approval fields out of run_evaluation()'s dict.

    Deliberately excludes result["val_profit_curve"] (a DataFrame) -- only
    scalars cross into self / MLflow. The curve, if wanted, would be a
    mlflow.log_artifact, not a flow artifact.
    """
    keys = [
        "best_threshold", "test_profit", "naive_profit", "improvement_over_naive",
        "approval_rate", "bad_rate_approved", "bad_rate_rejected",
    ]
    return {k: float(result[k]) for k in keys}


def fairness_scalars(result: dict) -> dict[str, float]:
    """
    Reduce run_fairness_audit()'s dict to scalars.

    Layer-3 AUC costs plus a count of Layer-1 states confirmed as
    geographic-proxy risk (verdict starting with "confirmed"). Drops fair_df
    (~Test-sized) and the layer DataFrames -- none of them touch self.
    """
    layer3 = result["layer3"]
    layer1 = result["layer1"]
    n_confirmed = int(layer1["verdict"].str.startswith("confirmed", na=False).sum())
    return {
        "auc_with_state": float(layer3["auc_with_state"]),
        "auc_no_state": float(layer3["auc_no_state"]),
        "auc_cost": float(layer3["auc_cost"]),
        "n_confirmed_proxy_states": float(n_confirmed),
    }


# --- Explain-step helpers ------------------------------------------------------
# global_importance() is an OBSERVABILITY signal, not a correctness gate, so the
# explain step is fail-OPEN: a broken explainer must not throw away a run whose
# model is already trained, calibrated, evaluated, and fairness-audited. These
# three helpers carry that policy and are pure enough to unit-test without a
# Metaflow run or a real shap install (see tests/test_training_flow.py).

def importance_metrics(imp: pd.DataFrame) -> dict[str, float]:
    """
    global_importance()'s frame -> flat MLflow metrics keyed shap_importance_*.

    `imp` has columns feature, mean_abs_shap (src/explain.py:614), one row per
    model FEATURE, in CONTRIBUTION_SCALE = "log_odds_margin" units. Pure; it
    imports no shap, so it stays testable on a hand-built frame.

    The RANK of these importances is stable across sampling: seeds 42/7/99 at
    n=4000, plus n=40000 and full Test (462,174 rows), all produce the identical
    top-to-bottom order (re-measured 2026-07-11). The VALUES are NOT exact -- at
    n=4000 the per-feature mean|SHAP| carries ~1-7% jitter across seeds (0.8% on
    dti_n and purpose, up to 5.1% on emp_length_ord and 6.8% on
    emp_length_missing, the two smallest). So log them, but do NOT compare
    shap_importance_* run-over-run as though exact: the shap_sample_n and
    shap_seed params the explain step logs are what make a given run's values
    reproducible. Raise the sample, or run full Test (~33 s), if you ever need a
    cross-run value comparison rather than a stable ranking.
    """
    return {
        f"shap_importance_{row.feature}": float(row.mean_abs_shap)
        for row in imp.itertuples(index=False)
    }


# The fail-OPEN allowlist for the explain step. Each member means "the model is
# intact, the explainer just cannot run in this environment right now" -- never
# "a wrong number was produced":
#   ImportError  (incl. ModuleNotFoundError) -- shap uninstalled / a submodule
#                moved; the explainer cannot even be constructed.
#   MemoryError  -- the 4000 x trees SHAP matrix did not fit; a resource limit.
#   OSError      (incl. FileNotFoundError, PermissionError) -- the artifact file
#                could not be read this instant. A genuinely missing/corrupt
#                artifact would already have hard-stopped the calibrate step (it
#                loads through the same load_model_artifact), so at THIS step an
#                OSError is a transient I/O race, not a real absence.
# Everything NOT listed propagates and hard-stops the flow -- crucially
# ValueError, which _assert_additivity raises on a bad ranking (src/explain.py:
# 221): a wrong-but-numeric ranking is a correctness failure that must be neither
# logged nor hidden. AttributeError / TypeError propagate too: a shap API drift
# could surface as one, but so does an ordinary bug, and swallowing them would
# silently blind the step. The boundary is drawn on an allowlist, not on
# exception type, precisely because _assert_additivity and load_model_artifact
# both raise a bare ValueError (src/explain.py:221, src/train.py:526).
_EXPLAIN_SKIP_ERRORS = (ImportError, MemoryError, OSError)


def guarded_importance(
    compute: Callable[[], pd.DataFrame],
) -> tuple[dict[str, float] | None, str | None]:
    """
    Run compute() -- which returns global_importance()'s frame -- and reduce it
    to metrics, failing OPEN over _EXPLAIN_SKIP_ERRORS only.

    Returns (metrics, None) on success, or (None, reason) when compute raised a
    skip-class error, so the caller logs a visible SKIP and continues to end()
    rather than letting the flow die for an explainability failure. Any other
    exception -- the additivity ValueError above all -- propagates unchanged:
    the inner fail-closed guard is not swallowed here.
    """
    try:
        return importance_metrics(compute()), None
    except _EXPLAIN_SKIP_ERRORS as exc:
        return None, f"{type(exc).__name__}: {exc}"


# --- The flow ------------------------------------------------------------------

class TrainingFlow(FlowSpec):
    """
    Linear pipeline: load+split -> train -> calibrate -> evaluate -> fairness
    -> explain.

    self.splits is loaded ONCE in start() and reused by every downstream step
    (Metaflow pickles it into its datastore between steps). This is a few
    hundred MB of DataFrames -- an accepted local overhead, and far cheaper than
    re-reading the 167 MB CSV in each step. Only scalar summaries are stored on
    self after that; the large DataFrames the src functions return stay local to
    their step.
    """

    @step
    def start(self):
        """Load + validate the data, run all four leakage sentinels, build the splits."""
        from src.data_loader import load_raw, temporal_split
        from src.features import CATEGORICAL, FEATURES, TARGET, add_features
        from src.leakage_check import (
            check_forbidden_features,
            check_single_feature_auc,
            check_temporal_consistency,
            prove_forbidden_absent,
        )

        configure_mlflow()

        df = load_raw()  # data_validation runs inside load_raw() (fail-closed)

        # Sentinels 1+2 of leakage_check, fail-closed: both raise ValueError on
        # any violation, which aborts the flow before any compute is spent on
        # training. Check the model's declared FEATURES, and prove the forbidden
        # columns are truly absent from the loaded frame (catches a
        # reintroduction via a bad join).
        print(check_forbidden_features(FEATURES))
        report = prove_forbidden_absent(df)
        print(
            f"Leakage gate passed: {len(report['confirmed_absent'])} known "
            "post-decision fields confirmed absent."
        )

        # Sentinel 3: temporal consistency -- no feature timestamp may post-date
        # the decision date. On the cleaned dataset there is usually nothing to
        # compare (the post-origination date fields are exactly the ones
        # sentinel 2 just proved absent), so the common path is a SKIP -- but an
        # explicit, logged one, never a silent absence. The scan self-arms the
        # moment a future join brings any date column back in, and violations
        # then fail the flow with the same closed semantics as sentinels 1+2.
        date_cols = feature_date_columns(df)
        if date_cols:
            for col in date_cols:
                bad = check_temporal_consistency(df, col, "issue_d")
                if not bad.empty:
                    raise ValueError(
                        f"Temporal leakage: {len(bad)} rows have {col} strictly "
                        "after issue_d. A feature computed from post-decision "
                        "data leaks the future -- see check_temporal_consistency "
                        "in leakage_check.py."
                    )
            print(
                f"Temporal-consistency gate passed: {date_cols} never "
                "post-date issue_d."
            )
        else:
            print(
                "Temporal-consistency check SKIPPED (explicitly, not silently): "
                "the cleaned dataset has no feature timestamp column to compare "
                "against issue_d -- the known post-origination date fields are "
                "all in DEFAULT_FORBIDDEN and were proven absent above. This "
                "sentinel self-arms if a future join reintroduces a date column."
            )

        self.splits = temporal_split(df)  # stored ONCE; reused by every step

        # Sentinel 4 of leakage_check: the standalone-AUC red flag, run on the
        # exact feature frame the model will train on. A single feature with
        # standalone AUC > 0.90 is almost certainly leaking the target
        # (legitimate signals rarely exceed ~0.75), and the check is model-free
        # -- it needs only the data -- so it runs HERE, before any training
        # compute is spent, with the same fail-closed semantics as the gates
        # above.
        print(check_single_feature_auc(
            add_features(self.splits["train"]), FEATURES, TARGET,
            categorical=CATEGORICAL,
        ))

        self.next(self.train)

    @step
    def train(self):
        """Train baseline + production LightGBM; locate the run to unify into."""
        from src.train import train_and_save

        configure_mlflow()  # so train_and_save's two runs land in lc_default_risk
        self.model_path = str(train_and_save(self.splits))

        # Option A recency lookup: find THIS execution's lgbm_production run so
        # the calibrate/evaluate/fairness steps can append onto it.
        self.prod_run_id = find_latest_run_id("lgbm_production")
        print(f"Model saved: {self.model_path}")
        print(f"Unified MLflow run (lgbm_production): {self.prod_run_id}")
        self.next(self.calibrate)

    @step
    def calibrate(self):
        """Fit isotonic calibration; append calibration metrics to the run."""
        from src.calibrate import calibrate_model

        configure_mlflow()
        _, metrics = calibrate_model(self.model_path, self.splits)
        self.calib_metrics = calib_scalars(metrics)  # scalars only
        log_metrics_to_run(self.prod_run_id, self.calib_metrics)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Threshold selection + profit; append profit metrics to the run."""
        from src.evaluate import run_evaluation

        configure_mlflow()
        result = run_evaluation(self.model_path, self.splits)
        self.eval_metrics = eval_scalars(result)  # drops val_profit_curve
        log_metrics_to_run(self.prod_run_id, self.eval_metrics)
        self.next(self.fairness)

    @step
    def fairness(self):
        """
        Three-layer fairness audit; append fairness metrics to the run.

        Slowest step: Layer 3 independently retrains a with-state and a no-state
        LightGBM for its ablation. This is expected, not a hang.
        """
        from src.fairness import run_fairness_audit

        configure_mlflow()
        # Layer 1 audits at the model's REAL operating point -- the
        # profit-maximizing threshold the evaluate step just chose on Val --
        # not at fairness.py's notebook-era fallback default. (Layer 3 keeps
        # its own deliberate 0.22; see ABLATION_THRESHOLD in src/fairness.py.)
        result = run_fairness_audit(
            self.model_path,
            self.splits,
            audit_threshold=self.eval_metrics["best_threshold"],
        )
        self.fairness_metrics = fairness_scalars(result)  # drops fair_df + layers
        # Record WHICH operating point Layer 1 was audited at, so the MLflow
        # archive answers "at what threshold", not just "what ratio".
        self.fairness_metrics["fairness_audit_threshold"] = float(
            self.eval_metrics["best_threshold"]
        )
        log_metrics_to_run(self.prod_run_id, self.fairness_metrics)
        self.next(self.explain)

    @step
    def explain(self):
        """
        Log global SHAP importance (mean|SHAP|, log-odds) onto the unified
        lgbm_production run, beside AUC / Brier / profit / fairness -- the "not
        yet wired" observability signal from docs/design.md Section 4, now wired.

        Observability, NOT a correctness gate. It reads self.model_path +
        self.splits and writes NOTHING to self. Its failure is fail-OPEN
        (guarded_importance): a broken explainer must not throw away a run whose
        model is already trained, calibrated, evaluated, and fairness-audited.
        The inner additivity guard inside global_importance stays fail-closed --
        a wrong ranking raises, and that ValueError propagates (it is not in
        _EXPLAIN_SKIP_ERRORS), so a wrong ranking is neither logged nor hidden.
        """
        configure_mlflow()

        # `params` is filled INSIDE compute (below) and read back only after a
        # clean success, so the reproducibility context is logged with -- never
        # without -- the values it describes.
        params: dict[str, object] = {}

        def compute() -> pd.DataFrame:
            # EVERY import of src.explain (which imports shap at module top) lives
            # inside this guarded region, so a missing/moved shap becomes a logged
            # SKIP -- ImportError is in _EXPLAIN_SKIP_ERRORS -- not a step crash.
            import numpy as np

            from src.explain import (
                CONTRIBUTION_SCALE,
                RANDOM_SEED,
                SHAP_SAMPLE_N,
                global_importance,
            )

            # Sample Test exactly as run_explanation does (src/explain.py:747-750)
            # -- same seed and size -- so the logged ranking IS the documented one
            # and shap_seed is a truthful param. ~0.4 s marginal: global_importance
            # re-loads the artifact and builds its own explainer, and the
            # additivity guard fires inside its _shap_matrix call.
            test = self.splits["test"]
            n = min(SHAP_SAMPLE_N, len(test))
            idx = np.random.default_rng(RANDOM_SEED).choice(len(test), size=n, replace=False)
            params.update(
                shap_sample_n=n, shap_seed=RANDOM_SEED, shap_scale=CONTRIBUTION_SCALE,
            )
            return global_importance(test.iloc[idx], model_path=self.model_path)

        metrics, skip_reason = guarded_importance(compute)
        if skip_reason is not None:
            print(f"explain SKIPPED (fail-open) -- {skip_reason}")
            self.next(self.end)
            return

        log_params_to_run(self.prod_run_id, params)
        log_metrics_to_run(self.prod_run_id, metrics)

        print(
            f"Global SHAP importance logged onto {self.prod_run_id} "
            f"({params['shap_scale']}, n={params['shap_sample_n']:,}, "
            f"seed={params['shap_seed']}):"
        )
        for key, value in metrics.items():
            print(f"  {key:34} {value:.6f}")

        self.next(self.end)

    @step
    def end(self):
        """Print a consolidated summary of the whole pipeline."""
        c, e, f = self.calib_metrics, self.eval_metrics, self.fairness_metrics
        line = "=" * 68
        print("\n" + line)
        print("TRAINING PIPELINE COMPLETE -- consolidated summary")
        print(line)
        print(f"Model artifact:     {self.model_path}")
        print(f"Unified MLflow run: {self.prod_run_id}  (experiment: {EXPERIMENT_NAME})")

        print("\n-- Calibration (test set) --")
        print(f"  Brier  raw -> cal:  {c['brier_raw']:.4f} -> {c['brier_cal']:.4f}")
        print(f"  AUC    raw -> cal:  {c['auc_raw']:.4f} -> {c['auc_cal']:.4f}")
        print(f"  mean_pred -> actual:{c['mean_pred_cal']:.4f} -> {c['actual_test_default_rate']:.4f}")

        print("\n-- Evaluation (profit at chosen threshold) --")
        print(f"  Best threshold:     {e['best_threshold']:.2f}")
        print(f"  Test profit:        ${e['test_profit']:,.0f}")
        print(f"  Improvement/naive:  ${e['improvement_over_naive']:,.0f}")
        print(f"  Approval rate:      {e['approval_rate']:.1%}")
        print(f"  Bad rate approved:  {e['bad_rate_approved']:.1%}")

        print("\n-- Fairness audit (Layer 3 ablation) --")
        print(f"  AUC with / no state:{f['auc_with_state']:.4f} / {f['auc_no_state']:.4f}")
        print(f"  AUC cost of dropping state: {f['auc_cost']:+.4f}")
        print(f"  Confirmed proxy-risk states: {int(f['n_confirmed_proxy_states'])}")

        print("\nBrowse the full unified run:")
        print(f"  mlflow ui --backend-store-uri sqlite:///mlflow.db   (experiment: {EXPERIMENT_NAME})")
        print(line)


if __name__ == "__main__":
    TrainingFlow()
