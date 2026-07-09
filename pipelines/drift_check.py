"""
Input-distribution drift monitor for the LendingClub credit-default model.

This module discharges the drift TODO that src/data_validation.py and
docs/data-decisions.md used to carry ("TODO(drift)"). The 495-row DTI
investigation established that
2016+ loans carry a dti_n reporting regime (real values in (100, 1000]) the
training years never saw: 0 such rows in Train (2007-2014), 7 in 2015, 488
in 2016-2018. The calibrated-probability gap on Test (mean_pred 0.1915 vs.
actual default rate 0.2323) is a direct trace of that shift and was
deliberately left in place at calibration time -- it is a signal to monitor,
not a bug to eliminate. This monitor is where it gets monitored.

Five signals, each traceable to a specific finding:
  1. psi_dti_n_<year>      Per-issue_year PSI against the training-years
                           distribution. Quantile bins are fixed ONCE from
                           the reference so all years are comparable;
                           sentinel rows are excluded.
  2. ks_dti_n_<year>       Per-issue_year two-sample KS statistic, same
                           window and exclusion. The statistic only, never
                           the p-value: at n ~ 4.5e5 every visible shift is
                           "significant", so only the effect size informs.
  3. sentinel_rate_<year>  Share of dti_n == 999 (the missing-value
                           sentinel). Kept OUT of PSI/KS: fed raw into them
                           it fakes a mass spike at 999, and a change in
                           missingness deserves to read as its own signal.
  4. tripwire_share_<year> Share of dti_n in (100, 1000] -- the direct probe
                           for the 2016+ regime population. Quantile-binned
                           PSI is nearly blind to this ~0.1% tail, so the
                           tripwire is reported independently; the training
                           years hold exactly zero such rows.
  5. calib_gap_<year>      Mean calibrated PD minus actual default rate,
                           scored with the SHIPPED model + calibrator
                           (never refit). Negative = under-predicted risk.

Design decisions (agreed before implementation):
  - Hand-rolled PSI/KS, no Evidently: four scalar distribution signals do not justify
    pulling litestar/uvicorn/watchdog/nltk + usage telemetry into a batch
    job, and hand-rolled matches this repo's hand-rolled fairness audit.
  - Plain module + callable entry, no Metaflow FlowSpec: the check is one
    linear step; a FlowSpec would pickle the full DataFrame across a
    subprocess boundary for zero orchestration benefit.
  - Own MLflow run (run name "drift_check", same lc_default_risk
    experiment): monitoring and training have different lifecycles, so
    drift metrics never append onto an lgbm_production run.
  - Report, don't raise (by default): unlike the leakage sentinels, where
    an alarm means the run's output is invalid, a drift alarm IS this
    tool's normal product -- on the real data 2016+ is EXPECTED to fire.
    Automation can opt into fail_on_alarm=True.

Run it:
  uv run python pipelines/drift_check.py
Browse the run:
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  (experiment: lc_default_risk, run name: drift_check)
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

# pipelines/ lives beside src/, not inside it -- same shim as training_flow.py,
# so `python pipelines/drift_check.py` can import from src/ and pipelines/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from pipelines.training_flow import configure_mlflow, log_metrics_to_run  # noqa: E402
from src.calibrate import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    apply_calibration,
    load_calibrator,
)
from src.data_loader import (  # noqa: E402
    HOLDOUT_YEAR_2018,
    TEST_YEAR_HI,
    TEST_YEAR_LO,
    TRAIN_YEAR_HI,
    TRAIN_YEAR_LO,
    VAL_CALIB_YEAR,
    load_raw,
    slice_years,
)
from src.data_validation import DTI_MAX_REAL, DTI_SENTINEL  # noqa: E402

# Reusing train.py's private encoding helpers over rewriting the encoding is
# the same anti-skew decision calibrate.py documents at its own import site:
# a second, independently written encoding path here would be exactly the
# train/serve skew this repo keeps warning about.
from src.train import _to_lgb_frame, _xy, load_model_artifact  # noqa: E402

MLFLOW_RUN_NAME = "drift_check"

# The pre-widening DTI ceiling. DTI_MAX_REAL was 100 before the 495-row
# investigation widened it to 1000 (see docs/data-decisions.md), so the old
# contract value no longer exists as an importable constant. Values in
# (100, 1000] are exactly the 2016+ reporting-regime population, which makes
# the historical ceiling the natural tripwire boundary -- a documented
# archaeological constant, not a new magic number.
DTI_TRIPWIRE_LO = 100.0

# Per-column missing-value sentinels excluded from the distribution signals
# (PSI/KS and the tripwire numerator). dti_n's 999 is tracked separately as
# sentinel_rate_<year>; add entries here if a future monitored column carries
# its own sentinel code.
COLUMN_SENTINELS: dict[str, float] = {"dti_n": DTI_SENTINEL}

DEFAULT_MONITORED_COLUMNS: tuple[str, ...] = ("dti_n",)

# Current windows: 2015 is the FULL year (drift is a per-year population
# question), not temporal_split()'s shuffled 40k val/calib subsamples.
DEFAULT_CURRENT_YEARS: tuple[int, ...] = (
    VAL_CALIB_YEAR,
    TEST_YEAR_LO,
    TEST_YEAR_HI,
    HOLDOUT_YEAR_2018,
)

# Alarm thresholds. All overridable via run_drift_check(thresholds=...) --
# and none of them is a theorem; each is a documented judgment call:
#   psi > 0.25             The classic population-stability convention from
#                          credit scoring (0.1 = watch, 0.25 = act).
#   ks > 0.10              Ten points of maximum CDF displacement is material
#                          for a feature that enters decisions. The p-value
#                          is deliberately not used (see module docstring).
#   sentinel_rate_delta    Alarm on |rate_year - rate_reference| > 0.05: the
#                          reference has its own sentinel share, so the
#                          signal is "the missingness mechanism changed",
#                          not "missingness exists".
#   tripwire_share > 1e-4  Data-anchored: 2015 sits at ~1.4e-5 (7 rows --
#                          stays quiet) and 2016+ at ~1e-3 (fires). The
#                          threshold sits inside that two-order-of-magnitude
#                          gap, close to the quiet side.
#   calib_gap_abs > 0.02   Half the known 2016-2017 gap (-0.041); at base
#                          rates of 0.17-0.23, two points of probability is
#                          ~10% relative mispricing -- an early-warning
#                          level, not a post-mortem one.
DEFAULT_ALARM_THRESHOLDS: dict[str, float] = {
    "psi": 0.25,
    "ks": 0.10,
    "sentinel_rate_delta": 0.05,
    "tripwire_share": 1e-4,
    "calib_gap_abs": 0.02,
}


# --- Distribution signals (pure functions, no I/O) -------------------------------

def _distribution_values(s: pd.Series, column: str) -> pd.Series:
    """Values entering PSI/KS: NaNs dropped, the column's sentinel removed."""
    s = s.dropna()
    sentinel = COLUMN_SENTINELS.get(column)
    if sentinel is not None:
        s = s[s != sentinel]
    return s


def quantile_bin_edges(
    reference: pd.Series,
    n_bins: int = 10,
    lo: float = 0.0,
    hi: float = DTI_MAX_REAL,
) -> np.ndarray:
    """
    PSI bin edges from the reference distribution's quantiles.

    Computed ONCE from the reference (training years) and reused for every
    current window -- cross-year comparability comes from the edges never
    moving. The outer edges are pinned to the data contract's bounds
    (defaults are dti_n's [0, DTI_MAX_REAL]) rather than the reference
    min/max: the training years top out far below 100, and an upper edge at
    their max would silently drop the 2016+ (100, 1000] tail out of every
    bin -- exactly the mass this monitor exists to see. Ten bins is the
    standard PSI decile granularity (more bins = noisier per-bin
    proportions, fewer = less sensitivity). Duplicate quantiles (mass
    points) are collapsed by np.unique; fewer effective bins is fine as
    long as every window shares the same edges.
    """
    if len(reference) == 0:
        raise ValueError("Cannot derive bin edges from an empty reference.")
    edges = reference.quantile(np.linspace(0.0, 1.0, n_bins + 1)).to_numpy()
    edges[0] = lo
    edges[-1] = hi
    return np.unique(edges)


def psi(
    reference: pd.Series,
    current: pd.Series,
    bin_edges: np.ndarray,
    epsilon: float = 1e-4,
) -> float:
    """
    Population Stability Index over shared bin edges.

    PSI = sum over bins of (p_cur - p_ref) * ln(p_cur / p_ref). Empty bins
    are handled by clipping each bin proportion at `epsilon`, with NO
    renormalisation: clipping touches only degenerate bins and leaves every
    non-empty bin's contribution exact, and at 1e-4 across ~10 bins the
    total mass distortion is <= 0.1%. The behavior at the limit is the
    honest one -- a bin holding 10% of reference that fully empties out
    contributes ~0.1 * ln(1000) ~ 0.69: PSI screams there, finitely,
    instead of overflowing to inf.
    """
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    p_ref = np.maximum(ref_counts / max(ref_counts.sum(), 1), epsilon)
    p_cur = np.maximum(cur_counts / max(cur_counts.sum(), 1), epsilon)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """
    Two-sample Kolmogorov-Smirnov statistic (scipy.stats.ks_2samp).

    Reports the statistic only -- the maximum CDF displacement, an effect
    size a threshold can be set on. The p-value is deliberately discarded:
    with ~4.5e5 reference rows, any visible shift is "significant", so the
    p-value carries no decision-relevant information here.
    """
    return float(ks_2samp(reference, current).statistic)


def sentinel_rate(s: pd.Series) -> float:
    """
    Share of rows carrying dti_n's 999 missing-value sentinel.

    Denominator: all rows of the window (matching tripwire_share, so the
    two shares read on the same base).
    """
    if len(s) == 0:
        return float("nan")
    return float((s == DTI_SENTINEL).mean())


def tripwire_share(s: pd.Series) -> float:
    """
    Share of rows with real dti_n in (DTI_TRIPWIRE_LO, DTI_MAX_REAL].

    The direct probe for the 2016+ DTI reporting-regime population
    (docs/data-decisions.md): the training years contain exactly zero such
    rows, so ANY mass here is a regime signal -- and the investigation
    showed this tail is ~0.1% of Test, far too small for quantile-binned
    PSI to register. Sentinel rows are excluded from the numerator (999
    falls inside (100, 1000] numerically but is a missing-value code owned
    by sentinel_rate); the denominator is all rows of the window.
    """
    if len(s) == 0:
        return float("nan")
    real = s[s != DTI_SENTINEL]
    in_band = (real > DTI_TRIPWIRE_LO) & (real <= DTI_MAX_REAL)
    return float(in_band.sum() / len(s))


def drift_metrics(
    df: pd.DataFrame,
    columns: Sequence[str] = DEFAULT_MONITORED_COLUMNS,
    reference_years: tuple[int, int] = (TRAIN_YEAR_LO, TRAIN_YEAR_HI),
    current_years: Sequence[int] = DEFAULT_CURRENT_YEARS,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute the distribution-drift signals (1-4) as one flat metrics dict.

    Pure computation: no file I/O, no MLflow, no model -- tests exercise it
    on synthetic frames. Signal 5 (the calibration gap) needs the shipped
    model artifacts and is added by run_drift_check().

    Keys: n_reference, sentinel_rate_reference, tripwire_share_reference,
    then per current year: n_<year>, psi_<col>_<year>, ks_<col>_<year>,
    sentinel_rate_<year>, tripwire_share_<year>. All values are floats
    (MLflow-ready); a year with no rows yields NaN signals, which both
    evaluate_alarms() and log_metrics_to_run() skip.
    """
    if "dti_n" not in columns:
        raise ValueError(
            "dti_n must be among the monitored columns -- it is the column "
            "this monitor exists for (see docs/data-decisions.md)."
        )

    reference = slice_years(df, *reference_years)
    if reference.empty:
        raise ValueError(
            f"Reference window {reference_years} selects no rows -- drift "
            "against an empty training-years distribution is meaningless."
        )

    ref_values = {c: _distribution_values(reference[c], c) for c in columns}
    edges = {c: quantile_bin_edges(ref_values[c], n_bins=n_bins) for c in columns}

    metrics: dict[str, float] = {
        "n_reference": float(len(reference)),
        "sentinel_rate_reference": sentinel_rate(reference["dti_n"]),
        "tripwire_share_reference": tripwire_share(reference["dti_n"]),
    }

    for year in current_years:
        cur = slice_years(df, year, year)
        metrics[f"n_{year}"] = float(len(cur))
        if cur.empty:
            for col in columns:
                metrics[f"psi_{col}_{year}"] = float("nan")
                metrics[f"ks_{col}_{year}"] = float("nan")
            metrics[f"sentinel_rate_{year}"] = float("nan")
            metrics[f"tripwire_share_{year}"] = float("nan")
            continue
        for col in columns:
            cur_values = _distribution_values(cur[col], col)
            if len(cur_values) == 0:
                metrics[f"psi_{col}_{year}"] = float("nan")
                metrics[f"ks_{col}_{year}"] = float("nan")
                continue
            metrics[f"psi_{col}_{year}"] = psi(ref_values[col], cur_values, edges[col])
            metrics[f"ks_{col}_{year}"] = ks_statistic(ref_values[col], cur_values)
        metrics[f"sentinel_rate_{year}"] = sentinel_rate(cur["dti_n"])
        metrics[f"tripwire_share_{year}"] = tripwire_share(cur["dti_n"])

    return metrics


# --- Calibration-gap signal ------------------------------------------------------

def calibration_gap(calibrated_probs: np.ndarray, actuals: pd.Series) -> float:
    """
    Mean calibrated predicted PD minus actual default rate.

    Sign convention: NEGATIVE means the model under-predicts risk. That is
    the known 2016-2017 state (mean_pred 0.1915 vs. actual 0.2323 ->
    -0.0408), and it is exactly the gap that was deliberately NOT "fixed"
    at calibration time: the calibrator is honest on 2015-shaped data, and
    the widening gap on later years is the drift trace this monitor exists
    to report.
    """
    if len(actuals) == 0:
        return float("nan")
    return float(np.mean(calibrated_probs) - actuals.mean())


def yearly_calibration_gap(df_year: pd.DataFrame, model_artifact: dict, calibrator) -> float:
    """
    Calibration gap for one issue_year slice, scored with shipped artifacts.

    The scoring path replicates calibrate_model()'s exactly
    (src/calibrate.py): _xy -> _to_lgb_frame with the TRAIN-derived
    category maps -> Booster.predict at best_iteration ->
    apply_calibration. Nothing is refit here. Note the calibrator was fit
    on the 2015 calib slice, so the 2015 gap is close to in-sample -- the
    natural "not yet drifted" baseline the later years are read against.
    """
    X, y = _xy(df_year)
    X_lgb = _to_lgb_frame(X, model_artifact["category_maps"])
    raw = model_artifact["model"].predict(
        X_lgb, num_iteration=model_artifact["best_iteration"]
    )
    return calibration_gap(apply_calibration(calibrator, raw), y)


# --- Alarms ----------------------------------------------------------------------

def evaluate_alarms(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """
    Compare the metrics dict against the alarm thresholds.

    Returns human-readable alarm strings (empty list = no drift). Partial
    threshold overrides merge over DEFAULT_ALARM_THRESHOLDS. Reference-
    window keys are context, never alarmed on; non-finite values (empty
    windows) are skipped, mirroring log_metrics_to_run()'s semantics.
    """
    t = {**DEFAULT_ALARM_THRESHOLDS, **(thresholds or {})}
    ref_sentinel = metrics.get("sentinel_rate_reference")
    alarms: list[str] = []

    for key, value in metrics.items():
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            continue
        if key.endswith("_reference"):
            continue
        if key.startswith("psi_") and value > t["psi"]:
            alarms.append(f"{key}={value:.4f} > PSI threshold {t['psi']}")
        elif key.startswith("ks_") and value > t["ks"]:
            alarms.append(f"{key}={value:.4f} > KS threshold {t['ks']}")
        elif (
            key.startswith("sentinel_rate_")
            and ref_sentinel is not None
            and math.isfinite(ref_sentinel)
            and abs(value - ref_sentinel) > t["sentinel_rate_delta"]
        ):
            alarms.append(
                f"{key}={value:.4f} deviates from reference sentinel rate "
                f"{ref_sentinel:.4f} by more than {t['sentinel_rate_delta']} "
                "-- the missingness mechanism changed"
            )
        elif key.startswith("tripwire_share_") and value > t["tripwire_share"]:
            alarms.append(
                f"{key}={value:.6f} > tripwire threshold {t['tripwire_share']:g} "
                f"(training years hold exactly zero dti_n in "
                f"({DTI_TRIPWIRE_LO:g}, {DTI_MAX_REAL:g}] -- this is the 2016+ "
                "reporting-regime population)"
            )
        elif key.startswith("calib_gap_") and abs(value) > t["calib_gap_abs"]:
            alarms.append(
                f"{key}={value:+.4f}, |gap| > {t['calib_gap_abs']} "
                "(negative = model under-predicts risk)"
            )

    return alarms


# --- Orchestration (the only part with side effects) -----------------------------

def _print_report(
    metrics: dict[str, float],
    alarms: list[str],
    current_years: Sequence[int],
    columns: Sequence[str],
) -> None:
    """Console report in the repo's aligned-print style."""
    line = "=" * 68
    print("\n" + line)
    print("DRIFT CHECK -- input distribution vs. training years")
    print(line)
    print(
        f"Reference        n={int(metrics['n_reference']):>9,}  "
        f"sentinel_rate={metrics['sentinel_rate_reference']:.4f}  "
        f"tripwire_share={metrics['tripwire_share_reference']:.6f}"
    )
    for year in current_years:
        parts = [f"{year}             n={int(metrics[f'n_{year}']):>9,}"]
        for col in columns:
            parts.append(f"psi[{col}]={metrics[f'psi_{col}_{year}']:.4f}")
            parts.append(f"ks[{col}]={metrics[f'ks_{col}_{year}']:.4f}")
        parts.append(f"sentinel={metrics[f'sentinel_rate_{year}']:.4f}")
        parts.append(f"tripwire={metrics[f'tripwire_share_{year}']:.6f}")
        gap_key = f"calib_gap_{year}"
        if gap_key in metrics:
            parts.append(f"calib_gap={metrics[gap_key]:+.4f}")
        print("  ".join(parts))
    if alarms:
        print(f"\n{len(alarms)} drift ALERT(s):")
        for alarm in alarms:
            print(f"  ALERT: {alarm}")
    else:
        print("\nNo drift alarms.")
    print(line)


def run_drift_check(
    df: pd.DataFrame | None = None,
    columns: Sequence[str] = DEFAULT_MONITORED_COLUMNS,
    reference_years: tuple[int, int] = (TRAIN_YEAR_LO, TRAIN_YEAR_HI),
    current_years: Sequence[int] = DEFAULT_CURRENT_YEARS,
    n_bins: int = 10,
    model_path: str | Path | None = None,
    calibrator_path: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
    log_to_mlflow: bool = True,
    fail_on_alarm: bool = False,
) -> dict:
    """
    Full drift check: distribution signals + calibration gap, report, log.

    Parameters
    ----------
    df : pd.DataFrame or None
        The modeling frame. None (default) loads via load_raw(), which runs
        the fail-closed validation gate -- data reaching the drift math has
        already honored the schema contract.
    columns : sequence of str
        Columns to monitor with PSI/KS; must include "dti_n".
    reference_years, current_years
        Windows built with data_loader's slice_years + year constants.
        2015 is compared as a FULL year, not the val/calib subsamples.
    model_path, calibrator_path
        Shipped artifacts for the calibration-gap signal. Defaults:
        models/lgbm_model.pkl and its sibling isotonic_calibrator.pkl.
    thresholds : dict or None
        Partial overrides merged over DEFAULT_ALARM_THRESHOLDS.
    log_to_mlflow : bool
        If True, log all metrics to a NEW MLflow run named "drift_check" in
        the lc_default_risk experiment (never appended to lgbm_production:
        monitoring and training have different lifecycles). Tests pass
        False so they never touch mlflow.db.
    fail_on_alarm : bool
        Default False: drift alarms are this tool's normal product on the
        real data (the 2016+ shift is a documented fact), so it reports
        instead of raising. CI/automation can set True to fail closed.

    Returns
    -------
    dict with "metrics" (flat float dict, as logged) and "alarms"
    (list of human-readable alarm strings).
    """
    if df is None:
        df = load_raw()

    metrics = drift_metrics(
        df,
        columns=columns,
        reference_years=reference_years,
        current_years=current_years,
        n_bins=n_bins,
    )

    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    calibrator_path = (
        Path(calibrator_path)
        if calibrator_path is not None
        else model_path.parent / "isotonic_calibrator.pkl"
    )
    for path in (model_path, calibrator_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- the drift check scores each year with "
                "the shipped model + calibrator and never refits. Run the "
                "training pipeline first: "
                "uv run python pipelines/training_flow.py run"
            )
    model_artifact = load_model_artifact(model_path)  # fail-closed on feature-contract mismatch
    calibrator = load_calibrator(calibrator_path, model_artifact=model_artifact)
    for year in current_years:
        if metrics[f"n_{year}"] == 0:
            metrics[f"calib_gap_{year}"] = float("nan")
            continue
        metrics[f"calib_gap_{year}"] = yearly_calibration_gap(
            slice_years(df, year, year), model_artifact, calibrator
        )

    alarms = evaluate_alarms(metrics, thresholds)
    metrics["n_drift_alarms"] = float(len(alarms))

    _print_report(metrics, alarms, current_years, columns)

    if log_to_mlflow:
        configure_mlflow()
        # Open (and immediately close) the run to obtain its id, then append
        # through log_metrics_to_run -- reusing its skip-non-finite semantics
        # instead of rewriting them here.
        with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as active:
            run_id = active.info.run_id
        log_metrics_to_run(run_id, metrics)
        print(f"Drift metrics logged to MLflow run {run_id} "
              f"(run name: {MLFLOW_RUN_NAME})")

    if fail_on_alarm and alarms:
        raise ValueError(
            f"{len(alarms)} drift alarm(s) fired:\n" + "\n".join(alarms)
        )

    return {"metrics": metrics, "alarms": alarms}


if __name__ == "__main__":
    run_drift_check()
