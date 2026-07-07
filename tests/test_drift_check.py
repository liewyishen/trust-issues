"""
Tests for pipelines/drift_check.py -- the input-distribution drift monitor.

Everything here runs on synthetic frames against the pure computation core
(drift_metrics and below): no 167 MB CSV, no model artifacts, and never a
write to mlflow.db. The end-to-end path (run_drift_check on real data +
MLflow logging) is the `uv run python pipelines/drift_check.py` smoke run,
same division of labor as test_training_flow.py.

The two headline tests mirror the agreed design:
  - should-pass: two random halves of one distribution -> PSI/KS ~ 0,
    zero alarms.
  - should-fire: a current window with an injected (100, 1000] tail -> the
    tripwire alarm fires. The same test also documents WHY the tripwire
    exists: quantile-binned PSI stays quiet on that ~1% tail.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from pipelines.drift_check import (
    DEFAULT_ALARM_THRESHOLDS,
    DTI_TRIPWIRE_LO,
    calibration_gap,
    drift_metrics,
    evaluate_alarms,
    ks_statistic,
    psi,
    quantile_bin_edges,
    sentinel_rate,
    tripwire_share,
)
from src.calibrate import apply_calibration
from src.data_validation import DTI_MAX_REAL, DTI_SENTINEL


def _dti_like(n: int, seed: int, sentinel_share: float = 0.02) -> np.ndarray:
    """
    Synthetic dti_n-shaped values: lognormal clipped to [0, 60] (the real
    training-years band tops out far below 100) plus a small share of 999
    sentinels, mirroring the real column's structure.
    """
    rng = np.random.default_rng(seed)
    values = np.clip(rng.lognormal(mean=2.8, sigma=0.5, size=n), 0.0, 60.0)
    values[rng.random(n) < sentinel_share] = DTI_SENTINEL
    return values


class TestQuantileBinEdges:
    def test_edges_pinned_to_contract_bounds(self):
        # The outer edges must sit on the data contract's [0, 1000], not the
        # reference min/max -- otherwise the 2016+ tail falls out of every bin.
        edges = quantile_bin_edges(pd.Series(_dti_like(5_000, seed=6)))
        assert edges[0] == 0.0
        assert edges[-1] == DTI_MAX_REAL

    def test_mass_points_deduplicate_but_edges_stay_increasing(self):
        ref = pd.Series([5.0] * 500 + list(np.linspace(10.0, 50.0, 500)))
        edges = quantile_bin_edges(ref, n_bins=10)
        assert np.all(np.diff(edges) > 0)
        assert edges[0] == 0.0 and edges[-1] == DTI_MAX_REAL

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="empty reference"):
            quantile_bin_edges(pd.Series([], dtype=float))


class TestPsi:
    def test_identical_samples_score_zero(self):
        ref = pd.Series(np.random.default_rng(3).uniform(0.0, 60.0, 10_000))
        edges = quantile_bin_edges(ref)
        assert psi(ref, ref, edges) == pytest.approx(0.0, abs=1e-9)

    def test_emptied_bins_stay_finite_and_scream(self):
        # Epsilon clipping: bins that fully empty must produce a large but
        # FINITE PSI, never inf/nan from ln(0).
        ref = pd.Series(np.linspace(0.0, 50.0, 1_000))
        edges = quantile_bin_edges(ref, n_bins=10)
        current = pd.Series(np.linspace(40.0, 50.0, 500))  # lower bins empty
        value = psi(ref, current, edges)
        assert np.isfinite(value)
        assert value > DEFAULT_ALARM_THRESHOLDS["psi"]


class TestKsStatistic:
    def test_identical_zero_and_shift_large(self):
        ref = pd.Series(np.random.default_rng(5).uniform(0.0, 30.0, 5_000))
        assert ks_statistic(ref, ref) == 0.0
        assert ks_statistic(ref, ref + 20.0) > 0.3


class TestSentinelAndTripwire:
    def test_sentinel_counted_by_rate_but_excluded_from_tripwire(self):
        # 999.0 sits inside (100, 1000] numerically, but it is a missing-value
        # code: sentinel_rate owns it, the tripwire must not double-count it.
        # 999.5 is NOT the sentinel and belongs to the tripwire band.
        s = pd.Series([10.0, 20.0, DTI_SENTINEL, 150.0, 999.5])
        assert sentinel_rate(s) == pytest.approx(0.2)
        assert tripwire_share(s) == pytest.approx(0.4)  # 150.0 and 999.5

    def test_empty_window_yields_nan(self):
        empty = pd.Series([], dtype=float)
        assert np.isnan(sentinel_rate(empty))
        assert np.isnan(tripwire_share(empty))


class TestDriftMetricsShouldPass:
    def test_random_halves_no_alarms(self):
        # Two random halves of ONE distribution assigned to reference vs.
        # current years: every signal must stay quiet.
        rng = np.random.default_rng(1)
        values = _dti_like(40_000, seed=1)
        years = np.array([2010] * 20_000 + [2016] * 20_000)
        rng.shuffle(years)
        df = pd.DataFrame({"dti_n": values, "issue_year": years})

        metrics = drift_metrics(
            df, reference_years=(2007, 2014), current_years=(2016,)
        )

        for key in (
            "n_reference", "sentinel_rate_reference", "tripwire_share_reference",
            "n_2016", "psi_dti_n_2016", "ks_dti_n_2016",
            "sentinel_rate_2016", "tripwire_share_2016",
        ):
            assert key in metrics
        assert metrics["psi_dti_n_2016"] < 0.05
        assert metrics["ks_dti_n_2016"] < 0.05
        assert metrics["tripwire_share_2016"] == 0.0
        assert evaluate_alarms(metrics) == []


class TestDriftMetricsShouldFire:
    def test_injected_regime_tail_fires_tripwire_while_psi_stays_blind(self):
        # The regime-shift probe: inject 1% of (100, 1000] values into the
        # current year. The tripwire must fire -- and PSI must NOT, which is
        # the documented reason (finding B) the tripwire exists at all.
        rng = np.random.default_rng(7)
        ref = _dti_like(20_000, seed=2, sentinel_share=0.0)
        cur = _dti_like(20_000, seed=3, sentinel_share=0.0)
        injected = rng.choice(20_000, size=200, replace=False)
        cur[injected] = rng.uniform(DTI_TRIPWIRE_LO + 1.0, 990.0, size=200)
        df = pd.DataFrame({
            "dti_n": np.concatenate([ref, cur]),
            "issue_year": np.array([2012] * 20_000 + [2017] * 20_000),
        })

        metrics = drift_metrics(
            df, reference_years=(2007, 2014), current_years=(2017,)
        )
        alarms = evaluate_alarms(metrics)

        assert metrics["tripwire_share_2017"] == pytest.approx(0.01)
        assert metrics["psi_dti_n_2017"] < DEFAULT_ALARM_THRESHOLDS["psi"]
        assert len(alarms) == 1
        assert alarms[0].startswith("tripwire_share_2017")

    def test_sentinel_surge_fires_its_own_alarm_not_the_tripwire(self):
        # 10% of the current year turns into 999: the missingness mechanism
        # changed. Exactly one alarm -- the sentinel delta; PSI/KS see only
        # non-sentinel values (a random subsample, same shape) and the
        # tripwire excludes 999 by construction.
        rng = np.random.default_rng(11)
        base = _dti_like(20_000, seed=4, sentinel_share=0.0)
        cur = base[10_000:].copy()
        cur[rng.choice(10_000, size=1_000, replace=False)] = DTI_SENTINEL
        df = pd.DataFrame({
            "dti_n": np.concatenate([base[:10_000], cur]),
            "issue_year": np.array([2007] * 10_000 + [2016] * 10_000),
        })

        metrics = drift_metrics(
            df, reference_years=(2007, 2014), current_years=(2016,)
        )
        alarms = evaluate_alarms(metrics)

        assert metrics["sentinel_rate_2016"] == pytest.approx(0.10)
        assert metrics["tripwire_share_2016"] == 0.0
        assert len(alarms) == 1
        assert "sentinel_rate_2016" in alarms[0]


class TestDriftMetricsContract:
    def test_dti_n_must_be_monitored(self):
        df = pd.DataFrame({"revenue": [1.0], "issue_year": [2010]})
        with pytest.raises(ValueError, match="dti_n"):
            drift_metrics(df, columns=("revenue",))

    def test_empty_reference_window_raises(self):
        df = pd.DataFrame({"dti_n": [10.0], "issue_year": [2016]})
        with pytest.raises(ValueError, match="[Rr]eference"):
            drift_metrics(df, reference_years=(2007, 2014), current_years=(2016,))

    def test_empty_current_year_yields_nan_not_crash(self):
        df = pd.DataFrame({
            "dti_n": _dti_like(1_000, seed=9),
            "issue_year": np.full(1_000, 2010),
        })
        metrics = drift_metrics(
            df, reference_years=(2007, 2014), current_years=(2018,)
        )
        assert metrics["n_2018"] == 0.0
        assert np.isnan(metrics["psi_dti_n_2018"])
        assert np.isnan(metrics["sentinel_rate_2018"])
        assert evaluate_alarms(metrics) == []  # NaN never alarms


class TestEvaluateAlarms:
    def test_each_threshold_fires_independently(self):
        metrics = {
            "sentinel_rate_reference": 0.02,
            "psi_dti_n_2016": 0.26,        # > 0.25 -> fires
            "ks_dti_n_2016": 0.09,         # < 0.10 -> quiet
            "sentinel_rate_2016": 0.03,    # delta 0.01 < 0.05 -> quiet
            "tripwire_share_2016": 0.001,  # > 1e-4 -> fires
            "calib_gap_2016": -0.041,      # |gap| > 0.02 -> fires
        }
        alarms = evaluate_alarms(metrics)
        fired = "\n".join(alarms)
        assert len(alarms) == 3
        assert "psi_dti_n_2016" in fired
        assert "tripwire_share_2016" in fired
        assert "calib_gap_2016" in fired

    def test_partial_threshold_override_merges_with_defaults(self):
        metrics = {"psi_dti_n_2016": 0.15}
        assert evaluate_alarms(metrics) == []
        assert len(evaluate_alarms(metrics, {"psi": 0.1})) == 1

    def test_non_finite_values_never_alarm(self):
        metrics = {"psi_dti_n_2018": float("nan"), "ks_dti_n_2018": float("inf")}
        assert evaluate_alarms(metrics) == []


class TestCalibrationGap:
    def test_sign_convention_negative_means_underprediction(self):
        # Identity calibrator on [0, 1]: calibrated == raw. mean_pred 0.2 vs.
        # all-defaults actual 1.0 -> gap -0.8: negative = under-predicted risk,
        # matching the real 2016-17 direction (0.1915 - 0.2323 = -0.0408).
        iso = IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.0, 1.0])
        calibrated = apply_calibration(iso, np.array([0.1, 0.2, 0.3]))
        gap = calibration_gap(calibrated, pd.Series([1, 1, 1]))
        assert gap == pytest.approx(-0.8)

    def test_empty_actuals_yield_nan(self):
        assert np.isnan(calibration_gap(np.array([]), pd.Series([], dtype=float)))
