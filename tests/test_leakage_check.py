"""
Tests for src/leakage_check.py

The leakage module's whole job is to *prove* the data is clean rather than
assume it. These tests hold that module to the same standard: every guard is
checked on both sides — it passes clean input, AND it fires on dirty input.
A leakage check that can't fail is decoration; these tests prove ours can.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.leakage_check import (
    check_forbidden_features,
    prove_forbidden_absent,
    check_temporal_consistency,
    flag_suspicious_auc,
    single_feature_aucs,
    check_single_feature_auc,
    DEFAULT_FORBIDDEN,
)


# ---------------------------------------------------------------------------
# Fixtures — a clean frame and a leaked frame, defined once and reused.
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_features() -> list[str]:
    """The 9 application-time features the project actually ships."""
    return [
        "revenue", "dti_n", "loan_amnt", "fico_n",
        "emp_length_ord", "emp_length_missing",
        "purpose", "home_ownership_n", "addr_state",
    ]


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A frame with only decision-time columns — no post-decision fields."""
    return pd.DataFrame({
        "fico_n": [700, 680, 720],
        "dti_n": [12.0, 25.99, 8.5],
        "loan_amnt": [10000, 20000, 5000],
        "Default": [0, 1, 0],
    })


@pytest.fixture
def leaked_df() -> pd.DataFrame:
    """A frame that accidentally reintroduced two post-decision fields."""
    return pd.DataFrame({
        "fico_n": [700, 680],
        "int_rate": [13.5, 18.9],       # forbidden: LC's own pricing
        "recoveries": [0.0, 450.0],     # forbidden: only > 0 after default
        "Default": [0, 1],
    })


# ---------------------------------------------------------------------------
# 1. check_forbidden_features — passes clean, fires on dirty
# ---------------------------------------------------------------------------
class TestCheckForbiddenFeatures:
    def test_clean_features_pass(self, clean_features):
        msg = check_forbidden_features(clean_features)
        assert msg.startswith("OK")
        assert "9 candidate features" in msg

    def test_single_forbidden_feature_raises(self):
        with pytest.raises(ValueError, match="int_rate"):
            check_forbidden_features(["fico_n", "int_rate"])

    def test_all_violations_listed_at_once(self):
        """A good guard surfaces every problem, not just the first."""
        with pytest.raises(ValueError) as exc:
            check_forbidden_features(["fico_n", "int_rate", "grade", "recoveries"])
        message = str(exc.value)
        assert "int_rate" in message
        assert "grade" in message
        assert "recoveries" in message

    def test_empty_feature_list_is_vacuously_ok(self):
        assert check_forbidden_features([]).startswith("OK")

    def test_custom_blocklist_is_respected(self):
        with pytest.raises(ValueError, match="my_leak"):
            check_forbidden_features(["my_leak"], forbidden_list=["my_leak"])


# ---------------------------------------------------------------------------
# 2. prove_forbidden_absent — silence turned into proof
# ---------------------------------------------------------------------------
class TestProveForbiddenAbsent:
    def test_clean_df_confirms_all_absent(self, clean_df):
        report = prove_forbidden_absent(clean_df)
        assert report["present_leakage"] == []
        assert len(report["confirmed_absent"]) == len(DEFAULT_FORBIDDEN)

    def test_leaked_df_raises(self, leaked_df):
        with pytest.raises(ValueError, match="int_rate|recoveries"):
            prove_forbidden_absent(leaked_df)

    def test_report_shape_is_stable(self, clean_df):
        report = prove_forbidden_absent(clean_df)
        assert set(report.keys()) == {"present_leakage", "confirmed_absent"}


# ---------------------------------------------------------------------------
# 3. check_temporal_consistency — catches future-dated features
# ---------------------------------------------------------------------------
class TestTemporalConsistency:
    def test_no_leakage_returns_empty(self):
        df = pd.DataFrame({
            "issue_d": ["2016-01-01", "2016-02-01"],
            "last_credit_pull_d": ["2015-12-01", "2016-01-15"],  # both BEFORE
        })
        bad = check_temporal_consistency(df, "last_credit_pull_d", "issue_d")
        assert bad.empty

    def test_future_dated_feature_is_flagged(self):
        df = pd.DataFrame({
            "issue_d": ["2016-01-01", "2016-02-01"],
            "last_credit_pull_d": ["2016-06-01", "2016-01-15"],  # row 0 is AFTER
        })
        bad = check_temporal_consistency(df, "last_credit_pull_d", "issue_d")
        assert len(bad) == 1
        assert "2016-06-01" in bad["last_credit_pull_d"].astype(str).values

    def test_unparseable_dates_do_not_crash(self):
        """coerce turns junk into NaT; NaT comparisons are False, not errors."""
        df = pd.DataFrame({
            "issue_d": ["2016-01-01", "not a date"],
            "feat_d": ["2016-06-01", "2016-02-01"],
        })
        bad = check_temporal_consistency(df, "feat_d", "issue_d")
        assert len(bad) == 1  # only the first, valid, genuinely-late row


# ---------------------------------------------------------------------------
# 4. flag_suspicious_auc — the "0.9957 red flag" detector
# ---------------------------------------------------------------------------
class TestFlagSuspiciousAuc:
    def test_realistic_aucs_not_flagged(self):
        aucs = {"fico_n": 0.65, "dti_n": 0.58, "loan_amnt": 0.61}
        assert flag_suspicious_auc(aucs) == {}

    def test_leaky_feature_flagged(self):
        """A standalone 0.99 is the signature of the failed loan-risk repo."""
        flagged = flag_suspicious_auc({"fico_n": 0.65, "total_pymnt": 0.99})
        assert flagged == {"total_pymnt": 0.99}

    def test_threshold_is_configurable(self):
        aucs = {"a": 0.82, "b": 0.71}
        assert flag_suspicious_auc(aucs, threshold=0.80) == {"a": 0.82}

    def test_boundary_is_strict_greater_than(self):
        """Exactly at threshold is NOT flagged (strictly greater)."""
        assert flag_suspicious_auc({"a": 0.90}, threshold=0.90) == {}


# ---------------------------------------------------------------------------
# 5. single_feature_aucs + check_single_feature_auc — the computation half of
# sentinel 4, and the fail-closed gate that lets the pipeline wire it in.
# Same both-sides discipline: pass clean input AND fire on dirty input.
# ---------------------------------------------------------------------------
def _synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Mixed clean/leaky features around a binary target."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    nan_mask = rng.uniform(size=n) < 0.3
    return pd.DataFrame({
        "weak": 0.5 * y + rng.normal(0, 1.0, n),      # mild legitimate signal
        "noise": rng.normal(0, 1.0, n),                # no signal at all
        "leak": y + rng.normal(0, 0.01, n),            # numeric leak, AUC ~1
        "anti_leak": -y + rng.normal(0, 0.01, n),      # negatively-oriented leak
        "cat_leak": np.where(y == 1, "bad", "good"),   # categorical leak
        "cat_ok": rng.choice(["a", "b", "c"], n),      # uninformative categorical
        "with_nan": np.where(nan_mask, np.nan, 0.5 * y + rng.normal(0, 1.0, n)),
        "constant": 1.0,                               # degenerate: single value
        "Default": y,
    })


class TestSingleFeatureAucs:
    def test_orientation_normalized_to_upper_half(self):
        """max(auc, 1-auc): a protective/negative proxy must not hide below 0.5."""
        df = _synthetic_frame()
        aucs = single_feature_aucs(df, ["weak", "noise", "anti_leak"], "Default")
        assert all(v >= 0.5 for v in aucs.values())
        assert aucs["anti_leak"] > 0.95  # invisible to a >0.5-only detector

    def test_categorical_target_encoding(self):
        df = _synthetic_frame()
        aucs = single_feature_aucs(
            df, ["cat_leak", "cat_ok"], "Default",
            categorical=["cat_leak", "cat_ok"],
        )
        assert aucs["cat_leak"] > 0.95
        assert aucs["cat_ok"] < 0.7

    def test_nan_rows_dropped_pairwise_without_crash(self):
        df = _synthetic_frame()
        aucs = single_feature_aucs(df, ["with_nan"], "Default")
        assert 0.5 <= aucs["with_nan"] <= 1.0

    def test_degenerate_feature_skipped_not_scored(self):
        df = _synthetic_frame()
        aucs = single_feature_aucs(df, ["constant", "noise"], "Default")
        assert "constant" not in aucs
        assert "noise" in aucs


class TestCheckSingleFeatureAuc:
    def test_clean_features_pass(self):
        df = _synthetic_frame()
        msg = check_single_feature_auc(
            df, ["weak", "noise", "cat_ok"], "Default", categorical=["cat_ok"],
        )
        assert msg.startswith("OK")
        assert "3 features checked" in msg

    def test_numeric_leak_fails_closed(self):
        df = _synthetic_frame()
        with pytest.raises(ValueError, match="leak"):
            check_single_feature_auc(df, ["weak", "leak"], "Default")

    def test_all_violations_listed_at_once(self):
        """Same discipline as check_forbidden_features: report everything."""
        df = _synthetic_frame()
        with pytest.raises(ValueError) as exc:
            check_single_feature_auc(
                df, ["leak", "anti_leak", "cat_leak"], "Default",
                categorical=["cat_leak"],
            )
        message = str(exc.value)
        assert "leak" in message
        assert "anti_leak" in message
        assert "cat_leak" in message