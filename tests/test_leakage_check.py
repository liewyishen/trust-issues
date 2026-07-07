"""
Tests for src/leakage_check.py

The leakage module's whole job is to *prove* the data is clean rather than
assume it. These tests hold that module to the same standard: every guard is
checked on both sides — it passes clean input, AND it fires on dirty input.
A leakage check that can't fail is decoration; these tests prove ours can.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.leakage_check import (
    check_forbidden_features,
    prove_forbidden_absent,
    check_temporal_consistency,
    flag_suspicious_auc,
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