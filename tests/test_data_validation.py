"""
Tests for src/data_validation.py

Same discipline as test_leakage_check.py: every guard is checked on BOTH sides.
The schema must PASS data that matches the EDA'd contract, AND must FIRE on data
that drifts from it. A validation gate that only ever passes is decoration; these
tests prove ours actually stops the specific landmines the EDA found — the 999
sentinel's evil twins, out-of-band FICO, unknown categories, oversized loans.

Run:  pytest tests/test_data_validation.py -v
"""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from src.data_validation import (
    DTI_SENTINEL,
    validate_loan_data,
)


# ---------------------------------------------------------------------------
# Fixtures — one canonical clean frame, mutated per-test into a dirty one.
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_frame() -> pd.DataFrame:
    """
    Two rows that honor the full contract. Row 1 deliberately carries the
    KNOWN dti_n=999 sentinel, so 'clean' includes 'the sentinel we tolerate'.
    Extra columns (id, zip_code) are present to prove strict=False lets the
    non-modeling columns ride along.
    """
    return pd.DataFrame({
        "revenue":          [55000.0, 65000.0],
        "dti_n":            [12.0, DTI_SENTINEL],   # real value + tolerated 999
        "loan_amnt":        [3600.0, 24700.0],
        "fico_n":           [677.0, 717.0],
        "home_ownership_n": ["MORTGAGE", "MORTGAGE"],
        "purpose":          ["debt_consolidation", "small_business"],
        "emp_length":       ["10+ years", "10+ years"],
        "addr_state":       ["PA", "SD"],
        "Default":          [0, 0],
        # extra, non-modeling columns — should pass through untouched:
        "id":               [68407277, 68355089],
        "zip_code":         ["190xx", "577xx"],
    })


def _mutate(frame: pd.DataFrame, col: str, value) -> pd.DataFrame:
    """Return a copy with one cell corrupted — keeps tests terse and readable."""
    bad = frame.copy()
    bad.loc[0, col] = value
    return bad


# ---------------------------------------------------------------------------
# 1. The happy path — clean data (incl. the 999 sentinel) must pass unchanged.
# ---------------------------------------------------------------------------
class TestCleanDataPasses:
    def test_clean_frame_validates(self, clean_frame):
        out = validate_loan_data(clean_frame)
        assert len(out) == 2

    def test_known_sentinel_is_tolerated(self, clean_frame):
        """dti_n = 999 is a KNOWN sentinel, not an error — it must pass."""
        out = validate_loan_data(clean_frame)
        assert (out["dti_n"] == DTI_SENTINEL).any()

    def test_extra_columns_ride_through(self, clean_frame):
        """strict=False: id/zip_code are not modeling features but shouldn't fail."""
        out = validate_loan_data(clean_frame)
        assert "id" in out.columns and "zip_code" in out.columns

    def test_dti_boundaries_pass(self, clean_frame):
        """0 and 100 are well inside the widened real band [0, 1000]."""
        edge = clean_frame.copy()
        edge.loc[0, "dti_n"] = 0.0
        edge.loc[1, "dti_n"] = 100.0
        assert len(validate_loan_data(edge)) == 2


# ---------------------------------------------------------------------------
# 2. dti_n — real band widened to [0, 1000] after the 495-row investigation
#    (docs/data-decisions.md) confirmed a genuine 2016+ high-DTI borrower
#    population up to 991.57. 999 is still the distinct missing sentinel;
#    values with no evidentiary basis must still be rejected.
# ---------------------------------------------------------------------------
class TestDtiSentinelLogic:
    def test_negative_dti_rejected(self, clean_frame):
        """-1 is another common sentinel; DTI is never negative, so it must fail.
        (This is the exact bug the first draft missed: only the upper bound was
        guarded, so -1 slipped through. Regression-locked here.)"""
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "dti_n", -1.0))

    def test_high_but_real_dti_now_passes(self, clean_frame):
        """500 was rejected before the widening. The investigation found a real
        2016+ high-DTI population up to 991.57, so it's admitted now."""
        out = validate_loan_data(_mutate(clean_frame, "dti_n", 500.0))
        assert (out["dti_n"] == 500.0).any()

    def test_observed_max_real_dti_passes(self, clean_frame):
        """991.57 is the actual observed maximum from the investigation."""
        out = validate_loan_data(_mutate(clean_frame, "dti_n", 991.57))
        assert (out["dti_n"] == 991.57).any()

    def test_dti_beyond_widened_ceiling_rejected(self, clean_frame):
        """1500 is well past the observed real ceiling (991.57) -- the gate
        must still fire on values with no evidentiary basis."""
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "dti_n", 1500.0))

    def test_new_fake_sentinel_rejected(self, clean_frame):
        """If a future data source swaps 999 for 9999, the gate must trip."""
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "dti_n", 9999.0))


# ---------------------------------------------------------------------------
# 3. Numeric range guards — FICO / loan amount out of band = data error.
# ---------------------------------------------------------------------------
class TestNumericRanges:
    def test_fico_too_high_rejected(self, clean_frame):
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "fico_n", 1200.0))

    def test_fico_zero_rejected(self, clean_frame):
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "fico_n", 0.0))

    def test_loan_amount_over_cap_rejected(self, clean_frame):
        """Product caps loans at $40k; $999,999 means a bad join brought a wrong col."""
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "loan_amnt", 999_999.0))

    def test_negative_revenue_rejected(self, clean_frame):
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "revenue", -100.0))


# ---------------------------------------------------------------------------
# 4. Categorical vocabularies — unknown category = out-of-contract.
# ---------------------------------------------------------------------------
class TestCategoricalVocabularies:
    def test_unknown_purpose_rejected(self, clean_frame):
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "purpose", "crypto_gambling"))

    def test_unknown_home_ownership_rejected(self, clean_frame):
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "home_ownership_n", "SPACESHIP"))

    def test_malformed_state_rejected(self, clean_frame):
        """addr_state must be a 2-letter code; 'Pennsylvania' is malformed."""
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "addr_state", "Pennsylvania"))


# ---------------------------------------------------------------------------
# 5. Target integrity — Default must stay binary.
# ---------------------------------------------------------------------------
class TestTargetIntegrity:
    def test_nonbinary_target_rejected(self, clean_frame):
        with pytest.raises(pa_errors.SchemaErrors):
            validate_loan_data(_mutate(clean_frame, "Default", 2))


# ---------------------------------------------------------------------------
# 6. Fail-closed philosophy — lazy=True surfaces EVERY error at once.
# ---------------------------------------------------------------------------
class TestFailClosedReporting:
    def test_multiple_violations_all_surface(self, clean_frame):
        """Two independent problems (bad fico + unknown purpose) should both be
        reported in one run, not stop at the first — same 'show me everything'
        contract as check_forbidden_features."""
        bad = clean_frame.copy()
        bad.loc[0, "fico_n"] = 1200.0
        bad.loc[0, "purpose"] = "crypto_gambling"
        with pytest.raises(pa_errors.SchemaErrors) as exc:
            validate_loan_data(bad, lazy=True)
        message = str(exc.value)
        assert "fico_n" in message
        assert "purpose" in message
