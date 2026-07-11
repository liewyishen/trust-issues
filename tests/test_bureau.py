"""
Tests for serving/bureau.py

Locks down two things, and nothing about scoring: this module is not wired
into the /score path (see serving/bureau.py's module docstring), so nothing
here touches ScoreRequest, explain_applicants, or serving/app.py.

  1. CreditReport's contract: fico_n and dti_n are held to the SAME bounds
     ScoreRequest already binds to (imported from src/data_validation.py, not
     redeclared), and provenance metadata cannot be blank or out-of-enum.
  2. MockBureau is deterministic: fetch(applicant_id) returns a
     byte-identical report every call for the same applicant_id, and the
     report it returns always satisfies CreditReport's own validation.

Run:  pytest tests/test_bureau.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.data_validation import DTI_MAX_REAL, DTI_SENTINEL, FICO_MAX, FICO_MIN

from serving.bureau import CreditBureau, CreditReport, MockBureau

# One report that honors the full contract. Every mutation test below starts
# from a copy of this, the same pattern GOOD (tests/test_serving.py) and
# clean_frame (tests/test_data_validation.py) both use.
GOOD_REPORT = {
    "applicant_id": "applicant-0001",
    "bureau": "mock",
    "pulled_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "fico_version": "mock-fico-v1",
    "inquiry_window_days": 180,
    "fico_n": 700.0,
    "dti_n": 18.0,
}


def _mutate(**overrides) -> dict:
    return {**GOOD_REPORT, **overrides}


# ---------------------------------------------------------------------------
# 1. The happy path -- a fully valid report must construct.
# ---------------------------------------------------------------------------
def test_good_report_validates():
    report = CreditReport(**GOOD_REPORT)
    assert report.applicant_id == "applicant-0001"
    assert report.bureau == "mock"
    assert report.fico_n == 700.0
    assert report.dti_n == 18.0


# ---------------------------------------------------------------------------
# 2. fico_n -- bounds imported from src/data_validation.py, not redeclared.
#    Same FICO_MIN/FICO_MAX ScoreRequest.fico_n already binds to.
# ---------------------------------------------------------------------------
def test_fico_at_declared_bounds_passes():
    assert CreditReport(**_mutate(fico_n=FICO_MIN)).fico_n == FICO_MIN
    assert CreditReport(**_mutate(fico_n=FICO_MAX)).fico_n == FICO_MAX


def test_fico_below_min_rejected():
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(fico_n=FICO_MIN - 1.0))


def test_fico_above_max_rejected():
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(fico_n=FICO_MAX + 1.0))


def test_fico_is_strict_float_not_a_numeric_string():
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(fico_n="700"))


# ---------------------------------------------------------------------------
# 3. dti_n -- real band [0, DTI_MAX_REAL] OR the known DTI_SENTINEL, mirroring
#    LOAN_SCHEMA's / ScoreRequest's dti_n logic. Never computed -- see the
#    module docstring on why this module refuses to derive dti_n from any
#    other field.
# ---------------------------------------------------------------------------
def test_dti_real_value_passes():
    assert CreditReport(**_mutate(dti_n=37.16)).dti_n == 37.16


def test_dti_sentinel_is_tolerated():
    """999 is a KNOWN missing-value sentinel, not an error -- must pass."""
    assert CreditReport(**_mutate(dti_n=DTI_SENTINEL)).dti_n == DTI_SENTINEL


def test_dti_at_widened_ceiling_passes():
    assert CreditReport(**_mutate(dti_n=DTI_MAX_REAL)).dti_n == DTI_MAX_REAL


def test_negative_dti_rejected():
    """-1 is another common sentinel; DTI is never negative."""
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(dti_n=-1.0))


def test_dti_beyond_widened_ceiling_rejected():
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(dti_n=DTI_MAX_REAL + 500.0))


def test_fake_sentinel_rejected():
    """A future 9999 stand-in is not the known sentinel and must not slip in."""
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(dti_n=9999.0))


# ---------------------------------------------------------------------------
# 4. Provenance metadata -- non-blank, closed bureau enum, positive window.
# ---------------------------------------------------------------------------
def test_provenance_fields_are_present_and_non_blank():
    report = CreditReport(**GOOD_REPORT)
    assert report.applicant_id != ""
    assert report.bureau != ""
    assert report.fico_version != ""
    assert report.pulled_at is not None
    assert report.inquiry_window_days > 0


def test_blank_applicant_id_rejected():
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(applicant_id=""))


def test_blank_fico_version_rejected():
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(fico_version=""))


def test_bureau_is_restricted_to_the_closed_enum():
    for known in ("equifax", "experian", "transunion", "mock"):
        assert CreditReport(**_mutate(bureau=known)).bureau == known
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(bureau="dun_and_bradstreet"))


def test_inquiry_window_days_must_be_positive():
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(inquiry_window_days=0))
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(inquiry_window_days=-30))


def test_extra_field_is_rejected():
    """extra="forbid", the same closed-contract discipline as ScoreRequest."""
    with pytest.raises(ValidationError):
        CreditReport(**_mutate(revol_util=45.0))


# ---------------------------------------------------------------------------
# 5. MockBureau satisfies the CreditBureau protocol.
# ---------------------------------------------------------------------------
def test_mock_bureau_is_a_credit_bureau():
    assert isinstance(MockBureau(), CreditBureau)


# ---------------------------------------------------------------------------
# 6. MockBureau is deterministic: same applicant_id -> identical report,
#    every call, across separate MockBureau instances.
# ---------------------------------------------------------------------------
def test_same_applicant_produces_same_report():
    bureau = MockBureau()
    first = bureau.fetch("applicant-0001")
    second = bureau.fetch("applicant-0001")
    assert first == second


def test_determinism_holds_across_separate_bureau_instances():
    """No hidden per-instance state -- two fresh MockBureau()s agree too."""
    first = MockBureau().fetch("applicant-0001")
    second = MockBureau().fetch("applicant-0001")
    assert first == second


def test_different_applicants_produce_different_reports():
    bureau = MockBureau()
    reports = [bureau.fetch(f"applicant-{i:04d}") for i in range(10)]
    # Not every field need differ, but fico_n/dti_n should not collapse to a
    # single repeated pair across ten distinct applicant_ids.
    scores = {(r.fico_n, r.dti_n) for r in reports}
    assert len(scores) > 1


# ---------------------------------------------------------------------------
# 7. MockBureau's output always satisfies CreditReport's own validation, and
#    carries the fields fetch()'s caller asked about.
# ---------------------------------------------------------------------------
def test_mock_report_is_a_valid_credit_report():
    report = MockBureau().fetch("applicant-0001")
    assert isinstance(report, CreditReport)
    assert FICO_MIN <= report.fico_n <= FICO_MAX
    assert (0.0 <= report.dti_n <= DTI_MAX_REAL) or report.dti_n == DTI_SENTINEL


def test_mock_report_identifies_itself_as_mock():
    report = MockBureau().fetch("applicant-0001")
    assert report.bureau == "mock"


def test_mock_report_carries_the_requested_applicant_id():
    report = MockBureau().fetch("some-other-applicant")
    assert report.applicant_id == "some-other-applicant"


def test_mock_report_pulled_at_does_not_depend_on_wall_clock_time():
    """Two fetches, however far apart in real time, must not differ."""
    a = MockBureau().fetch("applicant-0001").pulled_at
    b = MockBureau().fetch("applicant-0001").pulled_at
    assert a == b
