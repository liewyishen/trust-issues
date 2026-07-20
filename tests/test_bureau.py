"""
Tests for serving/bureau.py

Locks down two things, and nothing about scoring: the /score wiring
(serving/app.py fetches fico_n through bureau.fetch() before scoring) is
exercised by tests/test_serving.py, so nothing here touches ScoreRequest,
explain_applicants, or serving/app.py.

  1. CreditReport's contract: fico_n and dti_n are bound to constants
     imported from src/data_validation.py, not redeclared -- for dti_n the
     same DTI_MAX_REAL/DTI_SENTINEL pair ScoreRequest still binds, for
     fico_n the FICO_MIN/FICO_MAX bound that lives on CreditReport alone now
     that ScoreRequest no longer carries the field -- and provenance metadata
     cannot be blank or out-of-enum.
  2. MockBureau is deterministic: fetch(applicant_id) returns a
     byte-identical report every call for the same applicant_id -- in one
     process, across instances, and ACROSS PROCESSES -- and the report it
     returns always satisfies CreditReport's own validation. The third is a
     separate claim from the first two, needs a subprocess to see, and is the
     one a reproducible demo actually rests on.

Run:  pytest tests/test_bureau.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from serving.bureau import CreditBureau, CreditReport, MockBureau
from src.data_validation import DTI_MAX_REAL, DTI_SENTINEL, FICO_MAX, FICO_MIN

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Distinct from each other; their values are otherwise arbitrary. The assertion
# is that they AGREE, so what matters is only that they differ -- and that they
# are set explicitly, rather than trusting this interpreter's seed to differ
# from a child's by luck.
_HASH_SEEDS = ("0", "1", "12345")

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
#    The same FICO_MIN/FICO_MAX that ScoreRequest.fico_n bound before Phase 1
#    moved the field to the bureau pull.
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


# The bureau roster, PINNED -- and deliberately not derived.
#
# serving/bureau.py declares `bureau: Literal[...]` inline; there is no importable
# constant, so the old version of the test below copied the four names into its
# own loop. That copy could not notice a fifth name being added: it iterated its
# memory of the enum, checked those four still validated, and passed while the
# name "the closed enum" had come to mean a different set. Measured, not assumed
# -- with "innovis" added to the Literal, that test reported `1 passed`.
#
# Deriving the loop's domain does NOT fix it, and this is the interesting part.
# get_args() over the live Literal, asserting each member validates, is VACUOUS:
# pydantic accepts Literal members by construction, so the derived loop would
# dutifully test "innovis", accept it, and pass -- weaker than the copy, not
# stronger, because it derives its domain from the very thing that changed. It
# would also stop noticing a DELETION, which the copy does catch (drop "mock" and
# CreditReport(bureau="mock") raises).
#
# So the copy is not the defect here. A pin cannot be derived from the thing it
# pins -- being written by hand is what makes it a pin. What was missing is that
# nothing compared the pin to the live enum. get_args() belongs on the OTHER side
# of that comparison: derive what the code says, pin what we decided, assert they
# are equal. Exact equality, not membership -- the same shape as
# `set(rc) == REASON_CODE_KEYS`, which is why that one has never drifted.
#
# Adding a bureau is allowed. Adding one without touching this line is not: a new
# bureau means a new provenance contract, and /score's callers read this field.
_EXPECTED_BUREAUS = frozenset({"equifax", "experian", "transunion", "mock"})


def test_bureau_is_restricted_to_the_closed_enum():
    """Under-claims on purpose: the name says the enum is closed, and the body
    additionally pins WHICH enum. A name narrower than its body is safe.

    What "the closed enum" refers to is now checkable -- before, the definite
    article pointed at a roster the test had copied, so "the" meant whichever set
    the code happened to hold.
    """
    live = frozenset(get_args(CreditReport.model_fields["bureau"].annotation))
    assert live == _EXPECTED_BUREAUS, (
        f"serving/bureau.py's bureau Literal is now {sorted(live)}, not "
        f"{sorted(_EXPECTED_BUREAUS)}. If a bureau was added, it needs a "
        "provenance decision (who pulls it, what fico_version it reports) and "
        "this pin updated in the same commit -- not after someone notices /score "
        "returning a source no client knows how to read."
    )
    for known in sorted(live):
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
# 6. MockBureau is deterministic: same applicant_id -> identical report, every
#    call, across separate MockBureau instances, and across separate PROCESSES.
#
#    The third is not a stronger version of the first two. It is a different
#    claim, it is the one MockBureau.fetch's own comment calls "exactly the
#    failure mode determinism exists to rule out", and until
#    test_determinism_holds_across_separate_processes below, nothing asserted it.
# ---------------------------------------------------------------------------
def test_same_applicant_produces_same_report():
    bureau = MockBureau()
    first = bureau.fetch("applicant-0001")
    second = bureau.fetch("applicant-0001")
    assert first == second


def test_determinism_holds_across_separate_bureau_instances_in_one_process():
    """No hidden per-instance state -- two fresh MockBureau()s agree too.

    `in_one_process` is in the name because that is the whole of what this
    asserts. The old name -- test_determinism_holds_across_separate_bureau_
    instances -- read as if "separate" reached further than it does, and it does
    not: both instances live in this interpreter, under this interpreter's one
    PYTHONHASHSEED. The cross-process claim is a different test, immediately
    below, and nothing here implies it. Renamed, not reassigned: the assertion
    is byte-for-byte the one it always made.
    """
    first = MockBureau().fetch("applicant-0001")
    second = MockBureau().fetch("applicant-0001")
    assert first == second


def test_determinism_holds_across_separate_processes():
    """A report must not depend on which interpreter asked for it.

    THE PROPERTY. "The same applicant id always returns the same credit report"
    is not a claim about one run. A demo reproduced tomorrow, on another machine,
    in CI, is a SECOND PROCESS -- so if the report only holds still within an
    interpreter, the sentence the frontend shows the user is false in the only
    situation anyone would check it. Reproducing across processes is not a
    stronger form of determinism here; it is the form that means anything.

    SHA-256 is HOW. MockBureau.fetch's own comment says why, and this test exists
    to hold that decision rather than let its reasoning live in prose: Python's
    built-in hash() is randomized per process (PYTHONHASHSEED), so seeding off it
    would make the same applicant_id draw a different report in every new
    interpreter -- "exactly the failure mode determinism exists to rule out", in
    that comment's words. It was right, and nothing enforced it.

    WHY A SUBPROCESS IS THE POINT. No same-process check can see this. Measured:
    with hash() substituted for SHA-256, every other test in this section stays
    green, because within one interpreter hash() IS stable. The bug is invisible
    to any assertion that does not cross the process boundary.

    Seeds are set EXPLICITLY, and there is more than one, so nothing rests on
    this interpreter's own seed being anything in particular: the proof is that
    the children AGREE WITH EACH OTHER. That they also agree with this process is
    the second assertion, and it would fail on its own if the parent were the odd
    one out.

    On the socket guard (tests/conftest.py): measured, a child interpreter does
    NOT inherit it, and spawning one records no violation -- subprocess uses
    pipes, not sockets. So these children run outside that guard. That is safe
    for a reason, not by luck: MockBureau performs no I/O, which is the property
    the guard exists to protect and this test exists to depend on. The same gap
    already applies to test_serving.py's import-graph subprocess.
    """
    _PROBE = (
        "import sys; sys.path.insert(0, %r); "
        "from serving.bureau import MockBureau; "
        "print(MockBureau().fetch('applicant-0001').model_dump_json())"
        % (str(PROJECT_ROOT),)
    )

    mine = MockBureau().fetch("applicant-0001").model_dump_json()

    theirs: dict[str, str] = {}
    for seed in _HASH_SEEDS:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True, text=True, check=True,
            cwd=PROJECT_ROOT, env={**os.environ, "PYTHONHASHSEED": seed},
        )
        theirs[seed] = result.stdout.strip()

    assert len(set(theirs.values())) == 1, (
        "MockBureau.fetch('applicant-0001') returned DIFFERENT reports under "
        "different PYTHONHASHSEEDs:\n"
        + "\n".join(f"  PYTHONHASHSEED={s}: {r}" for s, r in theirs.items())
        + "\n\nSomething on the path from applicant_id to the report is seeded "
        "off Python's built-in hash(), which is randomized per process. The same "
        "applicant now draws a different report in every new interpreter -- so a "
        "demo cannot be reproduced tomorrow, and the frontend's sentence about "
        "the bureau is false. See MockBureau.fetch: the draw must be seeded off "
        "hashlib.sha256(applicant_id), never hash()."
    )

    assert set(theirs.values()) == {mine}, (
        "Child interpreters agree with each other but not with this one:\n"
        f"  this process : {mine}\n"
        f"  children     : {next(iter(theirs.values()))}\n\n"
        "The report depends on something that differs between this process and a "
        "fresh one -- an env var, cwd, import order or module-level state -- "
        "rather than on applicant_id alone."
    )


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
