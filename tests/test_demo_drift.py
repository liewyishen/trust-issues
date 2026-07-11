"""
Tests for scripts/demo_drift.py -- guard that the drift demo's CONCLUSION still
holds, never that a specific PSI/KS value is exactly some number.

The demo's claim is a set of behavior invariants, not a table of magic numbers:
turn MockBureau's FICO knob 700 -> 650 and drift_check FIRES; leave it at 700
and it stays QUIET; dti_n (which the knob never touches) is a valid negative
control. Each test below locks one of those, and only ever asserts "over / under
the threshold" or "the two runs agree" -- an exact PSI would be the same
implementation-brittle mistake we just scrubbed out of the docs.

Nothing is mocked: the tests drive the demo's own build_batch / run_monitor,
which call drift_check.py's real drift_metrics + evaluate_alarms against
drift_check.py's own DEFAULT_ALARM_THRESHOLDS. If the real pipeline stopped
firing, these would fail -- which is the point.

Run:  pytest tests/test_demo_drift.py -v
"""

from __future__ import annotations

import pytest

from pipelines.drift_check import DEFAULT_ALARM_THRESHOLDS
from scripts.demo_drift import (
    MEAN_DOWNTURN,
    MEAN_NORMAL,
    YEAR_CUR,
    YEAR_REF,
    build_batch,
    run_monitor,
)
from serving.bureau import MockBureau

# Fixed, deterministic applicant ids -- no RNG anywhere, same discipline as the
# demo. Smaller than the demo's own N (these invariants hold at any size); big
# enough that the control run's sampling noise sits well under the thresholds.
N = 1000
REF_IDS = [f"ref-{i:04d}" for i in range(N)]
CUR_IDS = [f"cur-{i:04d}" for i in range(N)]


def _batches():
    """
    One reference population @700 and two current populations @700 / @650 that
    SHARE CUR_IDS -- so between them fico_n shifts by the knob while dti_n stays
    identical. Exactly the demo's own construction, just smaller.
    """
    reference = build_batch(MockBureau(mean_fico=MEAN_NORMAL), REF_IDS, YEAR_REF)
    normal = build_batch(MockBureau(mean_fico=MEAN_NORMAL), CUR_IDS, YEAR_CUR)
    downturn = build_batch(MockBureau(mean_fico=MEAN_DOWNTURN), CUR_IDS, YEAR_CUR)
    return reference, normal, downturn


def test_downturn_trips_the_fico_alarms():
    # 650 vs 700: fico_n's PSI and KS must BREACH drift_check's own thresholds
    # and the FICO alarm list must be non-empty. Asserts "past the threshold",
    # never a specific value.
    reference, _, downturn = _batches()
    result = run_monitor(reference, downturn, "downturn")
    assert result["psi_fico"] > DEFAULT_ALARM_THRESHOLDS["psi"]
    assert result["ks_fico"] > DEFAULT_ALARM_THRESHOLDS["ks"]
    assert result["fico_alarms"]  # non-empty -> the monitor fired on FICO


def test_control_stays_quiet():
    # 700 vs 700: same market, so fico_n's PSI and KS must stay UNDER the
    # thresholds and no FICO alarm may fire. This is what makes the downturn
    # alarm meaningful -- the monitor does not fire on everything.
    reference, normal, _ = _batches()
    result = run_monitor(reference, normal, "control")
    assert result["psi_fico"] < DEFAULT_ALARM_THRESHOLDS["psi"]
    assert result["ks_fico"] < DEFAULT_ALARM_THRESHOLDS["ks"]
    assert result["fico_alarms"] == []


def test_dti_is_an_unmoved_negative_control():
    # The knob touches only fico_n, so dti_n's drift signals must be the SAME in
    # the control and downturn runs (both current batches carry identical dti_n).
    # Locks the negative control's validity as a relative invariant -- the two
    # runs agree -- not as an absolute number.
    reference, normal, downturn = _batches()
    control_metrics = run_monitor(reference, normal, "control")["metrics"]
    downturn_metrics = run_monitor(reference, downturn, "downturn")["metrics"]
    assert control_metrics[f"psi_dti_n_{YEAR_CUR}"] == pytest.approx(
        downturn_metrics[f"psi_dti_n_{YEAR_CUR}"]
    )
    assert control_metrics[f"ks_dti_n_{YEAR_CUR}"] == pytest.approx(
        downturn_metrics[f"ks_dti_n_{YEAR_CUR}"]
    )


def test_demo_is_deterministic():
    # No randomness in the path (fixed ids + hash-seeded MockBureau), so the
    # same inputs must yield the same drift numbers on a fresh build.
    reference_a, _, downturn_a = _batches()
    first = run_monitor(reference_a, downturn_a, "downturn")
    reference_b, _, downturn_b = _batches()
    second = run_monitor(reference_b, downturn_b, "downturn")
    assert first["psi_fico"] == pytest.approx(second["psi_fico"])
    assert first["ks_fico"] == pytest.approx(second["ks_fico"])
