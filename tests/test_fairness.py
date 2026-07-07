"""
Tests for src/fairness.py

Locks down the CONTRACT of each audit layer, not real-dataset numbers (those
only mean something at real scale -- see fairness.py's module docstring for
the real MS 0.734 -> 0.990 finding this module reproduces against the real
CSV):
  1. bootstrap_eo_ci returns a valid (lo <= hi) interval.
  2. Layer 1 flags a state whose good applicants are systematically
     under-approved as "confirmed", and does not flag a normal state.
  3. Layer 1's min_n filter drops small-sample states instead of reporting
     a noisy CI for them.
  4. Layer 3's ablation moves a "shortcut" state's EO ratio in the expected
     direction once addr_state is removed (direction only, not magnitude).

Run:  pytest tests/test_fairness.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness import (
    bootstrap_eo_ci,
    audit_layer1,
    audit_layer3_ablation,
)


# ---------------------------------------------------------------------------
# 1. bootstrap_eo_ci returns a valid interval.
# ---------------------------------------------------------------------------
def test_bootstrap_eo_ci_returns_valid_interval():
    rng = np.random.default_rng(0)
    state_approved = rng.integers(0, 2, size=1000).astype(float)
    base_rate = 0.5

    lo, hi = bootstrap_eo_ci(state_approved, base_rate, n_boot=500, seed=1)

    assert lo <= hi
    assert lo >= 0.0


# ---------------------------------------------------------------------------
# Layer 1 fixture: a synthetic fair_df with one systematically-suppressed
# state ("BAD") and several normal states, sized well above MIN_N so bootstrap
# CIs are meaningful. Deliberately clean separation (BAD's good applicants
# always land above the threshold, normal states' good applicants always
# land below it) rather than noisy in-between values -- this makes the test
# robust/non-flaky. Real EO ratios in practice are noisier, which is exactly
# why Layer 1 bootstraps a CI instead of trusting a point estimate.
# ---------------------------------------------------------------------------
def _make_fair_df(rng, n_per_state=800, small_n=50):
    threshold = 0.2
    records = []

    for st in ("CA", "TX", "NY"):
        y_true = rng.binomial(1, 0.2, n_per_state)
        p = np.where(
            y_true == 0,
            rng.uniform(0.0, 0.15, n_per_state),   # good applicants: always approved
            rng.uniform(0.3, 0.9, n_per_state),
        )
        records.append(pd.DataFrame({"addr_state": st, "y_true": y_true, "p": p}))

    # BAD: good applicants are systematically pushed ABOVE the threshold,
    # i.e. rejected regardless of their true (good) outcome.
    y_true_bad = rng.binomial(1, 0.2, n_per_state)
    p_bad = np.where(
        y_true_bad == 0,
        rng.uniform(0.25, 0.5, n_per_state),       # good applicants: always rejected
        rng.uniform(0.5, 0.9, n_per_state),
    )
    records.append(pd.DataFrame({"addr_state": "BAD", "y_true": y_true_bad, "p": p_bad}))

    # TINY: same suppressed-approval pattern as BAD, but far too few good
    # applicants to trust -- must be filtered out by min_n, not flagged.
    y_true_tiny = rng.binomial(1, 0.2, small_n)
    p_tiny = np.where(
        y_true_tiny == 0,
        rng.uniform(0.25, 0.5, small_n),
        rng.uniform(0.5, 0.9, small_n),
    )
    records.append(pd.DataFrame({"addr_state": "TINY", "y_true": y_true_tiny, "p": p_tiny}))

    return pd.concat(records, ignore_index=True), threshold


def test_layer1_flags_suppressed_state_as_confirmed():
    rng = np.random.default_rng(3)
    fair_df, threshold = _make_fair_df(rng)

    report = audit_layer1(fair_df, threshold=threshold, min_n=500)

    bad_row = report[report["state"] == "BAD"].iloc[0]
    assert bad_row["verdict"].startswith("confirmed")
    assert bad_row["ci_high"] < 0.80


def test_layer1_does_not_flag_normal_state():
    rng = np.random.default_rng(3)
    fair_df, threshold = _make_fair_df(rng)

    report = audit_layer1(fair_df, threshold=threshold, min_n=500)

    for st in ("CA", "TX", "NY"):
        row = report[report["state"] == st].iloc[0]
        assert row["verdict"] == "clear"


# ---------------------------------------------------------------------------
# 3. min_n filters out small-sample states.
# ---------------------------------------------------------------------------
def test_layer1_min_n_filters_small_state():
    rng = np.random.default_rng(3)
    fair_df, threshold = _make_fair_df(rng, small_n=50)

    report = audit_layer1(fair_df, threshold=threshold, min_n=500)

    assert "TINY" not in set(report["state"])
    assert "BAD" in set(report["state"])


# ---------------------------------------------------------------------------
# 4. Layer 3 ablation: removing a feature used as a geographic shortcut
# moves that state's EO ratio toward parity (direction only -- not chasing
# the real dataset's 0.734 -> 0.990).
# ---------------------------------------------------------------------------
def _make_split(n, purposes, homes, states, emp_lengths, rng, shortcut_state=None, shortcut_bump=0.0, base_default_rate=0.15):
    fico = rng.uniform(620, 820, n)
    dti = rng.uniform(0, 40, n)
    state_arr = rng.choice(states, n)

    # Same simplified logistic signal as test_calibrate.py, so early
    # stopping has real structure to chase instead of plateauing after a
    # handful of rounds. shortcut_bump adds an EXTRA, state-only risk term,
    # independent of dti/fico -- a state that carries risk information the
    # model can only get from the addr_state label itself, not from any
    # other feature. That is precisely what Layer 3 is designed to detect:
    # a model relying on addr_state should lose that information (and see
    # the shortcut state's good-applicant approval rate move toward parity)
    # once addr_state is removed and the model can only judge applicants by
    # dti/fico, which do NOT differ for this state in this synthetic setup.
    z = 0.05 * (dti - 20) - 0.03 * (fico - 700)
    if shortcut_state is not None:
        z = z + np.where(state_arr == shortcut_state, shortcut_bump, 0.0)
    p_default = 0.5 / (1 + np.exp(-z)) + 0.5 * base_default_rate
    p_default = np.clip(p_default, 0.01, 0.99)
    default = rng.binomial(1, p_default)

    return pd.DataFrame({
        "revenue": rng.uniform(20_000, 150_000, n),
        "dti_n": dti,
        "loan_amnt": rng.uniform(1_000, 35_000, n),
        "fico_n": fico,
        "emp_length": rng.choice(emp_lengths, n),
        "purpose": rng.choice(purposes, n),
        "home_ownership_n": rng.choice(homes, n),
        "addr_state": state_arr,
        "Default": default,
    })


def test_layer3_ablation_removes_shortcut_and_raises_eo_ratio():
    # No "val" split needed: audit_layer3_ablation trains both variants
    # itself with a FIXED round count (no early stopping / valid_sets), the
    # same discipline notebook Cell 41 uses for its ablation retrain -- and
    # since the refactor, Layer 3 is fully self-contained: it no longer
    # takes a pre-loaded "with-state" fair_df at all, it builds and trains
    # both the with-state and no-state models itself from these splits.
    rng = np.random.default_rng(21)
    purposes = ["debt_consolidation", "credit_card", "other"]
    homes = ["MORTGAGE", "RENT", "OWN"]
    emp_lengths = ["< 1 year", "5 years", "10+ years", "NI"]
    states = ["CA", "TX", "NY", "MS"]

    splits = {
        "train": _make_split(1000, purposes, homes, states, emp_lengths, rng, shortcut_state="MS", shortcut_bump=0.9),
        "calib": _make_split(600, purposes, homes, states, emp_lengths, rng, shortcut_state="MS", shortcut_bump=0.9),
        "test": _make_split(400, purposes, homes, states, emp_lengths, rng, shortcut_state="MS", shortcut_bump=0.9),
    }

    layer3 = audit_layer3_ablation(
        splits, num_boost_round=120, threshold=0.30, watch_states=["MS"],
    )

    row = layer3["states"].set_index("state").loc["MS"]
    assert row["eo_no_state"] > row["eo_with_state"]
