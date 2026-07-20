"""
Tests for src/fairness.py

Locks down the CONTRACT of each audit layer, not real-dataset numbers (those
only mean something at real scale -- see fairness.py's module docstring for
the real MS 0.745 -> 0.988 finding this module reproduces against the real
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
    audit_layer1,
    audit_layer3_ablation,
    bootstrap_eo_ci,
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
# the real dataset's 0.745 -> 0.988).
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


# ---------------------------------------------------------------------------
# 5. Layer 3 hands back the Test-level frame for BOTH variants it trains, so a
# caller can put a bootstrap CI on each side of the ablation instead of
# comparing two bare point estimates. Three properties, because "it returned
# two DataFrames" is not the claim worth locking:
#
#   (a) the frames satisfy audit_layer1/audit_layer2's input contract, so a
#       caller reuses the REAL audit rather than growing a second CI path;
#   (b) they are the SOURCE of the EO ratios layer3 already reports, not a
#       parallel artifact that could drift from them;
#   (c) they are a controlled pair -- same applicants, same outcomes, one
#       feature toggled -- which is the only thing that makes them comparable.
# ---------------------------------------------------------------------------
def _ablation_layer3(rng_seed=21, threshold=0.30):
    rng = np.random.default_rng(rng_seed)
    purposes = ["debt_consolidation", "credit_card", "other"]
    homes = ["MORTGAGE", "RENT", "OWN"]
    emp_lengths = ["< 1 year", "5 years", "10+ years", "NI"]
    states = ["CA", "TX", "NY", "MS"]

    splits = {
        name: _make_split(n, purposes, homes, states, emp_lengths, rng,
                          shortcut_state="MS", shortcut_bump=0.9)
        for name, n in (("train", 1000), ("calib", 600), ("test", 400))
    }
    layer3 = audit_layer3_ablation(
        splits, num_boost_round=120, threshold=threshold, watch_states=["MS"],
    )
    return layer3, threshold


def test_layer3_fair_dfs_satisfy_audit_layer1s_input_contract():
    """(a) The point of returning them: audit_layer1 eats them as-is."""
    layer3, threshold = _ablation_layer3()

    for key in ("fair_df_with_state", "fair_df_no_state"):
        report = audit_layer1(layer3[key], threshold=threshold, min_n=50)
        assert list(report.columns) == [
            "state", "n_good", "eo_ratio", "ci_low", "ci_high", "verdict",
        ]
        assert "MS" in set(report["state"])
        assert (report["ci_low"] <= report["ci_high"]).all()


def test_layer3_fair_dfs_are_the_source_of_the_eo_ratios_layer3_reports():
    """
    (b) Recompute MS's EO ratio straight off each returned frame and demand it
    equals the ratio layer3 already published for that variant. If these frames
    were a re-derivation rather than the very arrays the ratios came from, this
    is where that would show up.
    """
    layer3, threshold = _ablation_layer3()
    reported = layer3["states"].set_index("state").loc["MS"]

    for side, key in (("eo_with_state", "fair_df_with_state"),
                      ("eo_no_state", "fair_df_no_state")):
        d = layer3[key]
        approved = (d["p"] < threshold).astype(int)
        good = d["y_true"] == 0
        eo = approved[good & (d["addr_state"] == "MS")].mean() / approved[good].mean()
        assert eo == pytest.approx(reported[side])


def test_layer3_fair_dfs_are_a_controlled_pair_only_the_predictions_differ():
    """
    (c) Same applicants, same ground truth, one feature toggled. If the two
    frames disagreed on addr_state or y_true they would not be an ablation at
    all, just two models scored on two datasets.
    """
    layer3, _ = _ablation_layer3()
    with_state = layer3["fair_df_with_state"]
    no_state = layer3["fair_df_no_state"]

    assert (with_state["addr_state"].values == no_state["addr_state"].values).all()
    assert (with_state["y_true"].values == no_state["y_true"].values).all()
    # The one thing that MUST differ -- otherwise the ablation changed nothing.
    assert not np.allclose(with_state["p"].values, no_state["p"].values)
