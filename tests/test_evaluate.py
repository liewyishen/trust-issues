"""
Tests for src/evaluate.py

Locks down the CONTRACT, not real-dataset dollar amounts (those only mean
something at real scale -- see run_evaluation()'s docstring for the real
~$190M / 0.26 / 80.3% numbers this module reproduces against the real CSV):
  1. total_profit's arithmetic matches a hand-computed example.
  2. Cost asymmetry (LGD > INT_MARGIN) drives the optimal threshold well
     below 0.5 -- conservative rejection is cheaper than a wrong approval.
  3. Threshold selection reads only Val, evaluation reads only Test -- the
     evaluation-leakage discipline is enforced structurally (missing the
     OTHER split's key does not raise), not just by convention.
  4. approval_rate / bad_rate outputs are valid probabilities in [0, 1].

Run:  pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import TARGET
from src.evaluate import (
    total_profit,
    select_threshold,
    evaluate_at_threshold,
    LGD,
    INT_MARGIN,
)


# ---------------------------------------------------------------------------
# 1. total_profit's arithmetic matches a hand-computed example.
# ---------------------------------------------------------------------------
def test_total_profit_matches_hand_computation():
    loan_amt = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    y        = np.array([0,      1,      0,      1])
    approved = np.array([True,   True,   False,  False])

    # approved good (loan 0):  +1000 * 0.12 = +120
    # approved bad  (loan 1):  -2000 * 0.65 = -1300
    # rejected good (loan 2):  -3000 * 0.12 = -360
    # rejected bad  (loan 3):   0 (correctly avoided, no term)
    expected = 120.0 - 1300.0 - 360.0

    assert total_profit(loan_amt, y, approved) == pytest.approx(expected)


def test_total_profit_all_rejected_is_pure_opportunity_cost():
    loan_amt = np.array([1000.0, 2000.0])
    y = np.array([0, 1])
    approved = np.array([False, False])

    # Only the rejected-good loan costs anything (foregone margin); the
    # rejected-bad loan contributes zero either way.
    expected = -(1000.0 * INT_MARGIN)
    assert total_profit(loan_amt, y, approved) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 2. Cost asymmetry (LGD > INT_MARGIN) pushes the optimal threshold well
# below 0.5 -- conservative rejection, not a coin-flip cutoff.
# ---------------------------------------------------------------------------
def test_cost_asymmetry_drives_threshold_below_half():
    assert LGD > INT_MARGIN  # the premise this test exercises

    rng = np.random.default_rng(5)
    n = 6000
    # A well-calibrated p (p IS the true default probability here) with a
    # full spread over [0, 1], so the profit-maximizing cutoff reflects the
    # cost model's break-even point rather than being an artifact of a
    # narrow probability range.
    p_true = rng.uniform(0.0, 1.0, n)
    y = rng.binomial(1, p_true)
    loan_amt = rng.uniform(1_000, 35_000, n)

    val_df = pd.DataFrame({"loan_amnt": loan_amt, TARGET: y})
    splits = {"val": val_df}

    best_t, curve = select_threshold(splits, p_val=p_true)

    # Break-even for a well-calibrated p under this cost model is
    # INT_MARGIN / (INT_MARGIN + LGD) = 0.12 / 0.77 ~= 0.156 -- comfortably
    # below 0.5. Assert the direction the task calls for without pinning to
    # that exact break-even value (finite-sample noise perturbs the argmax).
    assert best_t < 0.5
    assert set(curve.columns) == {"threshold", "val_profit"}


# ---------------------------------------------------------------------------
# 3. Evaluation-leakage discipline: select_threshold never needs "test",
# evaluate_at_threshold never needs "val" -- proven by NOT providing the
# other split's key at all (a KeyError would surface any accidental read).
# ---------------------------------------------------------------------------
def test_select_threshold_never_touches_test_split():
    rng = np.random.default_rng(6)
    n = 500
    p_val = rng.uniform(0.0, 1.0, n)
    y = rng.binomial(1, p_val)
    loan_amt = rng.uniform(1_000, 35_000, n)

    val_only_splits = {"val": pd.DataFrame({"loan_amnt": loan_amt, TARGET: y})}

    best_t, curve = select_threshold(val_only_splits, p_val=p_val)

    assert 0.05 <= best_t <= 0.94
    assert len(curve) > 0


def test_evaluate_at_threshold_never_touches_val_split():
    rng = np.random.default_rng(7)
    n = 500
    p_test = rng.uniform(0.0, 1.0, n)
    y = rng.binomial(1, p_test)
    loan_amt = rng.uniform(1_000, 35_000, n)

    test_only_splits = {"test": pd.DataFrame({"loan_amnt": loan_amt, TARGET: y})}

    result = evaluate_at_threshold(test_only_splits, best_t=0.26, p_test=p_test)

    assert "threshold" in result


# ---------------------------------------------------------------------------
# 4. approval_rate / bad_rate outputs are valid [0, 1] probabilities.
# ---------------------------------------------------------------------------
def test_evaluate_outputs_are_valid_probabilities():
    rng = np.random.default_rng(8)
    n = 2000
    p_test = rng.uniform(0.0, 1.0, n)
    y = rng.binomial(1, p_test)
    loan_amt = rng.uniform(1_000, 35_000, n)

    splits = {"test": pd.DataFrame({"loan_amnt": loan_amt, TARGET: y})}
    result = evaluate_at_threshold(splits, best_t=0.30, p_test=p_test)

    assert 0.0 <= result["approval_rate"] <= 1.0
    assert 0.0 <= result["bad_rate_approved"] <= 1.0
    assert 0.0 <= result["bad_rate_rejected"] <= 1.0
