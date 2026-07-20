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

from src.evaluate import (
    INT_MARGIN,
    LGD,
    PROFIT_OBJECTIVE,
    evaluate_at_threshold,
    pure_profit,
    regret_profit,
    select_threshold,
    total_profit,
)
from src.features import TARGET


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

    best_t, curve = select_threshold(splits, p_val=p_true)   # default objective = regret

    # This test only exercises the DIRECTION (cost asymmetry -> below 0.5). The
    # exact break-even depends on the objective: under the default "regret"
    # objective total_profit optimizes, it is 2m/(2m+LGD) = 0.270, NOT the
    # pure-profit m/(m+LGD) = 0.156 -- because the rejected-good opportunity-cost
    # term double-weights approving goods (see PROFIT_OBJECTIVE in evaluate.py).
    # The two exact break-evens are pinned in
    # test_regret_objective_breakeven_is_pinned /
    # test_pure_profit_objective_breakeven_is_pinned below.
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


# ---------------------------------------------------------------------------
# 5. (W3) The cost objective has an EXPLICIT identity now. The two objectives
# have different break-even thresholds for a well-calibrated p, and that
# difference moves the chosen operating point -- so each break-even is pinned.
# Previously the only threshold assertion was `best_t < 0.5`, which let the
# regret-vs-pure-profit conflation pass unnoticed. These make it fail.
# ---------------------------------------------------------------------------
def _wellcalibrated_val(n: int = 150_000, seed: int = 0):
    """A large, well-calibrated Val slice (p IS the true PD), so the
    profit-maximizing threshold lands on the cost model's analytic break-even
    rather than on finite-sample noise."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    y = rng.binomial(1, p)
    loan_amt = rng.uniform(1_000, 35_000, n)
    return {"val": pd.DataFrame({"loan_amnt": loan_amt, TARGET: y})}, p


def test_regret_objective_breakeven_is_pinned():
    """Default objective. The rejected-good opportunity-cost term double-weights
    approving goods, so break-even is 2m/(2m+LGD) = 0.270 -- NOT pure profit's
    m/(m+LGD) = 0.156. If the objective ever silently reverts to pure_profit,
    best_t collapses to ~0.15 and this fails."""
    assert PROFIT_OBJECTIVE == "regret"
    assert 2 * INT_MARGIN / (2 * INT_MARGIN + LGD) == pytest.approx(0.2697, abs=1e-3)
    splits, p = _wellcalibrated_val()
    best_t, _ = select_threshold(splits, p_val=p)          # default = regret
    assert 0.24 <= best_t <= 0.30, best_t                  # brackets 0.270
    assert best_t > 0.20                                    # firmly off the 0.156 side


def test_pure_profit_objective_breakeven_is_pinned():
    """Same data, objective='pure_profit'. Originated-book P&L has break-even
    m/(m+LGD) = 0.156, distinctly below regret's 0.270 -- pins the second
    identity so the two objectives can never be conflated again."""
    assert INT_MARGIN / (INT_MARGIN + LGD) == pytest.approx(0.1558, abs=1e-3)
    splits, p = _wellcalibrated_val()
    best_t, _ = select_threshold(splits, p_val=p, objective="pure_profit")
    assert 0.12 <= best_t <= 0.19, best_t                  # brackets 0.156


def test_two_objectives_pick_different_thresholds():
    """The crux of W3: the two objectives are not the same function. On
    identical data they select materially different operating points."""
    splits, p = _wellcalibrated_val()
    best_regret, _ = select_threshold(splits, p_val=p, objective="regret")
    best_pure, _ = select_threshold(splits, p_val=p, objective="pure_profit")
    assert best_regret > best_pure + 0.05


def test_regret_minus_pure_is_rejected_good_opportunity_cost():
    """regret_profit == pure_profit minus the foregone margin on rejected goods,
    exactly -- the algebraic identity behind the shifted break-even."""
    loan_amt = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    y        = np.array([0,      1,      0,      1])
    approved = np.array([True,   True,   False,  False])
    rejected_good_cost = 3000.0 * INT_MARGIN   # loan 2 is the only rejected good
    diff = pure_profit(loan_amt, y, approved) - regret_profit(loan_amt, y, approved)
    assert diff == pytest.approx(rejected_good_cost)


def test_total_profit_default_is_regret():
    """total_profit() with no objective arg == regret_profit, so every existing
    caller keeps its documented behavior."""
    loan_amt = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    y        = np.array([0,      1,      0,      1])
    approved = np.array([True,   True,   False,  False])
    assert total_profit(loan_amt, y, approved) == regret_profit(loan_amt, y, approved)


def test_unknown_objective_raises():
    """A typo in the objective name fails closed, not silently to a default."""
    loan_amt = np.array([1000.0])
    y = np.array([0])
    approved = np.array([True])
    with pytest.raises(ValueError, match="Unknown profit objective"):
        total_profit(loan_amt, y, approved, objective="max_vibes")
