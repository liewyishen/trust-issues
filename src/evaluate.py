"""
Cost-sensitive threshold selection for the LendingClub granting-model
LightGBM model.

Fixes notebooks/analysis.ipynb Section 8 (Cell 31) in place as reusable,
testable code: instead of an arbitrary 0.5 cutoff, pick the approval
threshold that minimizes expected loss under a stylized cost model, then
apply that ONE threshold to Test exactly once.

Cost model (STYLIZED -- a decision-analysis demonstration, not a production
P&L model; see total_profit()'s docstring for the simplifying assumptions):
    LGD (loss given default) = 0.65   approving a bad borrower loses ~65%
                                       of the loan principal.
    INT_MARGIN               = 0.12   approving a good borrower earns ~12%
                                       of the loan principal in interest
                                       margin.
    => approving a bad borrower costs ~5.4x (0.65 / 0.12) what approving a
       good borrower earns. This asymmetry is WHY the optimal threshold
       sits well below 0.5: a borrower the model thinks is only slightly
       more likely good than bad should still be rejected, because the
       downside of a wrong approval dwarfs the upside of a right one.

Evaluation-leakage discipline (the discipline this whole project is named
for): the threshold is selected by scanning VALIDATION ONLY
(select_threshold), then applied to TEST exactly once (evaluate_at_threshold)
-- never the other way around. Scanning thresholds on Test and reporting
whichever one looks best would let Test's own outcomes leak into the very
decision later evaluated against those same outcomes: the reported
"out-of-sample" profit would actually be an in-sample best case, the exact
failure mode an earlier version of this project made and the one this
project is named for not repeating. select_threshold() only ever reads
splits["val"] and evaluate_at_threshold() only ever reads splits["test"] --
see each function's docstring for how that is enforced structurally (by
what the function does and does not look at), not just by convention.

Honest framing (do not report this module's output as "the model made the
project money"): on the real dataset this reduces losses by roughly $185M
relative to a naive 0.50 cutoff, but the portfolio as a whole still loses
money either way -- this is a ~23% default-rate subprime cohort, and no
threshold choice can turn a structurally unprofitable loan book profitable.
This module's value is LOSS MINIMIZATION, not profit generation. See
run_evaluation()'s docstring for the real numbers.

Reuses train.py's _xy/_to_lgb_frame for encoding and calibrate.py's
apply_calibration/load_calibrator for calibration -- a second,
independently written encoding or calibration path here would drift from
what the shipped model/calibrator were actually fit against (the same
train/serve-skew rationale calibrate.py and fairness.py already apply).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import DEFAULT_MODEL_PATH, apply_calibration, load_calibrator
from .data_loader import load_raw, temporal_split
from .features import TARGET
from .train import _to_lgb_frame, _xy, load_model_artifact

# ---------------------------------------------------------------------------
# Cost model constants -- traced to notebooks/analysis.ipynb Cell 31, not
# guesses. Stylized: real underwriting would layer in term structure, APR,
# funding costs, servicing, prepayment, EAD, and discount rates. This is a
# decision-analysis demonstration, not a production P&L model.
# ---------------------------------------------------------------------------
LGD = 0.65
INT_MARGIN = 0.12

# ---------------------------------------------------------------------------
# PROFIT_OBJECTIVE -- which cost objective threshold selection optimizes. The
# two objectives below have DIFFERENT break-even thresholds for a well-
# calibrated p, and the difference is not cosmetic: it moves the chosen
# operating point. Naming both, and switching between them with an explicit
# constant, is what keeps that choice auditable instead of buried in one
# summation. select_threshold / evaluate_at_threshold / run_evaluation all
# forward an `objective` argument defaulting to this constant.
#
#   "pure_profit"  Originated-book P&L only: +margin on an approved good,
#                  -LGD on an approved bad. A rejected loan never originates,
#                  so it contributes nothing. For a well-calibrated p, approve
#                  iff (1-p)*m - p*LGD > 0  <=>  p < m/(m+LGD) = 0.156.
#
#   "regret"       pure_profit MINUS the foregone interest margin on every
#                  GOOD applicant that was rejected (an opportunity cost).
#                  Because the total good margin G over the population is
#                  fixed, rejected_good_margin = G - approved_good_margin, so
#                  this objective equals  2*approved_good_margin
#                  - approved_bad_LGD - G : it DOUBLE-WEIGHTS approving goods
#                  relative to pure_profit. Break-even shifts to approve iff
#                  (1-p)*m - p*LGD > -(1-p)*m  <=>  p < 2m/(2m+LGD) = 0.270.
#
# Default is "regret": every number the notebook, README, and
# docs/data-decisions.md report (best_t ~0.25-0.26, approval ~78-80%,
# improvement ~$185M) was computed under this objective -- best_t ~0.26 on the
# real data matches 0.270, not 0.156, so the term was always there. Naming it
# only makes the identity explicit; it does not change the shipped behavior.
# Flip to "pure_profit" to select on originated-book P&L instead.
PROFIT_OBJECTIVE = "regret"

DEFAULT_THRESHOLDS = np.arange(0.05, 0.95, 0.01)
NAIVE_THRESHOLD = 0.50


def _profit_terms(loan_amt: np.ndarray, y: np.ndarray, approved: np.ndarray):
    """
    Shared per-loan terms + outcome masks for both profit objectives. Uses
    each loan's REAL loan_amnt, not a portfolio average -- a $35k loan and a
    $1k loan carry very different absolute stakes for the same approve/reject
    decision, and averaging away that spread would distort which threshold
    looks optimal.
    """
    loan_amt = np.asarray(loan_amt)
    y = np.asarray(y)
    approved = np.asarray(approved)

    cost_default = loan_amt * LGD
    profit_good = loan_amt * INT_MARGIN

    approved_good = approved & (y == 0)
    approved_bad = approved & (y == 1)
    rejected_good = (~approved) & (y == 0)
    return profit_good, cost_default, approved_good, approved_bad, rejected_good


def pure_profit(loan_amt: np.ndarray, y: np.ndarray, approved: np.ndarray) -> float:
    """
    Originated-book P&L for one approve/reject decision vector. Only approved
    loans originate, so only they carry a term:
      - approved, good (y == 0):  +loan_amt * INT_MARGIN  (margin earned)
      - approved, bad  (y == 1):  -loan_amt * LGD         (principal lost)
      - rejected (either outcome):  0  (never originated)

    Break-even for a well-calibrated p is m/(m+LGD) = 0.156. See
    PROFIT_OBJECTIVE for how this differs from regret().
    """
    profit_good, cost_default, approved_good, approved_bad, _rej_good = _profit_terms(
        loan_amt, y, approved
    )
    return float(profit_good[approved_good].sum() - cost_default[approved_bad].sum())


def regret_profit(loan_amt: np.ndarray, y: np.ndarray, approved: np.ndarray) -> float:
    """
    pure_profit MINUS the foregone interest margin on every GOOD applicant
    that was rejected (an opportunity cost). Four outcomes, only the last
    carries no term:
      - approved, good (y == 0):  +loan_amt * INT_MARGIN
      - approved, bad  (y == 1):  -loan_amt * LGD
      - rejected, good (y == 0):  -loan_amt * INT_MARGIN  (opportunity cost)
      - rejected, bad  (y == 1):   0                      (correctly avoided)

    This double-weights approving goods relative to pure_profit, moving
    break-even to 2m/(2m+LGD) = 0.270. It is the objective the notebook and
    README numbers were always computed under. See PROFIT_OBJECTIVE.
    """
    profit_good, cost_default, approved_good, approved_bad, rejected_good = _profit_terms(
        loan_amt, y, approved
    )
    return float(
        profit_good[approved_good].sum()
        - cost_default[approved_bad].sum()
        - profit_good[rejected_good].sum()
    )


_PROFIT_OBJECTIVES = {"pure_profit": pure_profit, "regret": regret_profit}


def total_profit(
    loan_amt: np.ndarray,
    y: np.ndarray,
    approved: np.ndarray,
    objective: str = PROFIT_OBJECTIVE,
) -> float:
    """
    Portfolio objective for one approve/reject decision vector, dispatched to
    the chosen cost `objective`. The public entry point every caller
    (select_threshold, evaluate_at_threshold) imports; keeping the dispatch
    here means the objective is chosen in exactly one place, not re-decided at
    each call site.

    Parameters
    ----------
    loan_amt : array-like of float
        Original loan principal, one value per applicant.
    y : array-like of 0/1
        Actual outcome, 1 = defaulted.
    approved : array-like of bool
        The approve/reject decision being evaluated, one value per applicant
        -- typically (p < threshold) for some calibrated PD vector p.
    objective : str
        "regret" (default, break-even 0.270) or "pure_profit" (break-even
        0.156). See PROFIT_OBJECTIVE for the full semantic difference.

    Returns
    -------
    float
        Total portfolio value (can be negative) under this decision and
        objective.

    Raises
    ------
    ValueError
        If `objective` is not a known objective name.
    """
    try:
        fn = _PROFIT_OBJECTIVES[objective]
    except KeyError:
        raise ValueError(
            f"Unknown profit objective {objective!r}; expected one of "
            f"{sorted(_PROFIT_OBJECTIVES)}. See PROFIT_OBJECTIVE in evaluate.py."
        )
    return fn(loan_amt, y, approved)


def _predict_calibrated(
    df: pd.DataFrame,
    model_path: str | Path | None,
    calibrator_path: str | Path | None,
) -> np.ndarray:
    """
    Load the packaged model + a previously fit calibrator, and return
    calibrated predicted probabilities for df.

    Reuses train.py's _xy/_to_lgb_frame for encoding and calibrate.py's
    load_calibrator/apply_calibration for calibration, rather than each
    caller in this module (select_threshold, evaluate_at_threshold)
    independently re-deriving its own encoding or re-fitting its own
    isotonic regression -- exactly the train/serve-skew hazard calibrate.py
    and fairness.py already avoid.

    Note this LOADS a calibrator rather than fitting one: calibrate.py's
    calibrate_model() is what fits and persists it (models/isotonic_
    calibrator.pkl by default). This module is downstream of that step, not
    a replacement for it.
    """
    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    calibrator_path = (
        Path(calibrator_path)
        if calibrator_path is not None
        else model_path.parent / "isotonic_calibrator.pkl"
    )

    artifact = load_model_artifact(model_path)  # fail-closed on feature-contract mismatch
    booster = artifact["model"]
    category_maps = artifact["category_maps"]
    best_iteration = artifact["best_iteration"]
    # Pass the model artifact so a stale calibrator (fit against a different
    # model instance) is rejected rather than silently applied.
    iso = load_calibrator(calibrator_path, model_artifact=artifact)

    X, _y = _xy(df)
    X_lgb = _to_lgb_frame(X, category_maps)
    p_raw = booster.predict(X_lgb, num_iteration=best_iteration)
    return apply_calibration(iso, p_raw)


def select_threshold(
    splits: dict[str, pd.DataFrame],
    model_path: str | Path | None = None,
    calibrator_path: str | Path | None = None,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
    p_val: np.ndarray | None = None,
    objective: str = PROFIT_OBJECTIVE,
) -> tuple[float, pd.DataFrame]:
    """
    Scan candidate thresholds on VALIDATION ONLY and return the
    profit-maximizing one. Mirrors notebooks/analysis.ipynb Cell 31's
    val_profits / best_t computation.

    This function reads ONLY splits["val"] -- splits["test"] is never
    looked at here, by construction (there is no code path in this
    function that indexes any key other than "val"). That is what makes
    "the threshold was chosen on validation" a structural fact about this
    function rather than a promise about how it happens to get called.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        Must contain "val". (Does not need to contain "test" -- see above.)
    model_path, calibrator_path : str, Path, or None
        Forwarded to _predict_calibrated if p_val is not supplied.
    thresholds : np.ndarray
        Candidate cutoffs. Default np.arange(0.05, 0.95, 0.01), matching
        the notebook.
    p_val : np.ndarray or None
        Precomputed calibrated probabilities for splits["val"]. If
        provided, this skips loading the joblib model artifact and
        calibrator entirely -- this is how tests feed small synthetic data
        without needing a real model/calibrator pair on disk.

    Returns
    -------
    (float, pd.DataFrame)
        best_t : the threshold (from `thresholds`) with the highest val
            profit.
        val_profit_curve : pd.DataFrame with columns "threshold" and
            "val_profit", one row per candidate threshold -- the full curve
            best_t was chosen from, useful for plotting/inspection.
    """
    if p_val is None:
        p_val = _predict_calibrated(splits["val"], model_path, calibrator_path)

    y_val = splits["val"][TARGET].values
    loan_val = splits["val"]["loan_amnt"].values

    profits = [total_profit(loan_val, y_val, p_val < t, objective=objective) for t in thresholds]
    best_idx = int(np.argmax(profits))
    best_t = float(thresholds[best_idx])

    val_profit_curve = pd.DataFrame({"threshold": thresholds, "val_profit": profits})
    return best_t, val_profit_curve


def evaluate_at_threshold(
    splits: dict[str, pd.DataFrame],
    best_t: float,
    model_path: str | Path | None = None,
    calibrator_path: str | Path | None = None,
    p_test: np.ndarray | None = None,
    naive_threshold: float = NAIVE_THRESHOLD,
    objective: str = PROFIT_OBJECTIVE,
) -> dict:
    """
    Apply best_t to TEST exactly once. Mirrors notebooks/analysis.ipynb
    Cell 31's test-side reporting.

    This function reads ONLY splits["test"] -- splits["val"] is never
    looked at here. It does not scan thresholds; it evaluates the single
    threshold (best_t) it is given, so there is no way to call this
    function in a manner that tunes best_t against Test. best_t must come
    from select_threshold(), called on splits["val"] beforehand.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        Must contain "test". (Does not need to contain "val".)
    best_t : float
        The threshold chosen on validation by select_threshold(). Applied
        here, once.
    model_path, calibrator_path : str, Path, or None
        Forwarded to _predict_calibrated if p_test is not supplied.
    p_test : np.ndarray or None
        Precomputed calibrated probabilities for splits["test"]. If
        provided, skips loading the joblib model artifact and calibrator
        entirely (see select_threshold's p_val for the same rationale).
    naive_threshold : float
        Comparison baseline -- an arbitrary, un-tuned cutoff. Default 0.50,
        matching the notebook.

    Returns
    -------
    dict
        {"threshold": float, "test_profit": float, "naive_profit": float,
         "naive_threshold": float, "improvement_over_naive": float,
         "approval_rate": float,
         "bad_rate_approved": float, "bad_rate_rejected": float}
    """
    if p_test is None:
        p_test = _predict_calibrated(splits["test"], model_path, calibrator_path)

    y_test = splits["test"][TARGET].values
    loan_test = splits["test"]["loan_amnt"].values

    approved = p_test < best_t
    naive_approved = p_test < naive_threshold

    test_profit = total_profit(loan_test, y_test, approved, objective=objective)
    naive_profit = total_profit(loan_test, y_test, naive_approved, objective=objective)

    return {
        "threshold": best_t,
        "test_profit": test_profit,
        "naive_profit": naive_profit,
        "naive_threshold": naive_threshold,
        "improvement_over_naive": test_profit - naive_profit,
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(y_test[approved].mean()) if approved.any() else float("nan"),
        "bad_rate_rejected": float(y_test[~approved].mean()) if (~approved).any() else float("nan"),
    }


def run_evaluation(
    model_path: str | Path | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
    calibrator_path: str | Path | None = None,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
    naive_threshold: float = NAIVE_THRESHOLD,
    p_val: np.ndarray | None = None,
    p_test: np.ndarray | None = None,
    objective: str = PROFIT_OBJECTIVE,
) -> dict:
    """
    select_threshold() on Val, then evaluate_at_threshold() on Test exactly
    once. Mirrors notebooks/analysis.ipynb Cell 31 end to end.

    On the real dataset (shipped model + calibrator, production feature set
    per features.py's INCLUDE_ADDR_STATE -- see docs/data-decisions.md,
    "Execute the fairness conclusion: remove addr_state from the production
    model"), this reproduces
    approximately:
        Threshold chosen on VAL: 0.25
        Test profit @ 0.25:       (a large negative number)
        Test profit @ naive 0.50: (a more negative number)
        Improvement over naive:  ~+$184,800,000
        Approval rate on test:    ~78.0%
        Bad rate among approved:  ~18.9%
        Bad rate among rejected:  ~38.5%

    Reading the numbers honestly: the portfolio loses money at EITHER
    threshold -- this is a ~23% default-rate subprime cohort, and no
    threshold choice fixes structurally unprofitable unit economics (LGD
    0.65 vs. margin 0.12 requires a much lower base default rate, or much
    higher pricing, to break even). What the chosen threshold achieves is
    reducing losses by ~$185M relative to guessing 0.50 -- loss
    minimization, not profit generation. Do not report this as "the model
    made the project money."

    Parameters
    ----------
    model_path : str, Path, or None
        Path to train.py's packaged model .pkl. Defaults to
        DEFAULT_MODEL_PATH. Ignored wherever p_val/p_test are supplied.
    splits : dict[str, pd.DataFrame] or None
        temporal_split() output; must contain "val" and "test". If None,
        calls load_raw() + temporal_split() itself.
    calibrator_path : str, Path, or None
        Path to calibrate.py's saved isotonic calibrator. Defaults to
        model_path.parent / "isotonic_calibrator.pkl".
    thresholds : np.ndarray
        Candidate cutoffs scanned on Val. Default matches the notebook.
    naive_threshold : float
        Un-tuned comparison baseline. Default 0.50.
    p_val, p_test : np.ndarray or None
        Precomputed calibrated probabilities for splits["val"]/["test"].
        Passing both is how tests exercise this end-to-end without a real
        model/calibrator pair on disk.

    Returns
    -------
    dict
        {"best_threshold": float, "val_profit_curve": pd.DataFrame,
         **evaluate_at_threshold()'s return dict}
    """
    if splits is None:
        df = load_raw()
        splits = temporal_split(df)

    best_t, val_profit_curve = select_threshold(
        splits, model_path=model_path, calibrator_path=calibrator_path,
        thresholds=thresholds, p_val=p_val, objective=objective,
    )
    result = evaluate_at_threshold(
        splits, best_t, model_path=model_path, calibrator_path=calibrator_path,
        p_test=p_test, naive_threshold=naive_threshold, objective=objective,
    )

    print(f"Threshold chosen on VAL: {best_t:.2f}\n")
    print("=== Test (out-of-sample, threshold applied once) ===")
    print(f"Chosen threshold {best_t:.2f}:  profit ${result['test_profit']:,.0f}")
    print(f"Naive {naive_threshold:.2f}:            profit ${result['naive_profit']:,.0f}")
    print(f"Improvement over naive: ${result['improvement_over_naive']:,.0f}\n")
    print(f"Approval rate on test: {result['approval_rate']:.1%}")
    print(f"Bad rate among approved: {result['bad_rate_approved']:.1%}")
    print(f"Bad rate among rejected: {result['bad_rate_rejected']:.1%}")

    return {"best_threshold": best_t, "val_profit_curve": val_profit_curve, **result}
