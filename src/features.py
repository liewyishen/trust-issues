"""
Feature engineering for the LendingClub granting-model dataset.

Fixes notebooks/analysis.ipynb Cell 17 in place as reusable, testable code:
ordinal-encode emp_length, derive an explicit missingness flag, and define
the feature/target contract every downstream module (train.py, calibrate.py,
evaluate.py, fairness.py) imports instead of re-declaring its own copy.

Single responsibility: derive columns and name the feature list. Imputation,
one-hot encoding, and scaling are train.py's job (sklearn Pipeline / LightGBM's
native categorical + NaN handling), not this module's.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# emp_length -> ordinal encoding. Anything not in this map (chiefly "NI", the
# "not disclosed" code) becomes NaN via .map() -- on purpose. This layer does
# not impute; that's left to whichever downstream step consumes emp_length_ord,
# so the "unknown" signal survives instead of being silently filled in here.
# ---------------------------------------------------------------------------
emp_order = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

# ---------------------------------------------------------------------------
# INCLUDE_ADDR_STATE -- production feature-set switch. Default OFF.
#
# fairness.py's three-layer audit (see docs/data-decisions.md, "Execute the
# fairness conclusion: remove addr_state from the production model") found
# addr_state functions as a
# geographic-proxy risk / digital-redlining shortcut: with addr_state in the
# model, Mississippi's good-applicant Equal Opportunity ratio was ~0.734
# (well below the 0.80 regulatory benchmark); removing addr_state and
# retraining brought it to ~0.990, at a cost of only -0.0035 test AUC
# (0.6689 -> 0.6654). That audit's conclusion -- "remove addr_state from the
# production feature set" -- is executed here: default OFF, so the SHIPPED
# model does not use addr_state as a feature.
#
# This is a config switch, not a deletion, on purpose: addr_state's raw
# column, emp_order, and every other piece of feature engineering below stay
# exactly as they were. Flipping this back to True reproduces the
# "with-state" baseline fairness.py's Layer 3 ablation compares against --
# without a switch, that audit could never be re-run or re-verified against
# a freshly trained model, only trusted as a historical finding written down
# once. The choice is made in exactly one place (here), not by whichever
# module happens to slice addr_state out of a column list by hand.
# ---------------------------------------------------------------------------
INCLUDE_ADDR_STATE = False

# ---------------------------------------------------------------------------
# Feature contract -- every downstream module imports these instead of
# re-declaring its own list, so notebook and production can never drift
# apart on what "the model's inputs" means.
#
# experience_c is deliberately excluded: the notebook found it has standalone
# AUC = 0.500 (zero discriminative power) and dropping it moved test AUC by
# only -0.0002 -- dead weight, not signal.
# ---------------------------------------------------------------------------
NUMERIC = ["revenue", "dti_n", "loan_amnt", "fico_n",
           "emp_length_ord", "emp_length_missing"]

# CATEGORICAL_BASE is deliberately toggle-INDEPENDENT -- these two columns
# are never in question, only addr_state is.
CATEGORICAL_BASE = ["purpose", "home_ownership_n"]


def build_categorical(include_addr_state: bool = INCLUDE_ADDR_STATE) -> list[str]:
    """
    Build the categorical feature list for a given addr_state choice.

    Defaults to the current INCLUDE_ADDR_STATE setting, so
    build_categorical() == CATEGORICAL always. Exposed as a function (not
    just the module-level CATEGORICAL constant) so fairness.py's Layer 3
    ablation -- which must compare a with-state model against a no-state
    model regardless of which one happens to be shipped right now -- can
    build BOTH variants explicitly (build_categorical(True) and
    build_categorical(False)) without depending on whatever
    INCLUDE_ADDR_STATE is currently set to in this file.
    """
    return CATEGORICAL_BASE + (["addr_state"] if include_addr_state else [])


CATEGORICAL = build_categorical()
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Default"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive emp_length_ord and emp_length_missing on a copy of df.

    Stateless and row-wise -- safe to call independently on each split
    (train/val/calib/test/holdout_2018) with identical behavior.

    Parameters
    ----------
    df : pd.DataFrame
        Must have an `emp_length` column (string, e.g. "10+ years", "NI").

    Returns
    -------
    pd.DataFrame
        A copy of df with two new columns:
          - emp_length_ord     : emp_length mapped through emp_order, NaN
                                  for anything unmapped (chiefly "NI").
          - emp_length_missing : 1 where emp_length == "NI", else 0. Kept as
                                  an explicit feature rather than silently
                                  imputed away, because declining to disclose
                                  tenure may itself carry a risk signal.
    """
    out = df.copy()
    out["emp_length_ord"] = out["emp_length"].map(emp_order)
    out["emp_length_missing"] = out["emp_length"].eq("NI").astype(int)
    return out
