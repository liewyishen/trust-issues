"""
Data loading and temporal split for the LendingClub granting-model dataset.

This module fixes two pieces of notebooks/analysis.ipynb in place as reusable,
testable code:
  - Cell 2  : read the raw CSV, parse issue_d, derive issue_year.
  - Cell 15 : the four-way temporal split (Train / Val / Calib / Test) plus
              a disjoint 2018 sensitivity holdout.

It also closes the gap left by data_validation.py being an island: load_raw()
runs validate_loan_data() right after parsing and before any split happens, so
a malformed dataframe never reaches modeling code. That check is fail-closed
by design -- pandera's SchemaErrors is allowed to propagate, not caught, since
a training run on unvalidated data is worse than a training run that doesn't
happen.

Two entry points:
  - load_raw(data_path=None, validate=True) -> pd.DataFrame
  - temporal_split(df) -> dict[str, pd.DataFrame]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_validation import validate_loan_data

# ---------------------------------------------------------------------------
# Constants -- every value here traces back to notebooks/analysis.ipynb
# Cell 2 (loading) and Cell 15 (split), not a guess.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "LC_loans_granting_model_dataset.csv"

RANDOM_SEED = 42

# Train on the past, test on the future. Test (2016-2017) is meant to be
# touched exactly once, at final evaluation -- nothing upstream of this
# module enforces that, but the split itself never mixes years across it.
TRAIN_YEAR_LO, TRAIN_YEAR_HI = 2007, 2014
VAL_CALIB_YEAR = 2015
TEST_YEAR_LO, TEST_YEAR_HI = 2016, 2017
HOLDOUT_YEAR_2018 = 2018

# Val and Calib are disjoint 40k-row slices carved out of the shuffled 2015
# data. 2015 must contain at least 2 * N_VAL rows for both to be full-size.
N_VAL = 40_000

_SPLIT_DISPLAY_NAMES = {
    "train": "Train",
    "val": "Val",
    "calib": "Calib",
    "test": "Test",
    "holdout_2018": "2018 (holdout)",
}


def slice_years(frame: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """
    Return rows whose issue_year falls in the closed interval [lo, hi].

    Parameters
    ----------
    frame : pd.DataFrame
        Must already have an `issue_year` column (see load_raw).
    lo, hi : int
        Inclusive year bounds.

    Returns
    -------
    pd.DataFrame
        A copy, so downstream mutation (feature engineering, etc.) never
        touches the original frame or other slices.
    """
    return frame[(frame["issue_year"] >= lo) & (frame["issue_year"] <= hi)].copy()


def load_raw(data_path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    """
    Load the raw LendingClub CSV, parse dates, and pass it through the
    data-validation gate.

    Parameters
    ----------
    data_path : str, Path, or None
        Path to the CSV. Defaults to DEFAULT_DATA_PATH
        (PROJECT_ROOT / "data" / "LC_loans_granting_model_dataset.csv").
    validate : bool
        If True (default), call validate_loan_data() on the parsed frame
        before returning it. Any contract violation raises
        pandera.errors.SchemaErrors -- this is a fail-closed gate, not a
        warning, so it is never caught here. Callers who need to load a
        small, deliberately non-conformant sample (e.g. in tests) can pass
        validate=False to bypass the gate explicitly.

    Returns
    -------
    pd.DataFrame
        The loaded frame with `issue_d` parsed to datetime and `issue_year`
        derived from it. If validate=True, this is the (possibly
        type-coerced) frame returned by validate_loan_data().

    Raises
    ------
    pandera.errors.SchemaErrors
        If validate=True and the frame does not honor LOAN_SCHEMA.
    """
    path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    df = pd.read_csv(path, low_memory=False)

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["issue_year"] = df["issue_d"].dt.year

    if validate:
        df = validate_loan_data(df)

    return df


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Four-way temporal split, plus a disjoint 2018 sensitivity holdout.

    Train (2007-2014) -> Val/Calib (disjoint 40k slices of shuffled 2015)
    -> Test (2016-2017) -> 2018 holdout. This mirrors notebooks/analysis.ipynb
    Cell 15 exactly: 2015 is shuffled once with random_state=RANDOM_SEED, then
    sliced by position so Val and Calib never overlap. 2018 is excluded from
    Test because its observed default rate is anomalously low (suspected
    right-censoring, see the notebook's EDA) -- it comes back as a separate
    sensitivity holdout, never mixed into Test.

    Parameters
    ----------
    df : pd.DataFrame
        Must already have `issue_year` and `Default` columns (see load_raw).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "train", "val", "calib", "test", "holdout_2018".

    On the full LendingClub dataset this yields approximately:
        Train           n=453,804  default_rate=0.1702
        Val             n= 40,000  default_rate=0.2057
        Calib           n= 40,000  default_rate=0.2020
        Test            n=462,174  default_rate=0.2323
        2018 (holdout)  n= 56,160  default_rate=0.1575
    """
    df_train = slice_years(df, TRAIN_YEAR_LO, TRAIN_YEAR_HI)
    df_2015 = slice_years(df, VAL_CALIB_YEAR, VAL_CALIB_YEAR).sample(
        frac=1, random_state=RANDOM_SEED
    )

    df_val = df_2015.iloc[:N_VAL].copy()
    df_calib = df_2015.iloc[N_VAL : 2 * N_VAL].copy()
    df_test = slice_years(df, TEST_YEAR_LO, TEST_YEAR_HI)
    df_2018 = slice_years(df, HOLDOUT_YEAR_2018, HOLDOUT_YEAR_2018)

    splits = {
        "train": df_train,
        "val": df_val,
        "calib": df_calib,
        "test": df_test,
        "holdout_2018": df_2018,
    }

    for key, d in splits.items():
        label = _SPLIT_DISPLAY_NAMES[key]
        print(f"{label:<16} n={len(d):>8,}  default_rate={d['Default'].mean():.4f}")

    return splits
