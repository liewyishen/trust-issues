"""
Data validation for the LendingClub granting-model dataset.

The EDA already *found* the landmines in this data:
  - dti_n has a 999 sentinel (a placeholder for "uncomputable" dressed up
    as a number),
  - revenue has an 8-figure self-reported tail,
  - fico_n lives in a narrow, sane band,
  - a handful of categoricals must stay inside a known set.

This module turns those *observations* into an automatic, fail-closed gate.
Instead of a reader trusting that the loaded frame looks like the EDA frame,
`validate_loan_data` proves it — and raises the moment the data drifts from
what the model was built on. That is the same discipline as leakage_check.py:
don't assume the data is clean, prove it, and stop the pipeline if it isn't.

Known data-quality investigation (full write-up: docs/data-decisions.md):
wiring this gate into load_raw() and running it against the real CSV for the
first time surfaced 495 rows with dti_n between 100 and 999 -- neither
in-band nor the known sentinel. The investigation concluded these are a real
2016+ high-DTI borrower population, not corrupt data, which is why dti_n's
real-value ceiling below is 1000, not 100. This is a live train/serve
distribution-shift signal (Train, 2007-2014, has none of these rows; Test and
the 2018 holdout do) -- so pipelines/drift_check.py now monitors dti_n's
year-over-year distribution, not just its marginal range. See the dti_n
Check in LOAN_SCHEMA below for the full evidence trail.

Two entry points:
  - LOAN_SCHEMA          : the Pandera schema (declarative contract)
  - validate_loan_data() : fail-closed wrapper — returns the frame or raises

Run the doctest:  python -m pytest --doctest-modules src/data_validation.py
"""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Domain constants — every number here traces back to a specific EDA finding,
# not a guess. If a value changes, the comment says which observation justified
# it, so the contract stays auditable.
# ---------------------------------------------------------------------------

# dti_n = 999 is a sentinel, not a real debt-to-income ratio: a placeholder
# for "uncomputable" dressed up as a number. It is tracked separately from
# DTI_MAX_REAL below because its meaning is categorically different from a
# real high DTI, even though (post-widening) it now also falls inside the
# real-value range numerically.
DTI_SENTINEL = 999.0

# DTI_MAX_REAL was originally 100 ("real DTI sits roughly 0-60, anything past
# 100 must be junk or the sentinel"). That held until the validation gate was
# wired into load_raw() and run against the full real CSV for the first time
# (see docs/data-decisions.md), which surfaced 495 rows with dti_n strictly
# between 100.04 and 991.57 -- neither in the old [0,100] band nor the 999
# sentinel.
#
# The investigation ruled out the two obvious alternatives:
#   - NOT a decimal-shift artifact: dividing these 495 values by 100 would
#     make them the LOWEST-leverage borrowers in the dataset (median ~1.5%
#     DTI), predicting a BELOW-average default rate. The observed default
#     rate among them is 27.07% vs. 19.98% overall -- the wrong direction for
#     a "should've been divided by 100" typo.
#   - NOT sentinel/encoding residue: sentinels cluster on a handful of
#     reused, exact values. These 495 rows are continuous and almost
#     entirely unique (483/495 distinct values), at the same two-decimal
#     precision as ordinary DTI.
#
# What survives: a real subpopulation of high-income, high-loan-amount
# borrowers (median revenue $103k vs. $65k; median loan_amnt $19.2k vs $12k)
# whose reported DTI genuinely exceeds 100%. Critically, this population is
# almost entirely a 2016+ phenomenon: 0 rows in Train (2007-2014), 7 in 2015,
# 488 in 2016-2018 -- consistent with LendingClub changing its DTI
# computation/reporting methodology around 2015-2016, not with corruption.
# That means the model NEVER sees this DTI regime in Train, only in Test and
# the 2018 holdout: a genuine train/serve distribution shift, not just a
# data-hygiene footnote -- now watched by pipelines/drift_check.py, which
# monitors dti_n's distribution by issue_year, not just its range.
#
# DTI_MAX_REAL is therefore widened to 1000: wide enough to admit the
# observed real ceiling (991.57) with a small margin, narrow enough that a
# value with no evidentiary basis (e.g. a stray 9999) still trips the gate.
DTI_MAX_REAL = 1000.0

# FICO in the source is a tight, sane band (~612–847). We pad it slightly so
# legitimate edge scores pass, but a 0 or a 900 (data error) is caught.
FICO_MIN, FICO_MAX = 300.0, 900.0

# Loan amounts are capped at $40k in the product; $500 floor is the smallest
# offered. Outside this range means the join brought in the wrong column.
LOAN_MIN, LOAN_MAX = 500.0, 40_000.0

# revenue has a real 8-figure tail (self-reported). We don't clip it (trees are
# rank-based), but a negative or zero income is a data error, not a rich outlier.
REVENUE_MIN = 1.0

# Closed vocabularies straight out of the EDA value_counts. handle_unknown at
# serve time is separate; here we assert the TRAINING contract.
VALID_HOME_OWNERSHIP = {"MORTGAGE", "RENT", "OWN", "OTHER"}
VALID_PURPOSE = {
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "medical", "small_business", "car", "vacation",
    "moving", "house", "wedding", "renewable_energy", "educational",
}


# ---------------------------------------------------------------------------
# The schema — a declarative contract for the modeling frame.
# nullable / coerce choices are deliberate and commented.
# ---------------------------------------------------------------------------
LOAN_SCHEMA = DataFrameSchema(
    {
        "revenue": Column(
            float,
            checks=Check.greater_than_or_equal_to(REVENUE_MIN),
            nullable=False,
            coerce=True,
            description="Annual income; positive, heavy right tail kept as-is.",
        ),
        "dti_n": Column(
            float,
            # Real DTI band [0, 1000] OR the known 999 sentinel. The OR is now
            # numerically redundant (999 <= 1000) but kept explicit on purpose:
            # 999 is semantically a missing-value sentinel, not a real DTI, even
            # though the widened real band happens to contain it. See
            # DTI_MAX_REAL's comment above for the full 495-row investigation
            # that justified widening this from 100 to 1000.
            # Note the lower bound: DTI is never negative, so -1 (another common
            # sentinel) must be REJECTED, not mistaken for a real value.
            checks=Check(
                lambda s: ((s >= 0.0) & (s <= DTI_MAX_REAL)) | (s == DTI_SENTINEL),
                element_wise=False,
                error="dti_n outside real [0,1000] band and not the known 999 sentinel",
            ),
            nullable=False,
            coerce=True,
            description="Debt-to-income; real band widened to [0,1000] after "
                        "confirming a genuine 2016+ high-DTI population (see "
                        "docs/data-decisions.md); 999 remains the distinct "
                        "missing-value sentinel.",
        ),
        "loan_amnt": Column(
            float,
            checks=Check.in_range(LOAN_MIN, LOAN_MAX),
            nullable=False,
            coerce=True,
        ),
        "fico_n": Column(
            float,
            checks=Check.in_range(FICO_MIN, FICO_MAX),
            nullable=False,
            coerce=True,
        ),
        "home_ownership_n": Column(
            str,
            checks=Check.isin(VALID_HOME_OWNERSHIP),
            nullable=False,
        ),
        "purpose": Column(
            str,
            checks=Check.isin(VALID_PURPOSE),
            nullable=False,
        ),
        # emp_length arrives as a string ("< 1 year", "NI", ...) BEFORE feature
        # engineering maps it to ordinal. We only assert it's present & string;
        # the ordinal encoding is validated downstream, not here.
        "emp_length": Column(str, nullable=True),
        "addr_state": Column(
            str,
            checks=Check.str_length(min_value=2, max_value=2),
            nullable=False,
            description="Two-letter US state; kept but audited for redlining.",
        ),
        "Default": Column(
            int,
            checks=Check.isin({0, 1}),
            nullable=False,
            coerce=True,
            description="Binary target: 1 = defaulted, 0 = paid off.",
        ),
    },
    strict=False,   # extra columns (id, issue_d, zip_code...) are allowed through
    coerce=False,   # per-column coerce above; no blanket coercion
    name="LendingClubLoanSchema",
)


def validate_loan_data(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """
    Fail-closed validation gate. Returns the frame if it honors the contract,
    raises pandera.errors.SchemaError(s) with every violation if it doesn't.

    Call this right after load and right before modeling — the same place
    prove_forbidden_absent() runs. Together they answer two questions:
    "is anything leaking?" (leakage_check) and "is anything malformed?" (here).

    Parameters
    ----------
    df : pd.DataFrame
        The loaded modeling frame.
    lazy : bool
        If True (default), collect ALL validation errors before raising, so one
        run surfaces every problem instead of stopping at the first — the same
        "show me everything that's wrong" philosophy as check_forbidden_features.

    Returns
    -------
    pd.DataFrame
        The validated frame (unchanged) on success.

    Raises
    ------
    pandera.errors.SchemaError / SchemaErrors
        On any contract violation.

    Examples
    --------
    >>> import pandas as pd
    >>> good = pd.DataFrame({
    ...     "revenue": [55000.0], "dti_n": [12.0], "loan_amnt": [3600.0],
    ...     "fico_n": [677.0], "home_ownership_n": ["MORTGAGE"],
    ...     "purpose": ["debt_consolidation"], "emp_length": ["10+ years"],
    ...     "addr_state": ["PA"], "Default": [0],
    ... })
    >>> validate_loan_data(good).shape
    (1, 9)
    """
    return LOAN_SCHEMA.validate(df, lazy=lazy)