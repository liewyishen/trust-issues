"""
Leakage detection utilities for credit risk models.

This module operationalizes leakage judgment as reusable, testable code. Rather
than relying on a reader to *trust* that no leakage occurred, these functions
*prove* it.

Four checks are provided:
  1. Forbidden post-decision features are not used   -> check_forbidden_features
  2. Known post-decision fields are absent entirely  -> prove_forbidden_absent
  3. No feature timestamp post-dates the decision     -> check_temporal_consistency
  4. No single feature is suspiciously predictive     -> flag_suspicious_auc
     (single_feature_aucs computes the per-feature standalone AUCs it
      consumes; check_single_feature_auc bundles compute + flag + raise into
      one fail-closed gate for pipeline use)

Dataset context
---------------
This project uses the cleaned 2024 Lending Club "granting model" dataset
(Zenodo, CC BY 4.0). That release has ALREADY removed the worst leakage fields
(int_rate, grade, sub_grade) and all post-origination payment/balance fields.

Because those fields are gone, a naive forbidden-feature check would never fire
and would look like decoration. So this module also exposes prove_forbidden_absent(),
which actively confirms that none of the known post-decision fields survived in the
dataframe. The point is defensive and pedagogical: we know exactly what leakage
looks like, and we verify it is not present rather than assuming the data is clean.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import roc_auc_score

# Fields that are set AFTER the credit decision is made, using ORIGINAL Lending Club
# column names. Using any of these as a model input is classic target leakage: the
# model would learn the outcome (or the lender's own risk grade), not a risk signal
# available at decision time.
#
# NOTE: The cleaned Zenodo "granting model" dataset has already removed these. They
# are listed here on purpose -- as a defensive block list (in case a future raw merge
# reintroduces them) and as documentation of what leakage looks like in this domain.
DEFAULT_FORBIDDEN: list[str] = [
    # --- Pricing / grade fields produced by LC's own risk assessment ---
    "int_rate",      # Interest rate assigned post-approval; encodes LC's risk grade
    "grade",         # LC letter grade (A-G); literally their risk decision
    "sub_grade",     # Finer-grained grade; same problem
    # --- Outcome fields ---
    "loan_status",   # Raw status the target is derived from; direct leakage
    # --- Post-origination payment fields ---
    "total_pymnt",            # Sum of payments received so far; accrues after origination
    "total_pymnt_inv",        # Investor share of total_pymnt
    "total_rec_prncp",        # Principal recovered; post-event
    "total_rec_int",          # Interest recovered; post-event
    "recoveries",             # Collection amounts; only > 0 after default
    "collection_recovery_fee",# Fee on recoveries; same reason
    # --- Post-origination date fields ---
    "last_pymnt_d",       # Date of last payment; only exists after payments are made
    "last_pymnt_amnt",    # Amount of last payment; post-origination
    "next_pymnt_d",       # Scheduled next payment; derived from payment history
    "last_credit_pull_d", # Credit re-pulled after origination for monitoring
    # --- Balance fields that change over the life of the loan ---
    "out_prncp",      # Outstanding principal; decreases as the borrower pays down
    "out_prncp_inv",  # Investor share of outstanding principal
]


def check_forbidden_features(
    feature_list: Sequence[str],
    forbidden_list: Sequence[str] = DEFAULT_FORBIDDEN,
) -> str:
    """
    Raise ValueError if any post-decision forbidden field appears in feature_list.

    Post-decision fields are populated after the credit committee approves or prices
    the loan. They encode the outcome or the lender's own risk assessment, making them
    forms of direct target leakage. No model should train on these.

    Parameters
    ----------
    feature_list : sequence of str
        Column names the model intends to use as inputs.
    forbidden_list : sequence of str
        Fields to block. Defaults to DEFAULT_FORBIDDEN.

    Returns
    -------
    str
        Confirmation message when no forbidden features are found.

    Raises
    ------
    ValueError
        Lists every forbidden feature found, so all violations surface at once.

    Examples
    --------
    >>> check_forbidden_features(["fico_n", "dti_n"])
    'OK: no forbidden features detected among 2 candidate features.'
    >>> check_forbidden_features(["fico_n", "int_rate", "grade"])
    Traceback (most recent call last):
        ...
    ValueError: Forbidden (post-decision) features detected: int_rate, grade ...
    """
    forbidden_set = set(forbidden_list)
    violations = [f for f in feature_list if f in forbidden_set]
    if violations:
        raise ValueError(
            f"Forbidden (post-decision) features detected: {', '.join(violations)}\n"
            "These fields are set after the credit decision and constitute target leakage.\n"
            "See DEFAULT_FORBIDDEN in leakage_check.py for per-field explanations."
        )
    return f"OK: no forbidden features detected among {len(list(feature_list))} candidate features."


def prove_forbidden_absent(
    df: pd.DataFrame,
    forbidden_list: Sequence[str] = DEFAULT_FORBIDDEN,
) -> dict[str, list[str]]:
    """
    Actively confirm that known post-decision fields are absent from the dataframe.

    On a cleaned dataset, check_forbidden_features() can never fire because the
    leakage columns simply do not exist. That silence is not the same as a proof.
    This function turns the silence into an explicit, auditable statement: it reports
    which forbidden fields are present (should be none) and which are confirmed absent.

    Use this right after loading the data, before any feature selection, to document
    that the cleaned release really did remove the known leakage fields -- and to catch
    the case where a later join accidentally reintroduces one.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset to inspect.
    forbidden_list : sequence of str
        Fields expected to be absent. Defaults to DEFAULT_FORBIDDEN.

    Returns
    -------
    dict[str, list[str]]
        {
          "present_leakage": [...],  # forbidden fields found in df.columns (want: empty)
          "confirmed_absent": [...], # forbidden fields verified not present
        }

    Raises
    ------
    ValueError
        If any forbidden field is actually present in df, since that means the
        cleaned-data assumption is violated and downstream modeling is unsafe.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"fico_n": [700], "dti_n": [12.0], "Default": [0]})
    >>> report = prove_forbidden_absent(df)
    >>> report["present_leakage"]
    []
    """
    columns = set(df.columns)
    present = [f for f in forbidden_list if f in columns]
    absent = [f for f in forbidden_list if f not in columns]

    if present:
        raise ValueError(
            f"Known post-decision fields are present in the data: {', '.join(present)}\n"
            "The cleaned-data assumption is violated. Drop these before modeling, or "
            "the model will leak the outcome."
        )
    return {"present_leakage": present, "confirmed_absent": absent}


def check_temporal_consistency(
    df: pd.DataFrame,
    feature_date_col: str,
    decision_date_col: str,
) -> pd.DataFrame:
    """
    Flag rows where a feature's timestamp is strictly after the decision date.

    Future leakage occurs when a feature is populated using data that only existed
    after the loan was originated (e.g. a credit-pull date that post-dates issue_d).
    Even a few such rows can distort learned feature weights.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at least the two date columns.
    feature_date_col : str
        Column name of the feature's timestamp (e.g. "last_credit_pull_d").
    decision_date_col : str
        Column name of the loan origination / decision date (e.g. "issue_d").

    Returns
    -------
    pd.DataFrame
        Subset of rows where feature_date > decision_date. An empty frame means no
        temporal leakage. Both date columns are included to make violations obvious.

    Examples
    --------
    >>> bad_rows = check_temporal_consistency(df, "last_credit_pull_d", "issue_d")
    >>> assert bad_rows.empty, f"{len(bad_rows)} rows have future feature dates"
    """
    feature_dates = pd.to_datetime(df[feature_date_col], errors="coerce")
    decision_dates = pd.to_datetime(df[decision_date_col], errors="coerce")

    mask = feature_dates > decision_dates
    return df.loc[mask, [feature_date_col, decision_date_col]].copy()


def flag_suspicious_auc(
    auc_per_feature: dict[str, float],
    threshold: float = 0.9,
) -> dict[str, float]:
    """
    Return features whose standalone AUC-ROC exceeds the suspicion threshold.

    A single feature that predicts default with AUC > 0.90 is almost certainly leaking
    the target or acting as a direct proxy. Legitimate risk signals rarely exceed ~0.75
    standalone -- real default prediction is genuinely hard. A high standalone AUC is a
    red flag, exactly like the 0.9957 test AUC seen in poorly-built loan-risk projects.

    This is a statistical sanity check, not definitive proof. Flagged features must be
    manually inspected before removal.

    Parameters
    ----------
    auc_per_feature : dict[str, float]
        Mapping from feature name to its standalone AUC-ROC when used as the sole
        predictor (e.g. sklearn.metrics.roc_auc_score against the binary target).
    threshold : float
        AUC above which a feature is considered suspicious. Default 0.9.

    Returns
    -------
    dict[str, float]
        Subset of auc_per_feature where AUC > threshold. Empty dict means no flags.

    Examples
    --------
    >>> flag_suspicious_auc({"fico_n": 0.65, "dti_n": 0.58, "total_pymnt": 0.99})
    {'total_pymnt': 0.99}
    """
    return {feature: auc for feature, auc in auc_per_feature.items() if auc > threshold}


def single_feature_aucs(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    categorical: Sequence[str] = (),
) -> dict[str, float]:
    """
    Compute each candidate feature's standalone AUC-ROC against the target.

    Produces the auc_per_feature mapping flag_suspicious_auc() consumes, so the
    "no single feature is suspiciously predictive" check can run as an
    automated pipeline gate instead of depending on someone remembering to
    hand-compute the AUCs in a notebook.

    Per-feature scoring:
      - Numeric features: the raw values are the ranking score (AUC is
        rank-based, so no scaling is needed). Rows where the feature is NaN
        are dropped pairwise -- e.g. emp_length_ord is NaN wherever emp_length
        was "NI", and those rows simply don't participate in that feature's
        AUC.
      - Categorical features: each category is target-encoded to its mean
        default rate within df, then ranked like a numeric score. In-sample
        encoding is deliberately optimistic -- for a red-flag detector an
        upper bound points the right way: a feature that stays under the
        threshold even with this advantage is definitely not leaking.
      - Orientation: the reported value is max(auc, 1 - auc). A protective
        feature (higher fico -> LESS default) has raw AUC below 0.5, and a
        detector that only looked above 0.5 would be blind to negatively
        oriented leaks (e.g. a payments-received column, where more payments
        means less default).
      - Degenerate cases (a feature with fewer than 2 distinct non-NaN values,
        or a target left single-class after the NaN drop) are skipped, not
        scored -- there is no meaningful AUC to report for them.

    Parameters
    ----------
    df : pd.DataFrame
        Frame containing the candidate feature columns and the target.
    features : sequence of str
        Column names to score.
    target : str
        Binary 0/1 target column name.
    categorical : sequence of str
        Which of `features` are categorical (target-encoded before ranking).

    Returns
    -------
    dict[str, float]
        Feature name -> standalone AUC in [0.5, 1.0], degenerate features
        skipped. Feed directly into flag_suspicious_auc().
    """
    categorical_set = set(categorical)
    y = df[target]

    aucs: dict[str, float] = {}
    for feature in features:
        if feature in categorical_set:
            score = df[feature].map(y.groupby(df[feature]).mean())
        else:
            score = pd.to_numeric(df[feature], errors="coerce")

        mask = score.notna() & y.notna()
        score_m, y_m = score[mask], y[mask]
        if score_m.nunique() < 2 or y_m.nunique() < 2:
            continue
        auc = float(roc_auc_score(y_m, score_m))
        aucs[feature] = max(auc, 1.0 - auc)
    return aucs


def check_single_feature_auc(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    categorical: Sequence[str] = (),
    threshold: float = 0.9,
) -> str:
    """
    Fail-closed gate: raise if any single feature is suspiciously predictive.

    Bundles single_feature_aucs() + flag_suspicious_auc() + raise into the
    same fail semantics as check_forbidden_features / prove_forbidden_absent,
    so a pipeline can wire "no single feature exceeds AUC 0.90" as a gate that
    stops the run, not a warning that scrolls by. Model-free by construction
    (only the data is needed), so it can run before any training compute is
    spent.

    Parameters
    ----------
    df, features, target, categorical :
        Forwarded to single_feature_aucs().
    threshold : float
        Standalone AUC above which a feature fails the gate. Default 0.9.

    Returns
    -------
    str
        Confirmation message when no feature exceeds the threshold.

    Raises
    ------
    ValueError
        Lists every flagged feature with its AUC (highest first), so all
        violations surface at once -- the same reporting discipline as
        check_forbidden_features.
    """
    aucs = single_feature_aucs(df, features, target, categorical=categorical)
    flagged = flag_suspicious_auc(aucs, threshold=threshold)
    if flagged:
        listing = ", ".join(
            f"{name} (AUC={auc:.4f})"
            for name, auc in sorted(flagged.items(), key=lambda kv: -kv[1])
        )
        raise ValueError(
            f"Suspiciously predictive standalone features detected: {listing}\n"
            f"A single feature with standalone AUC > {threshold} almost always "
            "means target leakage, not signal (legitimate risk signals rarely "
            "exceed ~0.75 standalone). Investigate before training -- see "
            "flag_suspicious_auc in leakage_check.py."
        )
    return (
        f"OK: no single feature exceeds standalone AUC {threshold} "
        f"({len(aucs)} features checked)."
    )