"""
SHAP reason codes for the LendingClub granting-model LightGBM model.

Fixes notebooks/analysis.ipynb's two SHAP cells in place as reusable, testable
code -- the global-importance cell and the local adverse-action cell -- the same
pattern calibrate.py applied to Cell 28.

Three things this module does that those cells did not:

  - It loads the SHIPPED artifact. The notebook retrains a separate no-state
    model at a fixed num_boost_round and scores it with no num_iteration
    argument at all, so none of its numbers are reproducible from what is on
    disk. This module loads models/lgbm_model.pkl through
    model_io.load_model_artifact() (fail-closed on the feature contract) and
    models/isotonic_calibrator.pkl through calibrate.load_calibrator() with
    model_artifact= passed (fail-closed on a stale calibrator binding). It
    never calls lgb.train. Since features.py's INCLUDE_ADDR_STATE is False,
    the shipped model already IS the no-state model the notebook retrained.

  - It DECLARES ITS SCALE. Every contribution returned here is in the model's
    raw log-odds margin, never in calibrated probability. The keys say so:
    `contribution_log_odds`, `base_value_log_odds`, `raw_margin_log_odds`,
    plus a top-level `scale` field pinned to CONTRIBUTION_SCALE.

  - It refuses to convert. docs/explainability.md Section 5 establishes that
    percentage-point attribution is UNDEFINED under the shipped isotonic
    calibrator: that calibrator is a 52-level step function, dp_cal/dp_raw is
    exactly 0 over 99.31% of the reject region, and p_cal = 0.25 is not even an
    attainable output. A local-slope chain rule multiplies by zero almost
    everywhere; black-box SHAP on calibrate(model(x)) explains a piecewise
    constant taking 21 distinct values across the whole reject region. Neither
    is a bug to be fixed here; both are consequences of the shipped artifact.

    There is consequently no `contribution_to_probability` key -- not present,
    not set to None. A null field is an invitation with a schema attached: it
    tells the next reader that a value belongs there and someone merely has not
    computed it yet. Nothing belongs there. tests/test_explain.py asserts the
    reason-code key set exactly, so the field cannot reappear by accident.

Rank order is what survives. Positive SHAP raises the raw margin, which raises
p_raw = sigmoid(margin), which -- isotonic being non-decreasing -- can never
lower p_cal. So the SIGN and the ORDER of contributions survive both transforms
even though their magnitudes do not. That asymmetry is the whole reason this
module ranks and declines to convert. ECOA / Regulation B adverse-action notices
ask for the principal reasons in rank order rather than numeric contributions;
that is context for why rank order is sufficient here, not a legal conclusion.
We are not lawyers.

Whether the shipped calibrator should be replaced is NOT decided by this module,
and nothing here depends on the answer (docs/explainability.md Section 8).

Reuses model_io.py's encoding path (_x / _to_lgb_frame / load_model_artifact) and
calibrate.py's calibration path (load_calibrator / apply_calibration) rather
than re-deriving either -- the same train/serve-skew rationale calibrate.py's
module docstring gives. Note _x, not _xy: _xy indexes engineered[TARGET] and so
raises KeyError on any unlabeled frame, which is every live serving request.
This is the first module in this repo whose input can be unlabeled.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.special import expit

from .calibrate import DEFAULT_MODEL_PATH, apply_calibration, load_calibrator
from .data_loader import RANDOM_SEED, load_raw, temporal_split
from .features import FEATURES
from .model_io import _to_lgb_frame, _x, load_model_artifact

# ---------------------------------------------------------------------------
# CONTRIBUTION_SCALE -- the axis every number in this module's output lives on.
#
# Exported as a constant, not spelled inline, so a caller can assert against it
# (`out["scale"] == explain.CONTRIBUTION_SCALE`) instead of string-matching a
# literal that could be silently reworded. TreeExplainer attributes the raw
# margin for objective="binary" boosters (model_io.py's LGB_PARAMS); the decision
# uses the isotonic-calibrated probability. Two transforms separate them --
# sigmoid, then isotonic. See docs/explainability.md Section 1.
# ---------------------------------------------------------------------------
CONTRIBUTION_SCALE = "log_odds_margin"

# ---------------------------------------------------------------------------
# DEFAULT_EXPLAIN_THRESHOLD -- fallback operating threshold for DIRECT calls
# (a notebook, a REPL -- anywhere with no pipeline context). It is an
# APPROXIMATION, and the approximation is not free.
#
# Threshold selection is owned by evaluate.py's select_threshold(), which scans
# DEFAULT_THRESHOLDS = np.arange(0.05, 0.95, 0.01) on Val. Floating-point
# arange does not produce the decimal you would write down: the element nearest
# 0.25 is 0.25000000000000006, and that -- not 0.25 -- is the value MLflow run
# cca4c361 logged as best_threshold (and as fairness_audit_threshold).
#
# The gap is one ULP wide and it changes an outcome. Rejection is
# `p_cal >= threshold`, so an applicant whose calibrated probability is exactly
# 0.25 is:
#     APPROVED under the real selected threshold (0.25 >= 0.25000000000000006
#              is False)
#     REJECTED under this constant                (0.25 >= 0.25 is True)
#
# Pipeline callers MUST pass the threshold select_threshold() actually returned,
# exactly as pipelines/training_flow.py already forwards best_threshold into
# fairness.py's audit_threshold. This constant exists only so a standalone call
# has a documented default, not so that anything can derive an operating point
# from it. Mirrors fairness.py's DEFAULT_AUDIT_THRESHOLD note.
#
# (In practice, on the shipped calibrator, p_cal == 0.25 exactly is
# unreachable: docs/explainability.md Section 4 shows 0.25 is not an attainable
# output of a 52-level step function -- the decision boundary falls inside a
# ramp of width 4.013e-06. The discrepancy is therefore real but currently
# unreachable through the shipped calibrator. It becomes reachable the moment
# the calibrator is replaced with a continuous one, which Section 8 leaves
# open. Documented now, while it is cheap.)
# ---------------------------------------------------------------------------
DEFAULT_EXPLAIN_THRESHOLD = 0.25

# Principal adverse factors listed per applicant. 4 is INHERITED from
# notebooks/analysis.ipynb, which slices `adverse[:4]`. The notebook gives no
# justification for 4 and neither does this module: it is not derived from the
# feature count, from any measured elbow in the contribution distribution, or
# from a legal requirement this project has verified. How many principal
# reasons an adverse-action notice must carry is a question for counsel. The
# number is exposed as a parameter precisely because it is not ours to fix.
DEFAULT_MAX_REASONS = 4

# Test rows sampled for global importance. Also INHERITED from the notebook,
# which draws 4000 rows with no stated rationale. It is not a convergence
# threshold anyone measured -- mean |SHAP| over 4000 of Test's 462,174 rows is
# simply what that cell did. Kept for continuity with the notebook's published
# importance ranking, and exposed as a parameter. Raise it if you need a
# stabler ranking; nothing here depends on the value being 4000.
SHAP_SAMPLE_N = 4000

# ---------------------------------------------------------------------------
# ADDITIVITY_ATOL -- how far base + sum(contributions) may drift from the raw
# margin before _assert_additivity() refuses to explain.
#
# Measured on the shipped 240-tree booster over 200 real Test rows: the worst
# absolute discrepancy is 7.77e-15, i.e. one float64 ULP at a margin magnitude
# of ~2. 1e-9 leaves six orders of headroom over that noise while still
# catching every failure this guard exists for -- a tree_limit mismatch between
# the explained trees and the scored trees, a corrupted contribution matrix, a
# multiclass booster whose last column is one class's base rather than the
# model's, or a change in shap's return contract. Those all land at O(0.1) or
# worse, not O(1e-15).
#
# For contrast, shap's own assert_additivity uses atol=1e-2, rtol=1e-2 (inside
# TreeExplainer.assert_additivity's nested check_sum) -- thirteen orders looser,
# and on the LightGBM path it never runs at all. See _shap_matrix.
# ---------------------------------------------------------------------------
ADDITIVITY_ATOL = 1e-9


def _assert_additivity(
    sv: np.ndarray,
    base: float,
    tree_limit: int | None,
    booster=None,
    X_lgb: pd.DataFrame | None = None,
    atol: float = ADDITIVITY_ATOL,
) -> None:
    """
    Prove base + sum(contributions) IS the margin the booster scores with --
    when there is a booster to prove it against.

    This is the check shap's `check_additivity=True` is widely believed to
    perform and does not (see _shap_matrix). It costs one extra
    `booster.predict(..., raw_score=True)` -- ~1.3 ms on the shipped booster
    for a single row -- and it runs on the serving path, not only under pytest.

    Fail-closed on purpose, the same discipline as load_model_artifact()'s
    feature contract and load_calibrator()'s trained_at binding: a violated
    additivity identity means the contributions being ranked into an
    adverse-action notice do not describe the margin the applicant was actually
    scored on. That is a wrong explanation, not a degraded one, and there is no
    safe way to serve it.

    Self-arming, like leakage_check's temporal sentinel
    --------------------------------------------------
    `booster` and `X_lgb` are optional. Additivity is `base + sum(sv) == the
    margin the booster scores`, so WITHOUT a booster (and the exact frame it
    scores) there is no margin to compare against and nothing to check. In that
    state this SKIPs -- explicitly, by returning -- exactly as
    check_temporal_consistency SKIPs when the cleaned dataset carries no date
    column to compare against issue_d (pipelines/training_flow.py). It arms the
    moment both are in hand.

    _shap_matrix ALWAYS passes both, so the loading path is always checked. The
    precomputed escape hatch loads no booster, so by default it passes neither
    and the values are trusted -- but a hatch caller that already holds the
    booster and the frame it scored (explain_applicants' `booster`/`X_lgb`
    parameters) arms this check for free, no new load. `booster` XOR `X_lgb`
    (one supplied, the other not) also skips: a half-supplied check is not a
    check, and silently half-checking is worse than a clean skip.

    `num_iteration=tree_limit` mirrors exactly what shap forwards to LightGBM
    (inside TreeExplainer.shap_values), so the margin compared against is the
    margin of the same trees that were explained.

    Raises
    ------
    ValueError
        If a booster and frame are in hand and any row's reconstruction drifts
        from the raw margin by more than `atol`.
    """
    if booster is None or X_lgb is None:
        return                          # not in hand -> nothing to check -> skip

    margin = np.asarray(
        booster.predict(X_lgb, num_iteration=tree_limit, raw_score=True), dtype=float,
    ).ravel()
    recon = float(base) + np.asarray(sv, dtype=float).sum(axis=1)

    diff = np.abs(recon - margin)
    if diff.size and diff.max() > atol:
        i = int(np.argmax(diff))
        raise ValueError(
            "Additivity check failed -- refusing to explain.\n"
            f"  row {i}: base_value + sum(contributions) = {recon[i]!r}\n"
            f"           booster raw-score margin        = {margin[i]!r}\n"
            f"  |difference| = {diff[i]:.6e}  (tolerance {atol:.1e}, "
            f"worst of {diff.size} row(s))\n"
            "base + sum(contributions) must BE the margin the booster scored "
            "with. It is not, so these contributions do not describe this "
            "applicant's score and must not be ranked into reason codes. "
            "Check that tree_limit matches the num_iteration used to score, "
            "and that shap's return contract has not changed -- see "
            "_shap_matrix."
        )


def _get_explainer(booster, explainer: shap.TreeExplainer | None = None) -> shap.TreeExplainer:
    """
    Return the TreeExplainer for `booster`, constructed fresh unless one is
    handed in.

    Constructed per call, deliberately, and this is a measured choice rather
    than a default. On the shipped 240-tree booster, shap 0.50.0:

        shap.TreeExplainer(booster)   ~65 ms
        explainer.shap_values(1 row)  ~2.6 ms

    so construction costs roughly 25 single-row explanations and an implicit
    module-level cache would look like an obvious win. It is not taken, for one
    reason: TreeExplainer is MUTABLE. `shap_values()` overwrites
    `explainer.expected_value` as a side effect of the call (see _shap_matrix).
    A cached explainer is therefore shared mutable state whose base value
    depends on whoever called it last, and it is not safe to hand to two
    concurrent callers. Trading that hazard for 65 ms, in a repository with no
    serving loop, is the wrong side of the trade.

    Batch callers amortize explicitly instead: build one explainer, pass it in.
    run_explanation() does exactly that -- one construction for the whole
    sample. If a serving path ever needs the cache, it should key on the model
    artifact's `trained_at`, so a retrained model cannot be explained by a
    stale explainer, the same binding discipline load_calibrator() enforces.
    """
    return explainer if explainer is not None else shap.TreeExplainer(booster)


def _shap_matrix(
    explainer: shap.TreeExplainer,
    X_lgb: pd.DataFrame,
    booster,
    tree_limit: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Call the explainer and normalize its return shapes to (contributions, base).

    Two shape hazards, both handled here ONCE so no other function in this
    module has to think about them:

    1. `shap_values()` returns an (n, k) ndarray for this booster under shap
       0.50.0, but has historically returned a 2-element list
       [class_0, class_1] for binary classifiers. Both are accepted; the
       positive class is taken from index 1 of a list.

    2. `explainer.expected_value` is MUTATED BY THE CALL. At construction it is
       a length-1 ndarray; after the first `shap_values()` it is a scalar
       float64; after a `shap_values(..., tree_limit=L)` it is the base value
       for L trees, not for all of them. It is therefore read AFTER the call
       here, never before. notebooks/analysis.ipynb reads it after too, but by
       accident of line order rather than by intent -- had it read first, its
       `expected_value[1]` would have raised IndexError on a length-1 array.

    `tree_limit` is forwarded so this module explains exactly the trees the
    model scores with (num_iteration=best_iteration), rather than every tree
    the booster happens to carry. On the shipped artifact those coincide
    (num_trees() == best_iteration == 240) and the results are bit-identical,
    but a booster saved without truncation would silently break additivity.
    Passing it explicitly removes the precondition instead of testing for it.

    shap's `check_additivity` VERIFIES NOTHING HERE, despite defaulting to
    True. Inside `TreeExplainer.shap_values`'s LightGBM fast path
    `model_output_vals` is left at None -- only the xgboost branch ever assigns
    it (both inside `TreeExplainer.shap_values`) -- so the guard

        if check_additivity and model_output_vals is not None:
            self.assert_additivity(out, model_output_vals)

    (also inside `TreeExplainer.shap_values`) is short-circuited on every call
    this repo makes. Demonstrated, not
    inferred: a frame with fico_n set to all-NaN passes
    `shap_values(..., check_additivity=True)` without raising. This docstring
    previously claimed shap performed the check "for free"; it never did.

    So the check is performed HERE instead, by _assert_additivity(), against
    `booster.predict(..., raw_score=True)`. That is the `booster` parameter's
    only purpose. It costs one extra predict (~1.3 ms for one row) and it runs
    wherever contributions are produced -- which is why the check lives at the
    point of PRODUCTION rather than in explain_applicants(): run_explanation()
    and global_importance() come through here too.

    explain_applicants()'s precomputed escape hatch bypasses this function
    entirely. On that path additivity is a PRECONDITION on the caller, enforced
    when a booster is available and trusted otherwise: _assert_additivity()
    self-arms if the hatch caller hands explain_applicants a booster and the
    frame it scored, and skips if not (see its docstring). The one in-repo hatch
    caller, run_explanation(), satisfies the precondition upstream -- it computes
    sv/base through THIS function first, where the guard runs over the whole
    sample, before slicing one row back into the hatch -- so it does not need to
    re-arm the check, and does not.

    Returns
    -------
    (np.ndarray of shape (n, len(FEATURES)), float)
        Contributions to the raw log-odds margin, and the base value on the
        same axis.

    Raises
    ------
    ValueError
        If the explainer returns a shape this function does not recognize --
        loudly, rather than silently reducing the wrong axis -- or if
        base + sum(contributions) is not the margin the booster scores with.
    """
    sv = explainer.shap_values(X_lgb, tree_limit=tree_limit)
    if isinstance(sv, list):
        sv = sv[1]                      # positive class
    sv = np.asarray(sv)

    base = explainer.expected_value      # read AFTER the call -- see above
    base = np.asarray(base).ravel()
    if base.size == 2:
        base = base[1]                   # positive class
    elif base.size == 1:
        base = base[0]
    else:
        raise ValueError(
            f"Unrecognized expected_value of size {base.size} from "
            f"{type(explainer).__name__}; expected a scalar or a 1/2-element "
            "array. shap's return contract has changed -- see _shap_matrix."
        )

    if sv.ndim != 2 or sv.shape != (len(X_lgb), len(FEATURES)):
        raise ValueError(
            f"Unrecognized shap_values shape {sv.shape}; expected "
            f"{(len(X_lgb), len(FEATURES))}. shap's return contract has "
            "changed -- see _shap_matrix."
        )

    _assert_additivity(sv, float(base), tree_limit, booster=booster, X_lgb=X_lgb)
    return sv, float(base)


def _rank_adverse(
    features: list[str],
    contribs_row: np.ndarray,
    values_row: pd.Series,
    max_reasons: int,
) -> list[dict]:
    """
    Rank-order the RISK-INCREASING factors for one applicant.

    Positive SHAP pushes the raw margin up, i.e. toward default. A
    risk-DECREASING factor is not a denial reason, so `contribution <= 0` is
    filtered out entirely rather than listed with a minus sign. A
    zero-contribution feature did nothing, and did not deny anyone. Survivors
    sort by contribution descending and truncate to max_reasons.

    Feature VALUES are read from the engineered frame (_x's output), not from
    the LightGBM frame (_to_lgb_frame's output). The two differ for a category
    the model never saw in training: _to_lgb_frame maps it to NaN by design,
    and "your application was denied because purpose = nan" is not a reason
    anyone can act on. The engineered frame still holds the string.

    Returns [] when no factor is risk-increasing. That is a real case, not a
    defensive branch -- see run_explanation()'s "The all-negative applicant".
    """
    adverse = [
        (f, float(s), v)
        for f, s, v in zip(features, contribs_row, values_row.values)
        if s > 0
    ]
    adverse.sort(key=lambda t: t[1], reverse=True)
    return [
        {
            "rank": i,
            "feature": f,
            "value": str(v),
            "contribution_log_odds": s,
        }
        for i, (f, s, v) in enumerate(adverse[:max_reasons], start=1)
    ]


def explain_applicants(
    df: pd.DataFrame,
    model_path: str | Path | None = None,
    calibrator_path: str | Path | None = None,
    threshold: float = DEFAULT_EXPLAIN_THRESHOLD,
    max_reasons: int = DEFAULT_MAX_REASONS,
    explainer: shap.TreeExplainer | None = None,
    shap_values: np.ndarray | None = None,
    base_value: float | None = None,
    p_cal: np.ndarray | None = None,
    booster=None,
    X_lgb: pd.DataFrame | None = None,
    tree_limit: int | None = None,
) -> list[dict]:
    """
    Explain one row per applicant, on the raw log-odds margin.

    `df` carries the RAW schema (emp_length, purpose, ...) -- not the
    engineered feature frame, and NOT the target. This function never reads
    features.TARGET, so a live serving request with no known outcome flows
    through the identical code path a Test row does. tests/test_explain.py
    proves that structurally, by passing a frame with no target column at all.

    A single applicant is `df.iloc[[i]]` -- a one-row DataFrame, double
    brackets. A Series will not work: add_features() calls .map()/.eq() on
    columns and _to_lgb_frame() assigns pd.Categorical into columns.

    Parameters
    ----------
    df : pd.DataFrame
        One row per applicant, raw schema. A category value the model never saw
        in training degrades to NaN rather than raising (see _to_lgb_frame), so
        a novel `purpose` is scored, not rejected. Its SHAP contribution is
        finite and additivity still holds; only the reported `value` keeps the
        original string (see _rank_adverse).
    model_path, calibrator_path : str, Path, or None
        Defaults to DEFAULT_MODEL_PATH and model_path.parent /
        "isotonic_calibrator.pkl". Ignored entirely when the escape hatch below
        is used.
    threshold : float
        Rejection cutoff on the CALIBRATED probability: rejected iff
        `p_calibrated >= threshold`, matching evaluate.py's
        `approved = p < threshold` and fairness.py's audit. Not a cutoff on the
        margin, and not a cutoff on p_raw. Read DEFAULT_EXPLAIN_THRESHOLD's
        comment before passing the literal 0.25.
    max_reasons : int
        Cap on principal adverse factors per applicant. See DEFAULT_MAX_REASONS
        on why 4 is inherited rather than derived.
    explainer : shap.TreeExplainer or None
        Pass one to amortize construction across many calls. See _get_explainer.
    shap_values, base_value, p_cal : np.ndarray / float / np.ndarray or None
        Precomputed. If ALL THREE are supplied, this skips loading the joblib
        model artifact, the calibrator, and shap entirely -- the same escape
        hatch select_threshold()'s `p_val` provides, and how tests exercise
        this without a real artifact pair on disk. run_explanation() uses it to
        score the sample once and explain it once.
    booster, X_lgb, tree_limit : Booster / pd.DataFrame / int or None
        The escape hatch's additivity self-check, enforced-when-possible. On
        the hatch path the precomputed values are trusted (the hatch loads no
        booster, so nothing here can re-derive the margin). A caller that
        ALREADY holds the booster and the exact LightGBM frame it scored may
        pass them -- together with the tree_limit used -- and this proves
        base_value + sum(shap_values) IS that booster's raw margin before
        ranking any reason code, raising if not. All three default to None, in
        which case the check skips and behaviour is exactly as before: no new
        load, no cost. Every in-repo caller uses the default; run_explanation()
        verifies upstream (via _shap_matrix) and deliberately does not re-arm
        this. Ignored entirely on the loading path, where _shap_matrix already
        ran the identical check. See _assert_additivity's "Self-arming" note.

    Returns
    -------
    list[dict]
        One dict per row of `df`:
        {"scale": CONTRIBUTION_SCALE,
         "p_raw": float,                    # sigmoid(margin); calibrator INPUT
         "p_calibrated": float,             # the decided quantity
         "threshold": float,
         "decision": "REJECT" | "APPROVE",
         "base_value_log_odds": float,
         "raw_margin_log_odds": float,      # == base_value + sum(contributions)
         "contributions_log_odds": {feature: float, ...},
         "reason_codes": [{"rank", "feature", "value",
                           "contribution_log_odds"}, ...],
         "model_trained_at": str | None,
         "calibrator_trained_at": str | None}

    Raises
    ------
    ValueError
        If some but not all of (shap_values, base_value, p_cal) are supplied. A
        half-populated escape hatch would load an artifact and then throw half
        of it away, which is a silent way to explain one model and decide with
        another.
    """
    hatch = (shap_values is not None, base_value is not None, p_cal is not None)
    if any(hatch) and not all(hatch):
        raise ValueError(
            "The precomputed escape hatch is all-or-nothing: supply "
            "shap_values, base_value AND p_cal together, or none of them. "
            f"Got shap_values={hatch[0]}, base_value={hatch[1]}, p_cal={hatch[2]}."
        )

    # base_value is `float | None` in the signature and is never None past this
    # point, by either path: the hatch supplied it (the raise above is exactly
    # what makes "some but not all" unreachable), or _shap_matrix assigns it
    # below. mypy can follow neither -- `all(hatch)` is a bool over a tuple, not
    # a narrowing -- so the three float(base_value) calls downstream carry
    # `type: ignore[arg-type]` and point back here. The ValueError IS the guard.
    # An assert would be a weaker second copy of it, and a cast would state the
    # same conclusion while checking nothing.
    X = _x(df)                                  # label-free; never reads TARGET
    model_trained_at = calibrator_trained_at = None

    if not all(hatch):
        model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        calibrator_path = (
            Path(calibrator_path)
            if calibrator_path is not None
            else model_path.parent / "isotonic_calibrator.pkl"
        )
        artifact = load_model_artifact(model_path)   # fail-closed: feature contract
        booster = artifact["model"]
        best_iteration = artifact["best_iteration"]
        # Pass the artifact so a calibrator fit against a different model
        # instance is rejected rather than silently applied.
        iso = load_calibrator(calibrator_path, model_artifact=artifact)

        X_lgb = _to_lgb_frame(X, artifact["category_maps"])
        shap_values, base_value = _shap_matrix(
            _get_explainer(booster, explainer), X_lgb, booster,
            tree_limit=best_iteration,
        )
        p_raw_vec = booster.predict(X_lgb, num_iteration=best_iteration)
        p_cal = apply_calibration(iso, p_raw_vec)

        model_trained_at = artifact.get("trained_at")
        calibrator_trained_at = _calibrator_trained_at(calibrator_path)

    shap_values = np.asarray(shap_values)
    p_cal = np.asarray(p_cal, dtype=float)

    # Escape-hatch additivity self-check, enforced-when-possible. Hatch path
    # only -- the loading path already ran the identical check inside
    # _shap_matrix. This arms iff the caller put the booster and the exact frame
    # it scored in hand (the booster/X_lgb/tree_limit parameters), then proves
    # base + sum(sv) IS that booster's raw margin before any reason code is
    # ranked. When they are None -- every in-repo caller, today -- it skips, so
    # this is a no-op and nothing changes. It loads nothing: the booster and
    # frame are the caller's, already in memory. See _assert_additivity.
    if all(hatch):
        _assert_additivity(
            shap_values, float(base_value), tree_limit,  # type: ignore[arg-type]
            booster=booster, X_lgb=X_lgb,
        )

    # The margin is reconstructed from the attribution rather than re-predicted:
    # base + sum(contributions) IS the raw margin, because tree_path_dependent
    # SHAP is exactly additive. That identity is the one shap's
    # check_additivity=True is widely believed to guarantee and does not -- it
    # is inert on the LightGBM branch (model_output_vals stays None inside
    # TreeExplainer.shap_values; see _shap_matrix). What
    # actually enforces it depends on which path reached this line:
    #   - loading path: _shap_matrix just proved it, via _assert_additivity,
    #     against booster.predict(raw_score=True).
    #   - escape-hatch path: base_value and shap_values arrived precomputed.
    #     Additivity is a PRECONDITION on the caller here -- enforced by the
    #     self-check just above WHEN the caller handed over a booster and frame,
    #     trusted otherwise. The one in-repo caller, run_explanation(), verifies
    #     upstream instead: it computes sv/base through _shap_matrix, where the
    #     guard runs over the whole sample, before slicing one row back in here,
    #     so it passes no booster and this stays a no-op.
    #
    # Deriving p_raw as sigmoid(margin) is exact regardless of path: objective=
    # "binary" means Booster.predict() is precisely sigmoid of this quantity,
    # and tests/test_explain.py pins that identity to 0.0 absolute error.
    margins = float(base_value) + shap_values.sum(axis=1)  # type: ignore[arg-type]
    p_raw = expit(margins)

    out = []
    for i in range(len(X)):
        out.append({
            "scale": CONTRIBUTION_SCALE,
            "p_raw": float(p_raw[i]),
            "p_calibrated": float(p_cal[i]),
            "threshold": float(threshold),
            "decision": "REJECT" if p_cal[i] >= threshold else "APPROVE",
            "base_value_log_odds": float(base_value),  # type: ignore[arg-type]
            "raw_margin_log_odds": float(margins[i]),
            "contributions_log_odds": {
                f: float(s) for f, s in zip(FEATURES, shap_values[i])
            },
            "reason_codes": _rank_adverse(
                FEATURES, shap_values[i], X.iloc[i], max_reasons,
            ),
            "model_trained_at": model_trained_at,
            "calibrator_trained_at": calibrator_trained_at,
        })
    return out


def _calibrator_trained_at(calibrator_path: str | Path) -> str | None:
    """
    The calibrator artifact's own `trained_at`, for provenance.

    load_calibrator() returns the bare IsotonicRegression and drops the
    surrounding dict, so this re-reads it. An explanation is a claim about a
    model-CALIBRATOR PAIR: the margin comes from one, the decision from the
    other. An artifact that reproduces an explanation needs to identify both.
    """
    return joblib.load(calibrator_path).get("trained_at")


def global_importance(
    df: pd.DataFrame,
    model_path: str | Path | None = None,
    explainer: shap.TreeExplainer | None = None,
    shap_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Mean |SHAP| per feature over `df`. Mirrors the notebook's global cell.

    Mean ABSOLUTE contribution, so a feature that pushes hard in both
    directions across applicants ranks high instead of cancelling to zero. This
    is a magnitude ranking and not a direction: read direction from a
    particular applicant's signed contributions, never from this table.

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_abs_shap -- in CONTRIBUTION_SCALE units, as the
        column name says. Sorted descending.
    """
    if shap_values is None:
        model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        artifact = load_model_artifact(model_path)
        X_lgb = _to_lgb_frame(_x(df), artifact["category_maps"])
        shap_values, _base = _shap_matrix(
            _get_explainer(artifact["model"], explainer), X_lgb, artifact["model"],
            tree_limit=artifact["best_iteration"],
        )

    mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
    return (
        pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def run_explanation(
    model_path: str | Path | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
    calibrator_path: str | Path | None = None,
    threshold: float = DEFAULT_EXPLAIN_THRESHOLD,
    max_reasons: int = DEFAULT_MAX_REASONS,
    sample_n: int = SHAP_SAMPLE_N,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Global importance over a sampled Test slice, plus reason codes for the
    first rejected applicant in that sample. Mirrors both notebook SHAP cells
    end to end, against the SHIPPED model and calibrator.

    On the real dataset (shipped model + calibrator, production feature set per
    features.py's INCLUDE_ADDR_STATE = False, sample_n=4000, seed=42,
    threshold=0.25, model trained_at 2026-07-09T07:00:51Z), this reproduces:

        base value (log-odds):  -1.699279

        Global feature impact (mean |SHAP|, log-odds):
            fico_n              0.311524
            loan_amnt           0.295318
            revenue             0.218866
            dti_n               0.152459
            purpose             0.121781
            home_ownership_n    0.075739
            emp_length_ord      0.037181
            emp_length_missing  0.004130

        Rejected in sample (p_cal >= 0.25):  852 / 4,000  (21.3%)

        First rejected applicant: p_raw=0.307733  p_cal=0.346782
            1. loan_amnt   value=35000.0             SHAP=+0.634603
            2. dti_n       value=37.16               SHAP=+0.553845
            3. fico_n      value=677.0               SHAP=+0.294308
            4. purpose     value=debt_consolidation  SHAP=+0.050214

    Two things worth reading off that table. The sample's 21.3% rejection rate
    is consistent with run_evaluation()'s ~78% approval rate on all of Test.
    And emp_length_missing, the explicit not-disclosed flag features.py keeps
    rather than imputing away, carries the smallest mean |SHAP| of any feature
    by a factor of ~9 -- the model does use it, but barely. That is a fact
    about the model, not a defect in this module.

    The all-negative applicant
    --------------------------
    An applicant can be rejected with every contribution <= 0. The margin is
    base_value + sum(contributions); if base_value alone lands above the
    rejection boundary, every feature can push DOWN and the applicant is still
    rejected. `reason_codes` is then []. This module reports that honestly and
    does not manufacture a reason.

    It is sharper than it first looks. There is no margin corresponding to
    threshold=0.25, because p_cal = 0.25 is not an attainable output of the
    shipped 52-level calibrator -- the boundary falls inside a ramp of width
    4.013e-06 (docs/explainability.md Section 4). So "base_value alone exceeds
    the boundary" cannot be stated as a clean inequality on the margin axis
    without first choosing which end of that ramp is the boundary. What an
    adverse-action notice should say in this case is not decided here.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame] or None
        temporal_split() output; must contain "test". If None, this calls
        load_raw() + temporal_split() itself.
    sample_n, seed : int
        Rows drawn from Test for the global ranking, and the rng seed. See
        SHAP_SAMPLE_N on why 4000 is inherited, not derived.

    Returns
    -------
    dict
        {"scale": CONTRIBUTION_SCALE,
         "threshold": float,
         "n_sampled": int,
         "base_value_log_odds": float,
         "global_importance": pd.DataFrame,
         "n_rejected": int,
         "example": dict | None,        # explain_applicants() entry, or None
                                        # if no applicant in the sample was
                                        # rejected at this threshold
         "model_trained_at": str | None,
         "calibrator_trained_at": str | None}
    """
    if splits is None:
        splits = temporal_split(load_raw())

    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    calibrator_path = (
        Path(calibrator_path)
        if calibrator_path is not None
        else model_path.parent / "isotonic_calibrator.pkl"
    )

    test = splits["test"]
    rng = np.random.default_rng(seed)
    sample_n = min(sample_n, len(test))
    idx = rng.choice(len(test), size=sample_n, replace=False)
    sample = test.iloc[idx]

    artifact = load_model_artifact(model_path)
    booster = artifact["model"]
    best_iteration = artifact["best_iteration"]
    iso = load_calibrator(calibrator_path, model_artifact=artifact)
    calibrator_trained_at = _calibrator_trained_at(calibrator_path)

    # ONE explainer for the whole sample -- the amortization _get_explainer's
    # docstring describes, done explicitly rather than by a hidden cache.
    explainer = _get_explainer(booster)
    X_lgb = _to_lgb_frame(_x(sample), artifact["category_maps"])
    sv, base = _shap_matrix(explainer, X_lgb, booster, tree_limit=best_iteration)
    p_cal = apply_calibration(iso, booster.predict(X_lgb, num_iteration=best_iteration))

    imp = global_importance(sample, shap_values=sv)

    rejected = np.flatnonzero(p_cal >= threshold)
    example = None
    if rejected.size:
        who = int(rejected[0])
        example = explain_applicants(
            sample.iloc[[who]],
            threshold=threshold,
            max_reasons=max_reasons,
            shap_values=sv[[who]],
            base_value=base,
            p_cal=p_cal[[who]],
        )[0]
        # The escape hatch skips artifact loading, so provenance is filled in
        # from the artifacts this function already opened.
        example["model_trained_at"] = artifact.get("trained_at")
        example["calibrator_trained_at"] = calibrator_trained_at

    print(f"Scale: {CONTRIBUTION_SCALE}  (NOT calibrated probability)")
    print(f"Sampled {sample_n:,} of {len(test):,} test rows | "
          f"base value = {base:.6f}\n")
    print("=== Global feature impact (mean |SHAP|, log-odds) ===")
    print(imp.to_string(index=False))
    print(f"\nRejected in sample (p_cal >= {threshold}): "
          f"{rejected.size:,} / {sample_n:,} ({rejected.size / sample_n:.1%})")

    if example is None:
        print("No rejected applicant in sample at this threshold.")
    else:
        print("\n=== Principal adverse factors, first rejected applicant ===")
        print(f"p_raw={example['p_raw']:.6f}  p_cal={example['p_calibrated']:.6f}  "
              f"decision={example['decision']}")
        if not example["reason_codes"]:
            print("  (no risk-increasing factor -- see 'The all-negative applicant')")
        for rc in example["reason_codes"]:
            print(f"  {rc['rank']}. {rc['feature']:<18} value={rc['value']:<12} "
                  f"SHAP={rc['contribution_log_odds']:+.6f}")

    return {
        "scale": CONTRIBUTION_SCALE,
        "threshold": float(threshold),
        "n_sampled": int(sample_n),
        "base_value_log_odds": float(base),
        "global_importance": imp,
        "n_rejected": int(rejected.size),
        "example": example,
        "model_trained_at": artifact.get("trained_at"),
        "calibrator_trained_at": calibrator_trained_at,
    }
