"""
Fairness audit for the LendingClub granting-model LightGBM model.

Fixes notebooks/analysis.ipynb Section 9 in place as reusable, testable
code: a three-layer geographic-proxy risk audit of addr_state.

IMPORTANT -- legal framing, kept deliberately narrow throughout this module:
addr_state is NOT a protected class under ECOA (which covers race, sex, age,
national origin, and a handful of other explicitly enumerated categories).
What this module audits is a GEOGRAPHIC-PROXY RISK: a state label can
correlate with socioeconomic and demographic patterns the model never sees
directly, so a model that leans on addr_state as a risk shortcut can end up
disproportionately rejecting creditworthy applicants from a given state --
"digital redlining" in industry usage, not a disparate-impact finding under
ECOA/Reg B or any other statute. Every function and printed report in this
module says "geographic-proxy risk" / "digital redlining risk", never
"discrimination" or "disparate impact" -- those are legal conclusions this
module is not equipped to make, and the notebook it fixes in place is
careful to keep the same distinction. Preserve that wording in anything
that consumes this module's output.

Three layers, each existing because the previous one alone is not enough
(see notebooks/analysis.ipynb Section 9 for the full narrative):

  Layer 1 (audit_layer1)          Bootstrap 95% CIs on each state's Equal
                                  Opportunity ratio at the operating
                                  threshold. Flags a state only if the
                                  ENTIRE CI falls below 0.80 -- a point
                                  estimate alone is noise on a finite sample.
  Layer 2 (audit_layer2)          Threshold sweep: does Layer 1's "all
                                  clear" survive tightening the threshold,
                                  or was it an artifact of a permissive
                                  cutoff that washes out state-level
                                  differences?
  Layer 3 (audit_layer3_ablation) Retrain without addr_state. If a flagged
                                  state's EO ratio recovers, the model was
                                  using the state label as a shortcut
                                  (geographic-proxy risk); if it doesn't,
                                  the disparity reflects real economic
                                  differences the label was only proxying
                                  for, and removing the feature won't help.

Equal Opportunity here means: among applicants who actually paid off their
loans (y_true == 0, verified good borrowers), what fraction did the model
approve? A state's good-applicant approval rate divided by the national
good-applicant approval rate is that state's EO ratio.

Reuses train.py's _xy/_to_lgb_frame for Layer 1/2's encoding path (auditing
whatever model is CURRENTLY shipped, via features.py's live
INCLUDE_ADDR_STATE setting) -- the same train/serve-skew rationale as
calibrate.py: a second, independently written encoding path here would
drift from what the shipped model was actually trained on.

Layer 3 is different on purpose: it must always be able to compare a
with-state model against a no-state model, regardless of which one is
currently shipped as production (features.py's INCLUDE_ADDR_STATE toggle --
see docs/data-decisions.md, "Execute the fairness conclusion: remove
addr_state from the production model" -- now defaults to OFF, i.e.
production ships the NO-state model). So Layer 3
does not read features.py's toggle-dependent FEATURES/CATEGORICAL at all,
and does not reuse any pre-loaded model artifact as one side of the
comparison -- it trains BOTH variants itself from explicit,
toggle-independent feature lists defined in this module (see
FEATURES_WITH_STATE/FEATURES_NOSTATE below and _train_ablation_variant's
docstring). That is what lets this audit be re-run and keep meaning the same
thing no matter which variant happens to be in production on a given day.

On the real dataset, this reproduces the notebook's headline finding: at
threshold 0.22, Mississippi's Equal Opportunity ratio is ~0.734 with
addr_state in the model, and recovers to ~0.990 once addr_state is removed
-- the state label was a shortcut, not a reflection of MS applicants'
actual financials. The AUC cost of removing addr_state is small (~-0.0035).
See run_fairness_audit()'s docstring for the full numbers.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .calibrate import DEFAULT_MODEL_PATH, apply_calibration
from .data_loader import RANDOM_SEED, load_raw, temporal_split
from .features import NUMERIC, TARGET, add_features, build_categorical
from .train import LGB_PARAMS, _to_lgb_frame, _xy, load_model_artifact

# ---------------------------------------------------------------------------
# Constants -- traced to notebooks/analysis.ipynb Section 9, not guesses.
# ---------------------------------------------------------------------------
EO_THRESHOLD = 0.80   # regulatory "80% rule" benchmark for the EO ratio
MIN_N = 500           # states with fewer good-applicant rows are skipped (Layer 1)
N_BOOT = 2000         # bootstrap resamples per state (Layer 1)

# Fallback operating threshold for DIRECT calls to run_fairness_audit()
# (a notebook, a REPL -- anywhere with no pipeline context). Threshold
# selection is owned by src/evaluate.py's select_threshold(), scanned on Val
# under the stylized cost model (LGD=0.65, interest margin=0.12), and
# pipelines/training_flow.py passes THAT best_threshold in as
# audit_threshold -- so a pipeline run always audits Layer 1 at the model's
# real operating point, never at this constant. 0.26 is kept only as the
# documented notebook-era default for standalone use, not something this
# module derives itself.
DEFAULT_AUDIT_THRESHOLD = 0.26

SWEEP_THRESHOLDS = [0.12, 0.15, 0.18, 0.20, 0.22, 0.26, 0.30]
# Layer 3 deliberately audits at 0.22, NOT at the model's operating
# threshold: Section 9.2's sweep found the state disparity most visible
# there, and an ablation wants its comparison where the signal is strongest
# -- see audit_layer3_ablation()'s docstring. Do not "align" this with
# evaluate.py's best_threshold; the divergence is the point.
ABLATION_THRESHOLD = 0.22
WATCH_STATES = ["MS", "AL", "TN", "NV", "NE", "FL"]

# Layer 3's two feature sets -- built from NUMERIC and build_categorical()
# with an EXPLICIT True/False, not from features.py's toggle-dependent
# CATEGORICAL/FEATURES. This is what makes Layer 3 self-contained: it always
# knows how to build "the with-state model" and "the no-state model", no
# matter which one INCLUDE_ADDR_STATE currently points production at.
CATEGORICAL_WITH_STATE = build_categorical(True)
CATEGORICAL_NOSTATE = build_categorical(False)
FEATURES_WITH_STATE = NUMERIC + CATEGORICAL_WITH_STATE
FEATURES_NOSTATE = NUMERIC + CATEGORICAL_NOSTATE


def bootstrap_eo_ci(
    state_approved: np.ndarray,
    base_rate: float,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """
    95% bootstrap CI for one state's Equal Opportunity ratio.

    Parameters
    ----------
    state_approved : array-like of 0/1
        Per-applicant approval indicator, restricted to that state's GOOD
        applicants (y_true == 0) only.
    base_rate : float
        National good-applicant approval rate, i.e. the denominator of the
        EO ratio.
    n_boot : int
        Number of bootstrap resamples. Default 2000, matching the notebook.
    seed : int
        Seeds this call's own np.random.default_rng. The notebook instead
        draws every state's bootstrap sample from ONE continuing
        np.random.default_rng(RANDOM_SEED) shared across its groupby loop,
        so consecutive states draw from different positions in the same
        stream. This function reseeds fresh on every call instead, so it is
        independently callable and unit-testable outside of audit_layer1's
        loop -- that changes which exact random numbers a given state
        draws, but not the CI's statistical properties (2,000 resamples,
        percentile method, same per-state sample size and base rate either
        way).

    Returns
    -------
    (float, float)
        (2.5th percentile, 97.5th percentile) of the bootstrap distribution
        of (resampled approval rate / base_rate).
    """
    rng = np.random.default_rng(seed)
    state_approved = np.asarray(state_approved)
    n = len(state_approved)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_rates = state_approved[idx].mean(axis=1)
    boot_ratio = boot_rates / base_rate
    lo, hi = np.percentile(boot_ratio, [2.5, 97.5])
    return float(lo), float(hi)


def audit_layer1(
    fair_df: pd.DataFrame,
    threshold: float,
    min_n: int = MIN_N,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Layer 1: per-state EO ratio + bootstrap 95% CI at a single operating
    threshold. Mirrors notebooks/analysis.ipynb Cell 35.

    A state is only flagged "confirmed" if its ENTIRE CI sits below
    EO_THRESHOLD (0.80) -- a point estimate below 0.80 on a small sample is
    noise, not evidence, which is exactly why this bootstraps instead of
    trusting fair_df.groupby("addr_state")["approved"].mean() directly.

    Parameters
    ----------
    fair_df : pd.DataFrame
        Must have columns "addr_state", "y_true" (0/1, 1=defaulted), and
        "p" (calibrated predicted probability of default), one row per Test
        applicant.
    threshold : float
        Approval cutoff: approved = (p < threshold).
    min_n : int
        States with fewer than this many GOOD (y_true == 0) applicants are
        skipped entirely -- their sample is too small to trust even a
        bootstrap CI. Default 500, matching the notebook.
    n_boot, seed : passed through to bootstrap_eo_ci for every state.

    Returns
    -------
    pd.DataFrame
        Columns: state, n_good, eo_ratio, ci_low, ci_high, verdict.
        verdict is one of "confirmed (CI fully < 0.80)",
        "inconclusive (CI straddles 0.80)", or "clear". Sorted by eo_ratio
        ascending (worst-looking state first), matching the notebook.
    """
    fair_df = fair_df.copy()
    fair_df["approved"] = (fair_df["p"] < threshold).astype(int)

    good = fair_df[fair_df["y_true"] == 0]
    base_good_appr = good["approved"].mean()

    records = []
    for st, g in good.groupby("addr_state"):
        n_good = len(g)
        if n_good < min_n:
            continue
        appr = g["approved"].values
        eo_ratio = appr.mean() / base_good_appr
        lo, hi = bootstrap_eo_ci(appr, base_good_appr, n_boot=n_boot, seed=seed)
        if hi < EO_THRESHOLD:
            verdict = "confirmed (CI fully < 0.80)"
        elif lo < EO_THRESHOLD <= hi:
            verdict = "inconclusive (CI straddles 0.80)"
        else:
            verdict = "clear"
        records.append({
            "state": st,
            "n_good": n_good,
            "eo_ratio": eo_ratio,
            "ci_low": lo,
            "ci_high": hi,
            "verdict": verdict,
        })

    return (
        pd.DataFrame(records, columns=["state", "n_good", "eo_ratio", "ci_low", "ci_high", "verdict"])
        .sort_values("eo_ratio")
        .reset_index(drop=True)
    )


def audit_layer2(
    fair_df: pd.DataFrame,
    thresholds: list[float] = SWEEP_THRESHOLDS,
    watch_states: list[str] = WATCH_STATES,
) -> pd.DataFrame:
    """
    Layer 2: threshold sweep -- does Layer 1's "all clear" survive
    tightening the threshold? Mirrors notebooks/analysis.ipynb Cell 38.

    A permissive threshold approves nearly everyone, which washes out
    state-level differences and can make Layer 1 report "all clear" even
    when a disparity exists at more conservative cutoffs a lender might
    actually choose to run. Sweeping thresholds is what tells a *stable*
    finding apart from a threshold-specific artifact.

    Parameters
    ----------
    fair_df : pd.DataFrame
        Same shape as audit_layer1's: addr_state, y_true, p.
    thresholds : list[float]
        Cutoffs to sweep. Default matches the notebook's
        [0.12, 0.15, 0.18, 0.20, 0.22, 0.26, 0.30].
    watch_states : list[str]
        States to report per threshold (not every state -- the notebook
        deliberately narrows to a handful flagged as interesting by prior
        EDA / Layer 1, to keep the sweep table readable).

    Returns
    -------
    pd.DataFrame
        One row per threshold; columns: threshold, national_good_approval_
        rate, then one column per watch_state holding that state's EO ratio
        at that threshold.
    """
    good_mask = fair_df["y_true"] == 0
    rows = []
    for t in thresholds:
        appr = (fair_df["p"] < t).astype(int)
        base = appr[good_mask].mean()
        row = {"threshold": t, "national_good_approval_rate": base}
        for st in watch_states:
            m = good_mask & (fair_df["addr_state"] == st)
            row[st] = appr[m].mean() / base
        rows.append(row)
    return pd.DataFrame(rows)


def _xy_explicit(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Like train.py's _xy(), but selects an EXPLICIT feature list instead of
    the module-level FEATURES from features.py. add_features() itself is
    unchanged and still reused (this is not a second feature-engineering
    path) -- only the column selection AFTER it differs, because Layer 3
    must be able to build the with-state feature frame even when
    production's INCLUDE_ADDR_STATE toggle is False (i.e. addr_state is not
    in features.py's FEATURES right now).
    """
    engineered = add_features(df)
    return engineered[features], engineered[TARGET]


def _to_lgb_frame_explicit(
    X: pd.DataFrame, category_maps: dict[str, pd.Index], categorical_cols: list[str]
) -> pd.DataFrame:
    """
    Like train.py's _to_lgb_frame(), but casts an EXPLICIT categorical
    column list instead of the module-level CATEGORICAL from features.py.
    train.py's version cannot be reused as-is here: it iterates over
    CATEGORICAL, which reflects whichever ONE variant (with-state or
    no-state) happens to be configured as production right now, while
    Layer 3 needs to encode BOTH variants in the same call to
    audit_layer3_ablation. Same underlying logic either way (cast to
    train-derived categories, unseen -> NaN) -- just parameterized on which
    columns are categorical for this particular variant.
    """
    X = X.copy()
    for c in categorical_cols:
        X[c] = pd.Categorical(X[c], categories=category_maps[c])
    return X


def _train_ablation_variant(
    splits: dict[str, pd.DataFrame],
    features: list[str],
    categorical_cols: list[str],
    num_boost_round: int,
) -> tuple[np.ndarray, pd.Series]:
    """
    Train + calibrate ONE Layer-3 variant (with-state or no-state) from
    scratch: fixed-round lgb.train on Train (no early stopping / valid_sets
    -- see audit_layer3_ablation's docstring for why), isotonic calibration
    fit on Calib, applied to Test. Mirrors notebook Cell 41's per-variant
    training exactly, just factored out so both variants share one
    implementation instead of two hand-duplicated copies.

    Returns
    -------
    (np.ndarray, pd.Series)
        Calibrated Test predicted probabilities, and Test's actual target
        (y_test) -- identical values whichever variant this is called for,
        since both read the same splits["test"], just carried alongside the
        predictions for convenience at the call site.
    """
    X_train, y_train = _xy_explicit(splits["train"], features)
    X_calib, y_calib = _xy_explicit(splits["calib"], features)
    X_test, y_test = _xy_explicit(splits["test"], features)

    category_maps = {c: pd.Categorical(X_train[c]).categories for c in categorical_cols}
    X_train_lgb = _to_lgb_frame_explicit(X_train, category_maps, categorical_cols)
    X_calib_lgb = _to_lgb_frame_explicit(X_calib, category_maps, categorical_cols)
    X_test_lgb = _to_lgb_frame_explicit(X_test, category_maps, categorical_cols)

    dtrain = lgb.Dataset(X_train_lgb, y_train, categorical_feature=categorical_cols)
    model = lgb.train(dict(LGB_PARAMS), dtrain, num_boost_round=num_boost_round)

    p_calib_raw = model.predict(X_calib_lgb)
    p_test_raw = model.predict(X_test_lgb)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_calib_raw, y_calib)
    p_test_cal = apply_calibration(iso, p_test_raw)

    return p_test_cal, y_test


def audit_layer3_ablation(
    splits: dict[str, pd.DataFrame],
    num_boost_round: int,
    threshold: float = ABLATION_THRESHOLD,
    watch_states: list[str] = WATCH_STATES,
) -> dict:
    """
    Layer 3: train a WITH-state model and a NO-state model, both fresh, and
    compare their EO ratios. Mirrors notebooks/analysis.ipynb Cell 41.

    Distinguishes the two explanations Layer 2 cannot: is a persistently
    low EO ratio because the model uses the state label as a risk shortcut
    (geographic-proxy risk -- removing the label should fix it), or because
    that state's applicants genuinely have weaker financials that the state
    label was only proxying for (removing the label should NOT fix it)?

    Self-contained by design: this function does not read features.py's
    module-level FEATURES/CATEGORICAL (which reflect whatever
    INCLUDE_ADDR_STATE happens to be configured to in production right now)
    and does not accept a pre-loaded model artifact for either side of the
    comparison. It always trains BOTH variants itself, from this module's
    own FEATURES_WITH_STATE/FEATURES_NOSTATE constants -- that is what lets
    Layer 3 answer "what does adding/removing addr_state do" correctly no
    matter which one is currently shipped as production.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        temporal_split() output; must contain "train", "calib", "test".
        (Does not need "val" -- see num_boost_round below.)
    num_boost_round : int
        Fixed boosting-round count used for BOTH variants. No early
        stopping / valid_sets here, by design: this mirrors notebook
        Cell 41's lgb.train(params, dtr_ns, num_boost_round=best_rounds)
        exactly -- using the SAME round count for both variants keeps this
        a controlled ablation (one feature toggled, everything else held
        fixed) rather than two independent model-selection exercises that
        could confound the AUC/EO comparison. Typically sourced from
        whichever model is currently in production's own best_iteration
        (see run_fairness_audit) -- a fixed, already-validated round
        budget, not something re-derived per variant here.
    threshold : float
        Operating threshold for the EO comparison. Default 0.22, matching
        the notebook -- deliberately more conservative than
        DEFAULT_AUDIT_THRESHOLD (0.26), since that is where Section 9.2's
        sweep found the disparity most visible.
    watch_states : list[str]
        States to compare with-state vs. no-state.

    Returns
    -------
    dict
        {"threshold": float,
         "auc_with_state": float, "auc_no_state": float, "auc_cost": float,
         "base_approval_with_state": float, "base_approval_no_state": float,
         "states": pd.DataFrame with columns
             state, eo_with_state, eo_no_state, shift, verdict}
        verdict is one of "fixed by removal (shortcut / geographic-proxy
        risk)", "persists (reflects real economic differences)", or
        "was already clear".
    """
    p_test_with, y_test_with = _train_ablation_variant(
        splits, FEATURES_WITH_STATE, CATEGORICAL_WITH_STATE, num_boost_round,
    )
    p_test_ns, y_test_ns = _train_ablation_variant(
        splits, FEATURES_NOSTATE, CATEGORICAL_NOSTATE, num_boost_round,
    )

    auc_with = float(roc_auc_score(y_test_with, p_test_with))
    auc_ns = float(roc_auc_score(y_test_ns, p_test_ns))

    # Both variants are trained from the same splits["test"], so their
    # actual-outcome vectors are identical -- the state label itself is
    # always available from the raw split, whether or not the model
    # currently being evaluated uses it as a feature.
    states_vec = splits["test"]["addr_state"].values

    def _eo_by_state(p_vec, y_true_vec, states, thr):
        p_vec = np.asarray(p_vec)
        y_true_vec = np.asarray(y_true_vec)
        appr = (p_vec < thr).astype(int)
        good = y_true_vec == 0
        base = appr[good].mean()
        return (
            {s: appr[good & (states_vec == s)].mean() / base for s in states},
            float(base),
        )

    eo_with, base_with = _eo_by_state(p_test_with, y_test_with.values, watch_states, threshold)
    eo_ns, base_ns = _eo_by_state(p_test_ns, y_test_ns.values, watch_states, threshold)

    rows = []
    for s in watch_states:
        w, n = eo_with[s], eo_ns[s]
        if w < EO_THRESHOLD <= n:
            verdict = "fixed by removal (shortcut / geographic-proxy risk)"
        elif w < EO_THRESHOLD and n < EO_THRESHOLD:
            verdict = "persists (reflects real economic differences)"
        else:
            verdict = "was already clear"
        rows.append({
            "state": s,
            "eo_with_state": w,
            "eo_no_state": n,
            "shift": n - w,
            "verdict": verdict,
        })

    return {
        "threshold": threshold,
        "auc_with_state": auc_with,
        "auc_no_state": auc_ns,
        "auc_cost": auc_ns - auc_with,
        "base_approval_with_state": base_with,
        "base_approval_no_state": base_ns,
        "states": pd.DataFrame(rows),
    }


def run_fairness_audit(
    model_path: str | Path | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
    fair_df: pd.DataFrame | None = None,
    num_boost_round: int | None = None,
    audit_threshold: float = DEFAULT_AUDIT_THRESHOLD,
    sweep_thresholds: list[float] = SWEEP_THRESHOLDS,
    ablation_threshold: float = ABLATION_THRESHOLD,
    watch_states: list[str] = WATCH_STATES,
    min_n: int = MIN_N,
) -> dict:
    """
    Run all three fairness-audit layers and return their combined output.
    Mirrors notebooks/analysis.ipynb Section 9 end to end.

    Layer 1/2 audit whichever model is CURRENTLY shipped (per
    features.py's INCLUDE_ADDR_STATE); Layer 3 always independently trains
    and compares a with-state model against a no-state model, regardless of
    which one that happens to be (see audit_layer3_ablation). On the real
    dataset, Layer 3 (threshold=0.22) reproduces approximately:
        Test AUC  WITH state: 0.6689
        Test AUC  NO   state: 0.6654
        AUC cost of dropping state: -0.0035

        state    WITH state    NO state    verdict
        MS            0.734       0.990    fixed by removal (shortcut / geographic-proxy risk)

    i.e. Mississippi's Equal Opportunity ratio recovers from well below the
    0.80 line to essentially parity once addr_state is removed, at a small
    AUC cost -- the model was using the state label as a shortcut, not
    reflecting a real difference in MS applicants' financials.

    Parameters
    ----------
    model_path : str, Path, or None
        Path to the joblib-packaged model dict from train.py's
        train_and_save(). Defaults to DEFAULT_MODEL_PATH. Ignored if
        fair_df is provided directly.
    splits : dict[str, pd.DataFrame] or None
        temporal_split() output; must contain "train", "calib", "test". If
        None, this calls load_raw() + temporal_split() itself. Always
        needed (Layer 3 retrains from it even if fair_df is precomputed).
    fair_df : pd.DataFrame or None
        Precomputed Test-level frame with columns addr_state/y_true/p
        (calibrated probabilities) for the CURRENTLY-shipped model --
        drives Layer 1/2 only. If provided, this skips loading the joblib
        model artifact and re-fitting isotonic calibration entirely --
        useful when you already have calibrated probabilities lying around
        (e.g. from a prior calibrate_model() call), or when feeding small
        synthetic data in tests without needing a real model artifact on
        disk. num_boost_round must be supplied alongside it (see below;
        Layer 3 needs it regardless of fair_df, since Layer 3 trains its
        own with-state/no-state models rather than using fair_df at all --
        see audit_layer3_ablation).
    num_boost_round : int or None
        Fixed boosting-round count for Layer 3's with-state/no-state
        retrains (see audit_layer3_ablation). If fair_df is None, this
        defaults to the loaded model artifact's best_iteration -- whichever
        variant (with-state or no-state) that model happens to be, its
        best_iteration is just being reused here as a reasonable, already-
        validated round budget, not as something tied to being "the
        with-state model" specifically. If fair_df is provided directly,
        this is required (there is no model artifact to read
        best_iteration from in that case).
    audit_threshold, sweep_thresholds, ablation_threshold, watch_states,
    min_n : forwarded to audit_layer1 / audit_layer2 / audit_layer3_ablation.

    Returns
    -------
    dict
        {"fair_df": pd.DataFrame,
         "layer1": pd.DataFrame, "layer2": pd.DataFrame, "layer3": dict}
    """
    if splits is None:
        df = load_raw()
        splits = temporal_split(df)

    if fair_df is None:
        model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        artifact = load_model_artifact(model_path)  # fail-closed on feature-contract mismatch
        booster = artifact["model"]
        category_maps = artifact["category_maps"]
        best_iteration = artifact["best_iteration"]

        X_calib, y_calib = _xy(splits["calib"])
        X_test, y_test = _xy(splits["test"])
        X_calib_lgb = _to_lgb_frame(X_calib, category_maps)
        X_test_lgb = _to_lgb_frame(X_test, category_maps)

        p_calib_raw = booster.predict(X_calib_lgb, num_iteration=best_iteration)
        p_test_raw = booster.predict(X_test_lgb, num_iteration=best_iteration)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_calib_raw, y_calib)
        p_test_cal = apply_calibration(iso, p_test_raw)

        fair_df = pd.DataFrame({
            "addr_state": splits["test"]["addr_state"].values,
            "y_true": y_test.values,
            "p": p_test_cal,
        })
        if num_boost_round is None:
            num_boost_round = best_iteration

    if num_boost_round is None:
        raise ValueError(
            "num_boost_round is required for Layer 3's ablation when fair_df "
            "is passed directly -- there is no model artifact to read "
            "best_iteration from in that case."
        )

    layer1 = audit_layer1(fair_df, threshold=audit_threshold, min_n=min_n)
    layer2 = audit_layer2(fair_df, thresholds=sweep_thresholds, watch_states=watch_states)
    layer3 = audit_layer3_ablation(
        splits, num_boost_round=num_boost_round,
        threshold=ablation_threshold, watch_states=watch_states,
    )

    confirmed = layer1[layer1["verdict"].str.startswith("confirmed")]["state"].tolist()
    print(f"=== Layer 1: bootstrap CIs @ threshold={audit_threshold} ===")
    print(layer1.to_string(index=False))
    print(f"Confirmed geographic-proxy risk (CI fully < {EO_THRESHOLD}): {confirmed or 'none'}\n")

    print(f"=== Layer 2: threshold sweep ({watch_states}) ===")
    print(layer2.to_string(index=False))
    print()

    print(f"=== Layer 3: ablation @ threshold={ablation_threshold} ===")
    print(f"Test AUC  WITH state: {layer3['auc_with_state']:.4f}")
    print(f"Test AUC  NO   state: {layer3['auc_no_state']:.4f}")
    print(f"AUC cost of dropping state: {layer3['auc_cost']:+.4f}\n")
    print(layer3["states"].to_string(index=False))

    return {"fair_df": fair_df, "layer1": layer1, "layer2": layer2, "layer3": layer3}
