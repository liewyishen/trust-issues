"""
Probability calibration for the LendingClub granting-model LightGBM model.

Fixes notebooks/analysis.ipynb Cell 28 in place as reusable, testable code:
fit an isotonic regression on the Calib split to correct the raw LightGBM
scores into probabilities that mean what they say, then report Brier score
and mean predicted probability before/after.

This module loads the packaged {model, features, category_maps, ...} dict
produced by train.py's train_and_save() (models/lgbm_model.pkl) rather than
a bare model, and reuses model_io.py's private encoding helpers (_xy,
_to_lgb_frame) so Calib/Test are encoded EXACTLY the way the model was
trained -- a second, independently-written encoding path here would be
exactly the kind of train/serve skew train.py's own packaging comment warns
about.

Honest framing, worth stating up front since it's easy to get backwards:
removing scale_pos_weight (a model_io.py decision, made before this module ever
runs) is what does most of the calibration work -- it pulls mean predicted
probability from ~0.48 down to ~0.17 against an actual default rate of
~0.21-0.23, resolving roughly 90% of the gap on its own. Isotonic regression
here is a refinement of the REMAINING curve shape, not the rescue. See
calibrate_model()'s docstring for the numbers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data_loader import load_raw, temporal_split
from .model_io import DEFAULT_MODEL_DIR, _to_lgb_frame, _xy, load_model_artifact

DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "lgbm_model.pkl"


def apply_calibration(iso: IsotonicRegression, raw_probs) -> np.ndarray:
    """
    Thin wrapper around iso.transform().

    Exists so downstream callers (evaluate.py, serving) import one named
    function instead of each independently calling .transform() on whatever
    variable happens to hold the calibrator -- the calibration step becomes
    a documented part of the inference path rather than an implicit detail
    every caller has to remember to replicate.
    """
    return iso.transform(raw_probs)


def load_calibrator(
    calibrator_path: str | Path,
    model_artifact: dict | None = None,
) -> IsotonicRegression:
    """
    Load a previously saved isotonic calibrator, optionally enforcing that it
    was fit against the model you are about to score with.

    An isotonic calibrator is a lookup table shaped entirely by ONE model's raw
    score distribution on Calib; applied to a DIFFERENT (e.g. retrained) model's
    raw scores it produces confidently wrong probabilities with no error raised
    -- the exact failure calibrate_model()'s packaging note warns about.
    calibrate_model() records the model's identity (model_path + the model
    artifact's own `trained_at`) alongside the calibrator for precisely this
    reason. That binding used to be written and then discarded on load; now,
    pass the model artifact you are about to score with as `model_artifact` and
    this ENFORCES it: if the calibrator was fit against a different model
    instance (its `trained_at` differs), it raises rather than silently applying
    a stale calibrator. Omit `model_artifact` (default) only for bare
    inspection or tests that just want the transform.

    Raises
    ------
    ValueError
        If model_artifact is given and the calibrator's recorded
        model-trained_at does not match model_artifact["trained_at"].
    """
    artifact = joblib.load(calibrator_path)
    if model_artifact is not None:
        fit_against = artifact.get("model_trained_at")
        current = model_artifact.get("trained_at")
        if fit_against is not None and current is not None and fit_against != current:
            raise ValueError(
                "Stale calibrator -- refusing to apply.\n"
                f"  calibrator was fit against a model trained_at={fit_against!r}\n"
                f"  current model trained_at={current!r}\n"
                "An isotonic calibrator is valid only for the exact model whose "
                "raw scores it was fit on. Re-run calibrate_model() against the "
                "current model before scoring."
            )
    return artifact["calibrator"]


def calibrate_model(
    model_path: str | Path | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
    save_calibrator: bool = True,
    calibrator_path: str | Path | None = None,
) -> tuple[IsotonicRegression, dict]:
    """
    Fit an isotonic calibrator on Calib, evaluate raw vs. calibrated
    probabilities on Test. Mirrors notebooks/analysis.ipynb Cell 28.

    Why Calib and not Train or Test: Train is what the model already
    memorized, so a calibrator fit there would just rediscover the model's
    own (possibly overconfident) training-set behavior instead of correcting
    it -- calibration requires data the model has not seen shape its
    predictions from. Test is supposed to be touched exactly once, at final
    evaluation (see model_io.py's train_lgb() docstring on early stopping for
    the same discipline) -- fitting the calibrator there would spend that
    one look on calibration instead of on an honest final read. Calib is a
    third, disjoint 2015 slice that exists for exactly this purpose: data
    the model never trained on, and that Test never has to share.

    On the real dataset (LightGBM, no scale_pos_weight, best_iteration from
    train_lgb(), production feature set per features.py's INCLUDE_ADDR_STATE
    -- see docs/data-decisions.md, "Execute the fairness conclusion: remove
    addr_state from the production model"), this produces approximately:
        Actual test default rate:      0.2323
        Raw (uncalibrated):     Brier=0.1717  mean_pred=0.1705  AUC=0.6660
        Calibrated (isotonic):  Brier=0.1692  mean_pred=0.1915  AUC=0.6654

    The AUC barely moves (0.6660 -> 0.6654): isotonic regression is a
    monotonic transform, so it only rescales the probability axis, it does
    not reorder predictions. The tiny AUC change comes from isotonic's flat
    regions occasionally tying two examples that had different raw scores,
    not from any change in ranking direction.

    Parameters
    ----------
    model_path : str, Path, or None
        Path to the joblib-packaged model dict from train.py's
        train_and_save(). Defaults to DEFAULT_MODEL_PATH
        (PROJECT_ROOT / "models" / "lgbm_model.pkl").
    splits : dict[str, pd.DataFrame] or None
        temporal_split() output; must contain "calib" and "test". If None,
        this calls load_raw() + temporal_split() itself. Passing splits
        explicitly is how tests feed small synthetic data instead of the
        real CSV.
    save_calibrator : bool
        If True (default), persist the fitted calibrator (see
        save_calibrator's packaging note below).
    calibrator_path : str, Path, or None
        Where to write the calibrator artifact. Defaults to
        Path(model_path).parent / "isotonic_calibrator.pkl", i.e. it lives
        next to the model it was fit against.

    Returns
    -------
    (IsotonicRegression, dict)
        The fitted calibrator, and a metrics dict:
        {"actual_test_default_rate": float,
         "raw":        {"brier":..., "mean_pred":..., "auc":...},
         "calibrated": {"brier":..., "mean_pred":..., "auc":...}}
    """
    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    artifact = load_model_artifact(model_path)  # fail-closed on feature-contract mismatch
    model = artifact["model"]
    category_maps = artifact["category_maps"]
    best_iteration = artifact["best_iteration"]

    if splits is None:
        df = load_raw()
        splits = temporal_split(df)

    X_calib, y_calib = _xy(splits["calib"])
    X_test, y_test = _xy(splits["test"])
    X_calib_lgb = _to_lgb_frame(X_calib, category_maps)
    X_test_lgb = _to_lgb_frame(X_test, category_maps)

    p_calib_raw = model.predict(X_calib_lgb, num_iteration=best_iteration)
    p_test_raw = model.predict(X_test_lgb, num_iteration=best_iteration)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_calib_raw, y_calib)
    p_test_cal = apply_calibration(iso, p_test_raw)

    # Annotated rather than inferred: the values are heterogeneous (one float
    # beside two nested dicts), so mypy joins them to `object` and the
    # metrics['raw']['brier'] reads below stop being indexable. This is the one
    # place in this round where the type checker found something an annotation
    # genuinely fixes, rather than a contract a runtime raise already holds.
    metrics: dict[str, Any] = {
        "actual_test_default_rate": float(y_test.mean()),
        "raw": {
            "brier": float(brier_score_loss(y_test, p_test_raw)),
            "mean_pred": float(p_test_raw.mean()),
            "auc": float(roc_auc_score(y_test, p_test_raw)),
        },
        "calibrated": {
            "brier": float(brier_score_loss(y_test, p_test_cal)),
            "mean_pred": float(p_test_cal.mean()),
            "auc": float(roc_auc_score(y_test, p_test_cal)),
        },
    }

    print(f"Actual test default rate: {metrics['actual_test_default_rate']:.4f}\n")
    print(f"{'Stage':<28}{'Brier':>10}{'mean_pred':>12}{'AUC':>10}")
    print("-" * 60)
    print(f"{'Raw (uncalibrated)':<28}"
          f"{metrics['raw']['brier']:>10.4f}"
          f"{metrics['raw']['mean_pred']:>12.4f}"
          f"{metrics['raw']['auc']:>10.4f}")
    print(f"{'After isotonic':<28}"
          f"{metrics['calibrated']['brier']:>10.4f}"
          f"{metrics['calibrated']['mean_pred']:>12.4f}"
          f"{metrics['calibrated']['auc']:>10.4f}")

    if save_calibrator:
        # --- Packaging: dump a DICT bundling the calibrator with a
        # reference to the model it was fit against, not the bare
        # IsotonicRegression object.
        #
        # An isotonic calibrator is a lookup table shaped entirely by ONE
        # specific model's raw score distribution on Calib. If the
        # underlying LightGBM model is ever retrained (new data, new
        # hyperparameters, even a new random seed), its raw scores live on
        # a different scale and this exact mapping is no longer valid for
        # it -- silently applying a stale calibrator to a new model's raw
        # scores would produce confidently wrong probabilities with no
        # error raised. Storing model_path alongside the calibrator makes
        # that binding explicit and inspectable, the same discipline
        # train_and_save() applies to the model artifact itself.
        calibrator_path = (
            Path(calibrator_path)
            if calibrator_path is not None
            else model_path.parent / "isotonic_calibrator.pkl"
        )
        calibrator_artifact = {
            "calibrator": iso,
            "model_path": str(model_path),
            # The identity of the model this calibrator was fit against, so
            # load_calibrator() can refuse to apply it to a different (e.g.
            # retrained) model. This is the binding that makes the model_path
            # note above enforceable rather than merely inspectable.
            "model_trained_at": artifact.get("trained_at"),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(calibrator_artifact, calibrator_path)

    return iso, metrics
