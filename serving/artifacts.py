"""
Startup artifact loading. Fail-closed, once, before the first request.

Loading here is not about cost -- both loads are ~1 ms. It is so that a bad
artifact kills the process at boot rather than surfacing as a 500 on the first
real applicant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from serving.config import CALIBRATOR_PATH, MODEL_PATH, SELECTED_THRESHOLD
from src.calibrate import load_calibrator
from src.data_validation import VALID_HOME_OWNERSHIP, VALID_PURPOSE
from src.explain import _calibrator_trained_at
from src.model_io import load_model_artifact


@dataclass(frozen=True)
class ArtifactBundle:
    """
    Everything loaded at startup, and nothing that is loaded per request.

    ------------------------------------------------------------------------
    This bundle holds NO EXPLAINER, deliberately.

    Measured on the shipped 240-tree booster (shap 0.50.0, 7 reps):

        shap.TreeExplainer(booster)      66.97 ms min / 70.25 ms median

    Caching it would be the obvious win, and it is not taken. shap_values()
    OVERWRITES explainer.expected_value as a side effect of the call (inside
    shap's TreeExplainer.shap_values), and _shap_matrix reads that attribute
    AFTER the call, never before (explain.py). A cached explainer is
    therefore shared mutable state whose base value depends on whoever called
    it last. Under concurrency, request A's write inside
    TreeExplainer.shap_values can land
    between request B's shap_values() call and B's read of expected_value,
    handing B a wrong base_value_log_odds and a wrong raw_margin_log_odds --
    with no exception raised, because the arithmetic is still valid, merely
    about the wrong applicant.

    "UNDER CONCURRENCY" IS REAL IN PRINCIPLE AND UNREACHABLE AS SHIPPED, and
    this paragraph carried the first half without the second. /score is an
    `async def` handler that never awaits, so two calls do not interleave:
    measured, one thread, disjoint in-handler intervals. The mechanism above is
    unchanged and still the reason nothing is cached -- what was missing is that
    the danger is DOUBLE-covered, once by rebuilding per request
    (serving/app.py:383-388) and once by non-overlap, and only the first cover
    was written down. The second is one keyword deep: as a `def` handler /score
    runs in Starlette's threadpool and the same two calls measurably overlap,
    for no wall-clock gain, because the work is CPU-bound and the GIL serializes
    it regardless. So caching this explainer is safe TODAY for a reason the
    caching diff would not mention -- which is exactly how two individually
    correct changes become one silent wrong answer.

    5bc5ac7 made "as shipped" a checked fact rather than a sentence:
    tests/test_serving.py's test_score_cannot_yield_the_event_loop_mid_request
    asserts FastAPI's own dispatch switch (dependant.is_coroutine_callable) and
    the absence of any await in the handler. This was the THIRD copy of that
    claim; 5bc5ac7 scoped docs/design.md:131 and docs/explainability.md:868 and
    named this one, which a No-serving/ scope constraint held back exactly one
    commit -- not because it was any less over-broad.

    67 ms per request buys statelessness. See _get_explainer's docstring
    (explain.py) and docs/explainability.md Section 10, which records
    the pred_contrib migration that would remove the cost entirely and is
    deliberately not done.
    ------------------------------------------------------------------------

    The paths are carried because explain_applicants() has no parameter for
    "use this already-loaded artifact": it gates ALL artifact loading on the
    precomputed escape hatch being fully supplied (explain_applicants()'s
    all-or-nothing hatch check, explain.py), so passing
    only `explainer=` still re-reads both pickles from disk. Using the escape
    hatch instead would mean reimplementing _to_lgb_frame + _shap_matrix +
    apply_calibration here AND bypassing _assert_additivity, which lives inside
    _shap_matrix. The guard would not run on the serving path. So we pay the
    reload: ~1.2 ms against a ~77 ms request, absorbed by the page cache.
    Adding an `artifact=` parameter to explain_applicants() would remove it.
    That change is named, not made.
    """

    booster: lgb.Booster
    best_iteration: int
    category_maps: dict[str, pd.Index]
    calibrator: IsotonicRegression
    model_trained_at: str | None
    calibrator_trained_at: str | None
    threshold: float
    model_path: Path
    calibrator_path: Path


def _assert_serving_enums_match_artifact(category_maps: dict[str, pd.Index]) -> None:
    """
    Fail closed if the request schema's category sets have drifted from the
    categories the shipped model was actually fit on.

    schema.py validates `purpose` and `home_ownership_n` against
    data_validation's VALID_PURPOSE / VALID_HOME_OWNERSHIP. The model was fit
    against category_maps, derived from Train alone (model_io.py's
    _train_categories). Those
    are two independent sources of truth for the same list, and nothing has
    ever compared them. Today they agree exactly -- 14 purposes and 4 ownership
    values, verified by run. That agreement is a property of the current
    artifact, not a guarantee.

    If they diverge, the failure is silent and bad in both directions: a value
    the schema admits but the model never saw routes to an untrained NaN bin,
    and a value the model knows but the schema rejects is a 422 for a
    legitimate applicant. Neither raises anything today. This does.
    """
    expected = {"purpose": VALID_PURPOSE, "home_ownership_n": VALID_HOME_OWNERSHIP}
    for column, schema_values in expected.items():
        model_values = set(category_maps[column])
        if model_values != set(schema_values):
            raise ValueError(
                "Serving schema / model category mismatch -- refusing to serve.\n"
                f"  column: {column}\n"
                f"  the model was trained on: {sorted(model_values)}\n"
                f"  the request schema admits: {sorted(schema_values)}\n"
                f"  admitted but never trained: {sorted(set(schema_values) - model_values)}\n"
                f"  trained but not admitted:   {sorted(model_values - set(schema_values))}\n"
                "A value the schema admits but the model never saw scores from an "
                "untrained bin. Retrain, or reconcile data_validation.py's category "
                "sets with the shipped artifact."
            )


def load_bundle(
    model_path: str | Path | None = None,
    calibrator_path: str | Path | None = None,
    threshold: float = SELECTED_THRESHOLD,
) -> ArtifactBundle:
    """
    Load the model, the calibrator, and the category contract. Raise on any
    inconsistency between them.

    Three gates, none of them new:

      load_model_artifact (model_io.py)  raises if the packaged FEATURES /
          CATEGORICAL differ from what features.py declares now.
      load_calibrator (calibrate.py)   raises if the calibrator was fit
          against a different model instance (trained_at mismatch).
      _assert_serving_enums_match_artifact  raises if the request schema's
          category sets differ from the model's.

    A missing pickle raises FileNotFoundError from joblib, which is the fourth
    gate and needs no code here.

    Measured load cost (7 reps): load_model_artifact 1.08 ms min / 1.23 median;
    the calibrator 0.09 ms min / 0.10 median. Cheap. Boot-time failure is the
    point, not the milliseconds.
    """
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    calibrator_path = (
        Path(calibrator_path) if calibrator_path is not None else CALIBRATOR_PATH
    )

    artifact = load_model_artifact(model_path)          # feature contract
    calibrator = load_calibrator(calibrator_path, model_artifact=artifact)  # binding
    _assert_serving_enums_match_artifact(artifact["category_maps"])         # categories

    return ArtifactBundle(
        booster=artifact["model"],
        best_iteration=artifact["best_iteration"],
        category_maps=artifact["category_maps"],
        calibrator=calibrator,
        model_trained_at=artifact.get("trained_at"),
        calibrator_trained_at=_calibrator_trained_at(calibrator_path),
        threshold=threshold,
        model_path=model_path,
        calibrator_path=calibrator_path,
    )
