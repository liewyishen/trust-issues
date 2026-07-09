"""
The HTTP boundary. Two routes, one adapter function, no scoring logic.

    GET  /healthz   readiness, plus the identity of what is loaded
    POST /score     one applicant -> decision + rank-ordered reason codes

Run:  uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status

from serving.artifacts import ArtifactBundle, load_bundle
from serving.errors import ADDITIVITY_FAILURE_DETAIL, ARTIFACTS_UNAVAILABLE_DETAIL
from serving.schema import HealthResponse, ScoreRequest, ScoreResponse
from src.explain import explain_applicants

logger = logging.getLogger(__name__)


def _to_raw_frame(request: ScoreRequest) -> pd.DataFrame:
    """
    One applicant -> a one-row DataFrame in the RAW schema.

    Double brackets, not a Series: add_features() calls .map()/.eq() on columns
    and _to_lgb_frame() assigns pd.Categorical into columns, so a Series will
    not work (explain.py:394-396).

    model_dump() emits exactly the seven raw fields, with emp_length already
    normalized from null to "NI" by ScoreRequest's validator. Nothing is added,
    renamed, or engineered here -- add_features() owns that, reached through
    explain_applicants().
    """
    return pd.DataFrame([request.model_dump()])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load the artifacts once, and refuse to serve if they are wrong.

    Any exception here propagates: uvicorn exits non-zero and no request is
    ever accepted. That is the intent. A missing pickle, a feature-contract
    mismatch, a stale calibrator and a schema/category divergence are all
    deployment errors with nothing to retry. See serving/errors.py.
    """
    app.state.bundle = load_bundle()
    logger.info(
        "artifacts loaded: model_trained_at=%s calibrator_trained_at=%s threshold=%r",
        app.state.bundle.model_trained_at,
        app.state.bundle.calibrator_trained_at,
        app.state.bundle.threshold,
    )
    yield


def get_bundle(request: Request) -> ArtifactBundle:
    """503 only when the app object exists without startup having run -- tests."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ARTIFACTS_UNAVAILABLE_DETAIL,
        )
    return bundle


def create_app(bundle: ArtifactBundle | None = None) -> FastAPI:
    """
    Build the app. Pass `bundle` to inject a pre-loaded one and skip lifespan;
    tests do this. Production calls create_app() and lets lifespan load.
    """
    app = FastAPI(
        title="trust-issues credit scoring",
        description=(
            "Scores one applicant's default probability and returns a decision "
            "with rank-ordered reason codes on the raw log-odds margin."
        ),
        lifespan=None if bundle is not None else lifespan,
    )
    if bundle is not None:
        app.state.bundle = bundle

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(bundle: ArtifactBundle = Depends(get_bundle)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_trained_at=bundle.model_trained_at,
            calibrator_trained_at=bundle.calibrator_trained_at,
            threshold=bundle.threshold,
        )

    @app.post("/score", response_model=ScoreResponse)
    async def score(
        applicant: ScoreRequest,
        bundle: ArtifactBundle = Depends(get_bundle),
    ) -> ScoreResponse:
        """
        Score one applicant.

        `explainer=` is deliberately NOT passed. explain_applicants() forwards
        None to _get_explainer (explain.py:214), which then constructs a fresh
        shap.TreeExplainer on the booster it just loaded -- ~67 ms, per request,
        never shared. Passing bundle's booster's explainer would both defeat
        that and build the explainer on a different Booster object than the one
        _assert_additivity checks against.

        The threshold comes from the bundle, i.e. from serving/config.py, i.e.
        from MLflow run cca4c361 -- NOT from explain_applicants()'s default,
        which is the literal 0.25 and is a different float.

        max_reasons is left at DEFAULT_MAX_REASONS (4). That number is
        inherited from the notebook and is not ours to fix; see its comment at
        explain.py:121-128.
        """
        try:
            results = explain_applicants(
                _to_raw_frame(applicant),
                model_path=bundle.model_path,
                calibrator_path=bundle.calibrator_path,
                threshold=bundle.threshold,
            )
        except ValueError:
            # _assert_additivity (explain.py:158) is the case this catches:
            # the explanation does not reconstruct the score. Log everything,
            # return nothing. A decision without a valid explanation is worse
            # than no decision.
            logger.exception("additivity guard failed -- refusing to return a decision")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ADDITIVITY_FAILURE_DETAIL,
            ) from None

        return ScoreResponse(**results[0])

    return app


app = create_app()
