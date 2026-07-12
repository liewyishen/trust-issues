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
from serving.bureau import CreditBureau, CreditReport, MockBureau
from serving.errors import (
    ADDITIVITY_FAILURE_DETAIL,
    ARTIFACTS_UNAVAILABLE_DETAIL,
    BUREAU_UNAVAILABLE_DETAIL,
)
from serving.schema import HealthResponse, ScoreRequest, ScoreResponse
from src.explain import explain_applicants

logger = logging.getLogger(__name__)


def _to_raw_frame(request: ScoreRequest, report: CreditReport) -> pd.DataFrame:
    """
    One applicant -> a one-row DataFrame in the RAW schema.

    Merges two sources: `request`'s six applicant-reported fields (everything
    but applicant_id, which identifies the applicant to the bureau and is not
    itself a model feature) and `report.fico_n`, the bureau-sourced value
    CreditBureau.fetch(applicant_id) returned. Only fico_n is taken from
    `report` -- report.dti_n is currently unused (dti_n stays
    applicant-reported this round; see docs/data-decisions.md), and report's
    provenance fields (bureau, pulled_at, fico_version,
    inquiry_window_days) are deliberately NOT merged in here: this frame's
    contract is exactly the seven raw fields explain_applicants()/
    add_features() expect, nothing more, even though add_features() would
    silently ignore an extra column if one were added (_x() always subsets to
    FEATURES).

    Double brackets, not a Series: add_features() calls .map()/.eq() on columns
    and _to_lgb_frame() assigns pd.Categorical into columns, so a Series will
    not work (explain_applicants()'s docstring, explain.py).

    The result carries exactly the seven raw fields the RAW schema expects
    (revenue, dti_n, loan_amnt, fico_n, emp_length, purpose,
    home_ownership_n) -- byte-identical in shape to what this function
    produced before the bureau was wired in, just assembled from two sources
    now instead of one. emp_length is already normalized from null to "NI" by
    ScoreRequest's validator. Nothing is added, renamed, or engineered here --
    add_features() owns that, reached through explain_applicants().
    """
    return pd.DataFrame([{
        **request.model_dump(exclude={"applicant_id"}),
        "fico_n": report.fico_n,
    }])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load the artifacts once, and refuse to serve if they are wrong. Construct
    the credit-bureau client once too.

    Any exception from load_bundle() here propagates: uvicorn exits non-zero
    and no request is ever accepted. That is the intent for the model
    artifacts. A missing pickle, a feature-contract mismatch, a stale
    calibrator and a schema/category divergence are all deployment errors
    with nothing to retry. See serving/errors.py. MockBureau() cannot fail --
    it performs no I/O -- so there is nothing to fail closed on for the
    bureau side yet; a real bureau client's construction/connection failure
    mode is deferred (see docs/data-decisions.md's Phase 1 bureau-wiring
    entry).
    """
    app.state.bundle = load_bundle()
    app.state.bureau = MockBureau()
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


def get_bureau(request: Request) -> CreditBureau:
    """503 only when the app object exists without startup having run -- tests.

    Mirrors get_bundle exactly -- see its docstring and serving/errors.py.
    """
    bureau = getattr(request.app.state, "bureau", None)
    if bureau is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=BUREAU_UNAVAILABLE_DETAIL,
        )
    return bureau


def create_app(
    bundle: ArtifactBundle | None = None,
    bureau: CreditBureau | None = None,
) -> FastAPI:
    """
    Build the app. Pass `bundle`/`bureau` to inject pre-built ones and skip
    loading them at startup; tests do this. `bureau` injection is independent
    of `bundle`: lifespan is skipped only when `bundle` is supplied, so
    passing `bundle` without `bureau` still runs lifespan's MockBureau()
    construction -- every current test passes both together. Production
    calls create_app() with neither and lets lifespan load both.
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
    if bureau is not None:
        app.state.bureau = bureau

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
        bureau: CreditBureau = Depends(get_bureau),
    ) -> ScoreResponse:
        """
        Score one applicant.

        fico_n is fetched from `bureau` before scoring, not read off
        `applicant` -- ScoreRequest no longer carries it (see schema.py).
        `bureau.fetch(applicant.applicant_id)` is called unguarded: per
        CreditBureau's contract it "raises on a failed pull", but MockBureau
        (the only implementation wired in today) performs no I/O and cannot
        fail, so there is no reachable case to catch yet. A real bureau's
        failure mode is deliberately deferred -- see docs/data-decisions.md's
        Phase 1 bureau-wiring entry.

        `explainer=` is deliberately NOT passed. explain_applicants() forwards
        None to _get_explainer (explain.py), which then constructs a fresh
        shap.TreeExplainer on the booster it just loaded -- ~67 ms, per request,
        never shared. Passing bundle's booster's explainer would both defeat
        that and build the explainer on a different Booster object than the one
        _assert_additivity checks against.

        The threshold comes from the bundle, i.e. from serving/config.py, i.e.
        from MLflow run cca4c361 -- NOT from explain_applicants()'s default,
        which is the literal 0.25 and is a different float.

        max_reasons is left at DEFAULT_MAX_REASONS (4). That number is
        inherited from the notebook and is not ours to fix; see
        DEFAULT_MAX_REASONS's own comment in explain.py.
        """
        report = bureau.fetch(applicant.applicant_id)

        try:
            results = explain_applicants(
                _to_raw_frame(applicant, report),
                model_path=bundle.model_path,
                calibrator_path=bundle.calibrator_path,
                threshold=bundle.threshold,
            )
        except ValueError:
            # _assert_additivity (explain.py) is the case this catches:
            # the explanation does not reconstruct the score. Log everything,
            # return nothing. A decision without a valid explanation is worse
            # than no decision.
            logger.exception("additivity guard failed -- refusing to return a decision")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ADDITIVITY_FAILURE_DETAIL,
            ) from None

        return ScoreResponse(
            **results[0],
            bureau=report.bureau,
            fico_version=report.fico_version,
            credit_report_pulled_at=report.pulled_at,
        )

    return app


app = create_app()
