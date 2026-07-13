"""
The HTTP boundary. Five routes, one adapter function, no scoring logic.

    GET  /healthz     readiness, plus the identity of what is loaded
    GET  /calibrator  the shipped calibrator's own shape, off the loaded bundle
    POST /score       one applicant -> decision + rank-ordered reason codes
    POST /drift       the drift demo's FICO knob -> the real monitor's verdict
                      (DEV ONLY -- absent from the production image, see
                      DRIFT_DEMO_AVAILABLE)

Run:  uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from serving.artifacts import ArtifactBundle, load_bundle
from serving.bureau import CreditBureau, CreditReport, MockBureau
from serving.errors import (
    ADDITIVITY_FAILURE_DETAIL,
    ARTIFACTS_UNAVAILABLE_DETAIL,
    BUREAU_UNAVAILABLE_DETAIL,
)
# serving/fairness.py opens a JSON file and compares one string. It imports
# nothing from src/fairness.py, pipelines/ or scripts/ -- the audit itself needs
# the assessment CSV and ~40s, and neither belongs anywhere near this process.
# The import-graph guard in tests/test_serving.py covers this line too.
from serving.fairness import (
    FairnessAudit,
    audit_model_trained_at,
    is_stale,
    load_fairness_audit,
)
from serving.schema import (
    CalibratorResponse,
    DriftRequest,
    DriftResponse,
    FairnessResponse,
    HealthResponse,
    ScoredCreditReport,
    ScoreRequest,
    ScoreResponse,
)
from src.explain import explain_applicants

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DRIFT_DEMO_AVAILABLE -- whether POST /drift can be mounted HERE.
#
# /drift answers with pipelines/drift_check.py's real PSI/KS/alarms, reached
# through scripts/demo_drift.py's drift_report(). Neither module is importable
# in the production container, and that is deliberate on both counts:
#
#   .dockerignore excludes pipelines/, and the Dockerfile COPYs only src/,
#   serving/ and models/ -- so scripts/ and pipelines/ are not in the image.
#   The Dockerfile then installs `uv sync --no-group training`, which excludes
#   mlflow and metaflow -- and drift_check.py imports mlflow at module scope,
#   and pipelines/training_flow.py (which it imports from) needs metaflow.
#
# So importing drift_report at module scope here would kill the container at
# boot, on `import serving.app`, before a single request. It would also drag
# mlflow + metaflow back into the serving dependency set -- undoing the
# [dependency-groups] split whose entire purpose is keeping them out, and
# falsifying pyproject.toml's claim that neither "appears in sys.modules after
# `import serving.app`" (a claim tests/test_serving.py now enforces rather than
# merely asserts in prose).
#
# The honest reading is that /drift is a DEMO route, not a production one. It
# monitors a synthetic population drawn from MockBureau; there is no real
# applicant stream to watch at serve time, and the production drift path is the
# batch job (`uv run python pipelines/drift_check.py`), which needs no HTTP at
# all. So the route is mounted where its machinery exists -- local dev -- and is
# simply not there in the image. find_spec probes for the module WITHOUT
# importing it, so this costs nothing at startup and cannot itself drag mlflow
# in; a real breakage inside demo_drift.py still raises loudly at create_app()
# rather than being swallowed here.
# ---------------------------------------------------------------------------
def _drift_demo_importable() -> bool:
    try:
        return importlib.util.find_spec("scripts.demo_drift") is not None
    except ModuleNotFoundError:
        # scripts/ is not on the path at all -- i.e. the production image.
        return False


DRIFT_DEMO_AVAILABLE = _drift_demo_importable()

# ---------------------------------------------------------------------------
# CORS_ALLOW_ORIGINS -- the browser-frontend gate.
#
# The browser frontend (frontend/) runs on Vite's dev server and calls this API
# cross-origin, which a browser refuses without this header. docs/PROJECT_STATUS.md
# flagged CORS as "not yet added"; this is that addition.
#
# The origins are ENUMERATED, not "*", by policy. "*" would let any page on any
# origin the user happens to have open call this API from their browser. Nothing
# here is authenticated today, so "*" costs nothing TODAY -- and that is exactly
# why it would be the wrong default to write down: it becomes a real hole the
# moment a credential or a cookie is added, and nobody re-reads a middleware
# argument that has always been there. allow_credentials is False for the same
# reason: the service has no credentials to send, and saying so is cheaper than
# discovering later that it does.
#
# Both spellings of localhost are listed because a browser treats them as
# DIFFERENT origins: a page served from 127.0.0.1:5173 is not the same origin as
# one from localhost:5173, and Vite will hand out either depending on how it is
# reached.
# ---------------------------------------------------------------------------
CORS_ALLOW_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


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
    drift_demo: bool = DRIFT_DEMO_AVAILABLE,
    audit: FairnessAudit | None = None,
) -> FastAPI:
    """
    Build the app. Pass `bundle`/`bureau` to inject pre-built ones and skip
    loading them at startup; tests do this. `bureau` injection is independent
    of `bundle`: lifespan is skipped only when `bundle` is supplied, so
    passing `bundle` without `bureau` still runs lifespan's MockBureau()
    construction -- every current test passes both together. Production
    calls create_app() with neither and lets lifespan load both.

    `drift_demo` mounts POST /drift. It defaults to DRIFT_DEMO_AVAILABLE -- true
    in the repo, false in the container -- so the route appears in dev and is
    absent in production without anyone having to remember a flag. Pass it
    explicitly to force either way; passing True where the machinery is missing
    raises at construction, which is the intended failure (a 404 would be the
    service quietly disagreeing with the caller about what it is).

    `audit` is the frozen fairness audit backing GET /fairness. It is READ HERE,
    once, rather than per request -- but unlike the bundle it is deliberately
    NOT fail-closed: load_fairness_audit() never raises, it reports (see
    serving/fairness.py). A missing or corrupt reporting artifact must not stop
    the service scoring applicants. Tests inject a stale or absent one to
    exercise the 409 / 404 paths without touching the file on disk.
    """
    audit = audit if audit is not None else load_fairness_audit()
    app = FastAPI(
        title="trust-issues credit scoring",
        description=(
            "Scores one applicant's default probability and returns a decision "
            "with rank-ordered reason codes on the raw log-odds margin."
        ),
        lifespan=None if bundle is not None else lifespan,
    )
    # See CORS_ALLOW_ORIGINS -- explicit origins, never "*".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        allow_credentials=False,
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

    @app.get("/calibrator", response_model=CalibratorResponse)
    async def calibrator(
        bundle: ArtifactBundle = Depends(get_bundle),
    ) -> CalibratorResponse:
        """
        The shipped calibrator's shape, read off the bundle /score already uses.

        `bundle.calibrator` is the IsotonicRegression load_bundle() loaded at
        startup and passed every one of its gates -- including load_calibrator's
        model-binding check (calibrate.py), which raises if the calibrator was
        fit against a different model instance. So the arrays served here cannot
        describe a calibrator other than the one composing decisions on /score:
        it is the same object, not a second read of the same path.

        Depends(get_bundle) is not decoration -- it means an unloaded bundle is
        a 503 here for exactly the reason it is one on /score, rather than this
        route inventing its own way to load a pickle.

        The threshold travels with the curve on purpose. A client that draws the
        reject boundary needs the value the service decides at (SELECTED_THRESHOLD,
        serving/config.py), and the only way to guarantee it draws the same line
        the service uses is to hand it that line rather than let it assume 0.25 --
        which is a real, different float (see SELECTED_THRESHOLD's comment).
        """
        cal = bundle.calibrator
        y = cal.y_thresholds_.tolist()
        return CalibratorResponse(
            x_thresholds=cal.X_thresholds_.tolist(),
            y_thresholds=y,
            x_min=float(cal.X_min_),
            x_max=float(cal.X_max_),
            n_knots=len(cal.X_thresholds_),
            n_distinct_y=len(set(y)),
            threshold=bundle.threshold,
        )

    @app.get("/fairness", response_model=FairnessResponse)
    async def fairness(
        bundle: ArtifactBundle = Depends(get_bundle),
    ) -> FairnessResponse:
        """
        The frozen three-layer audit -- but only if it is about THIS model.

        Unlike /drift, this route computes nothing and wraps nothing. The audit
        needs the 167 MB assessment CSV (the first line of .dockerignore,
        because the brief forbids redistributing it) and ~40s of retraining, so
        it runs offline in scripts/audit_fairness.py and freezes ~35 KB of
        derived ratios. This route opens that file. See serving/fairness.py.

        Two ways it declines, and neither of them invents a number:

          404  There is no audit artifact here. The body carries the reason
               (absent / unreadable / unknown schema_version), so a human can
               tell those apart, and no failure of a REPORTING artifact can
               take /score down with it -- the audit is blue in
               docs/architecture.html, not a gate.

          409  The audit is about a different model than the one being served.
               A frozen artifact is the one thing in this service that CAN go
               stale against the booster: retrain, and the JSON still cheerfully
               reports Mississippi at 0.7448. So it is bound to the model by
               trained_at -- the same binding load_calibrator() enforces between
               the calibrator and the booster (calibrate.py) -- and on mismatch
               the response carries BOTH timestamps and NOT ONE RATIO.

               Withholding the numbers is the point. A client handed stale
               ratios plus a warning label will draw the ratios; the only
               reliable way to stop a stale number being rendered as a current
               one is to not send it.

        Depends(get_bundle) for the same reason /calibrator uses it: the
        comparison is against the bundle /score actually decides with, not a
        second read of the same pickle.
        """
        if not audit.available:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=audit.unavailable_reason,
            )
        if is_stale(audit.audit, bundle):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": (
                        "The fairness audit on disk was run against a different "
                        "model than the one this service is serving. Refusing to "
                        "report its ratios: they describe a booster that is not "
                        "making these decisions. Re-run scripts/audit_fairness.py."
                    ),
                    "audit_model_trained_at": audit_model_trained_at(audit.audit),
                    "shipped_model_trained_at": bundle.model_trained_at,
                },
            )
        return FairnessResponse(
            **audit.audit,
            shipped_model_trained_at=bundle.model_trained_at,
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
            credit_report=ScoredCreditReport(
                # The fico_n the model actually consumed: the same
                # report.fico_n _to_raw_frame merged into the scored frame,
                # read off the same CreditReport object. It cannot disagree
                # with what was scored, because there is no second fetch.
                #
                # report.dti_n is NOT passed -- ScoredCreditReport has no such
                # field, deliberately. See its docstring (schema.py).
                fico_n=report.fico_n,
                bureau=report.bureau,
                fico_version=report.fico_version,
                pulled_at=report.pulled_at,
            ),
        )

    if drift_demo:

        @app.post("/drift", response_model=DriftResponse)
        async def drift(request: DriftRequest) -> DriftResponse:
            """
            Turn the drift demo's FICO knob and return what the REAL monitor said.

            This handler computes nothing. drift_report() (scripts/demo_drift.py)
            draws the two populations out of MockBureau and hands them to
            pipelines/drift_check.py's drift_metrics() and evaluate_alarms() --
            the same functions the batch job calls, on the same
            DEFAULT_ALARM_THRESHOLDS. There is no PSI in serving/, no KS in
            serving/, and no alarm rule in serving/. A second implementation
            here would put a number on the client's screen that the monitor that
            actually runs would never produce -- which is drift between two
            sources of truth, in the endpoint built to demonstrate drift.

            Deterministic: same mean_fico in, byte-identical numbers out (fixed
            applicant ids, hash-seeded MockBureau). A client can drag a slider
            and get a curve rather than a shimmer, and a demo can be repeated
            exactly.

            No Depends(get_bundle), unlike every other route here, and that is
            not an oversight. The distribution monitor touches no model: PSI, KS
            and the alarms are computed from the population's fico_n/dti_n alone.
            drift_check.py's ONE model-dependent signal is the calibration gap,
            which needs real labels -- a synthetic MockBureau population has no
            defaults to be right or wrong about -- so it is not part of this
            demo, and the artifacts it would need are not a dependency of this
            route. Declaring one anyway would be decoration, and would make a
            missing model look like a reason the drift monitor cannot run.
            """
            # Imported INSIDE the handler, and this is the whole reason /drift
            # can exist on this app at all. `app = create_app()` runs at module
            # scope, so importing drift_report anywhere above this line would
            # put mlflow + metaflow + pipelines into serving.app's import graph
            # -- exactly what the [dependency-groups] split and the slim image
            # exist to prevent, and what tests/test_serving.py's subprocess guard
            # now enforces. Deferring it to the first request keeps the module
            # graph clean and costs one lazy import, once, on a demo route.
            #
            # What that import costs, measured, because the two figures are
            # different and only one of them is the user's:
            #
            #   ~1.3s   `import scripts.demo_drift` in a BARE interpreter. This is
            #           the cost of the mistake -- what module-scope import would
            #           add to `import serving.app`, i.e. to container startup.
            #   ~0.40s  the first POST /drift on a running uvicorn (0.409 / 0.397
            #           across two cold restarts; 0.05s every call after).
            #
            # The second is smaller because uvicorn has ALREADY imported
            # pandas/scipy/sklearn/shap at startup, for /score. So the first
            # request pays only mlflow and metaflow's MARGINAL cost on a warm
            # graph, not the whole 1.3s. Quoting the bare-interpreter number as
            # the request cost overstates it ~3x -- and it is the number a client
            # would size its loading state against, so it is worth being exact.
            #
            # The trade is deliberate and it is a real trade: a breakage inside
            # demo_drift.py surfaces here as a 500 on the first call, not as a
            # startup crash. That is acceptable HERE and would not be for the
            # artifacts -- a bad model must never serve a request, whereas an
            # unimportable demo route just fails to demo. tests/test_demo_drift.py
            # imports the module directly, so the breakage is caught in CI either
            # way.
            from scripts.demo_drift import drift_report

            return DriftResponse(**drift_report(request.mean_fico))

    return app


app = create_app()
