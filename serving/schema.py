"""
The request contract and the response contract.

The request contract is NOT LOAN_SCHEMA. LOAN_SCHEMA (data_validation.py)
is the TRAINING contract: it requires `Default` and `addr_state` as non-nullable
columns, and a live request has neither. It is correct as it is and is not
touched. What the two share -- every bound, every category set -- is IMPORTED
from data_validation and features, never retyped. A second copy of a category
list is the drift features.py's feature contract exists to prevent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data_validation import (
    DTI_MAX_REAL,
    DTI_SENTINEL,
    LOAN_MAX,
    LOAN_MIN,
    REVENUE_MIN,
    VALID_HOME_OWNERSHIP,
    VALID_PURPOSE,
)
from src.features import emp_order

# ---------------------------------------------------------------------------
# EMP_LENGTH_NOT_DISCLOSED -- the string the training data uses for "declined to
# disclose". add_features() keys emp_length_missing off this exact value
# (features.py's add_features(): `out["emp_length"].eq("NI")`).
# ---------------------------------------------------------------------------
EMP_LENGTH_NOT_DISCLOSED = "NI"

# The closed set of emp_length values the model was fit on: the eleven ordinal
# strings in features.emp_order (features.py) plus "NI". Measured against
# Train (n=453,804): exactly 12 distinct values, zero nulls.
VALID_EMP_LENGTH: frozenset[str] = frozenset(emp_order) | {EMP_LENGTH_NOT_DISCLOSED}


class ScoreRequest(BaseModel):
    """
    One applicant: six applicant-reported fields, plus applicant_id.

    applicant_id identifies the applicant to the credit-bureau layer
    (serving/bureau.py's CreditBureau protocol) -- it is not itself a model
    feature. fico_n is NO LONGER an applicant-reported field as of Phase 1's
    bureau wiring: serving/app.py's /score handler fetches it via
    CreditBureau.fetch(applicant_id) and merges it with these six fields
    before feature engineering runs (see _to_raw_frame in serving/app.py).
    extra="forbid" below means a client that submits its own fico_n gets the
    same 422 as submitting any other unrecognized field -- the service, not
    the applicant, is now that value's source of truth. dti_n stays
    applicant-reported this round: Phase 1 moved fico_n first and
    deliberately left dti_n for a later step (see docs/data-decisions.md).

    Feature engineering turns the resulting seven raw fields (these six plus
    the bureau-sourced fico_n) into the eight the model scores (features.py:
    emp_length yields both emp_length_ord and the explicit
    emp_length_missing flag). The service accepts six applicant fields; it
    never accepts fico_n, emp_length_missing, or emp_length_ord directly,
    because a client that could set any of those independently can describe
    an applicant who does not exist -- or, for fico_n specifically, an
    applicant whose score did not come from a bureau pull at all.

    ------------------------------------------------------------------------
    emp_length: JSON `null` MEANS "declined to disclose", and is normalized to
    the string "NI" before anything downstream sees it.

    This is a product decision, and the data forces it. Measured on Train
    (n=453,804): emp_length has ZERO nulls and exactly 12 distinct values --
    the 11 in emp_order plus "NI". Now consider what an un-normalized null
    does. add_features() maps it through emp_order to NaN and computes
    `.eq("NI")` as False, producing (emp_length_ord=NaN, emp_length_missing=0).
    That combination occurs in 0 of 453,804 training rows. It is off the
    manifold the model was fit on.

    Measured on one fixed row (revenue=60000, dti_n=18, loan_amnt=10000,
    fico_n=700, purpose=debt_consolidation, home_ownership_n=RENT):

        emp_length    emp_length_ord  emp_length_missing   p_raw
        "5 years"     5.0             0                    0.138976
        "NI"          NaN             1                    0.192338
        null          NaN             0                    0.183507
        "bogus"       NaN             0                    0.183507

    Note the last two rows. `null`, `NaN`, and an arbitrary unmapped string are
    INDISTINGUISHABLE once add_features() has run -- all three collapse to the
    same off-manifold encoding and the same score. Normalizing null to "NI"
    moves it onto row two, where the model has 453,804 rows of support.

    The other reading of null -- "field absent, reject the request" -- is not
    reachable by accident because it is not reachable at all: emp_length is a
    closed enum over VALID_EMP_LENGTH, so no unmapped string can arrive, and
    null is mapped below. There is no code path from a validated request to
    add_features() carrying an off-manifold value.
    ------------------------------------------------------------------------

    Two deliberate asymmetries with LOAN_SCHEMA:

    1. `extra="forbid"`, where LOAN_SCHEMA sets strict=False
       (data_validation.py). A training frame legitimately carries id,
       issue_d, zip_code. A REQUEST that carries an unrecognized field is a
       client that believes it is sending something the model reads. It is
       not -- and as of the bureau wiring, this now also catches a client
       that submits its own fico_n: that field is real and model-consumed,
       and still rejected here, because the service sources it independently
       and a client-submitted value would silently go nowhere.

    2. An unseen `purpose` is rejected, where _to_lgb_frame degrades it to NaN
       and scores it (model_io.py's _to_lgb_frame, whose docstring calls that
       degradation deliberate: "a malformed/missing feature value at inference
       time shouldn't take down serving"). This overrides that, at the HTTP
       boundary only, on the following grounds. purpose in training is
       nullable=False, isin(VALID_PURPOSE) (LOAN_SCHEMA's purpose Column,
       data_validation.py), and category_maps is derived from Train alone
       (model_io.py's _train_categories). No training row therefore
       encodes purpose as NaN, so LightGBM's NaN bin for that column received
       zero training rows. Scoring an unseen purpose is not graceful
       degradation; it is a well-formed float from an untrained branch.
       (Measured: purpose="cryptocurrency_yolo" scores p_raw=0.205021 against
       0.138976 for debt_consolidation on the row above, with additivity
       intact and no warning.) An unknown purpose is a client bug, not an
       applicant attribute.

    Floats are strict. `"700"` is a 422, not because it cannot be parsed but
    because the client that sent it has a bug. Verified on pydantic 2.13.4:
    Field(strict=True) on a float ACCEPTS int 700 and float 700.0, and REJECTS
    the string "700" and the bool True with error type `float_type`. Lax mode
    accepts all four, silently turning True into 1.0.
    """

    model_config = ConfigDict(extra="forbid")

    # Identifies the applicant to the credit-bureau layer (CreditBureau.fetch,
    # serving/bureau.py) -- not itself a model feature.
    applicant_id: str = Field(min_length=1)
    revenue: float = Field(strict=True, ge=REVENUE_MIN)
    dti_n: float = Field(strict=True)
    loan_amnt: float = Field(strict=True, ge=LOAN_MIN, le=LOAN_MAX)
    emp_length: str | None
    purpose: str
    home_ownership_n: str

    @field_validator("dti_n")
    @classmethod
    def _dti_in_band_or_sentinel(cls, v: float) -> float:
        """
        Mirrors LOAN_SCHEMA's dti_n Column check (data_validation.py).

        The OR is numerically redundant -- 999 <= 1000 -- and kept explicit for
        the same reason it is kept explicit there: 999 is a missing-value
        sentinel, not a real DTI, even though the widened real band contains
        it. The lower bound matters: DTI is never negative, so -1 (another
        common sentinel) must be REJECTED, not mistaken for a real value.
        """
        if (0.0 <= v <= DTI_MAX_REAL) or v == DTI_SENTINEL:
            return v
        raise ValueError(
            f"dti_n outside real [0,{DTI_MAX_REAL}] band and not the known "
            f"{DTI_SENTINEL} sentinel"
        )

    @field_validator("emp_length")
    @classmethod
    def _null_means_not_disclosed(cls, v: str | None) -> str:
        """null -> "NI". Any other unmapped string is rejected, not collapsed."""
        if v is None:
            return EMP_LENGTH_NOT_DISCLOSED
        if v not in VALID_EMP_LENGTH:
            raise ValueError(
                f"emp_length must be null or one of {sorted(VALID_EMP_LENGTH)}; "
                f"got {v!r}. An unmapped string is not 'unknown tenure' -- it "
                f"encodes identically to null and scores off the manifold."
            )
        return v

    @field_validator("purpose")
    @classmethod
    def _purpose_is_known(cls, v: str) -> str:
        if v not in VALID_PURPOSE:
            raise ValueError(f"purpose must be one of {sorted(VALID_PURPOSE)}; got {v!r}")
        return v

    @field_validator("home_ownership_n")
    @classmethod
    def _home_ownership_is_known(cls, v: str) -> str:
        if v not in VALID_HOME_OWNERSHIP:
            raise ValueError(
                f"home_ownership_n must be one of {sorted(VALID_HOME_OWNERSHIP)}; got {v!r}"
            )
        return v


class ReasonCode(BaseModel):
    """
    One principal adverse factor, as _rank_adverse emits it (explain.py).

    `value` is a string for every feature, including numeric ones: _rank_adverse
    writes `str(v)` into the dict it returns. An "NI" applicant's emp_length_ord
    therefore arrives as the string "nan", not as a float NaN, and no NaN ever
    reaches the JSON encoder -- which is fortunate, because JSON has no NaN
    literal. This model does not "fix" that by retyping the field; it records
    why the field is what it is.

    `contribution_log_odds` is on the RAW LOG-ODDS MARGIN axis, never on the
    probability axis. See ScoreResponse.
    """

    rank: int          # 1-based (_rank_adverse's enumerate(..., start=1))
    feature: str
    value: str
    contribution_log_odds: float


class ScoredCreditReport(BaseModel):
    """
    The part of the credit pull the DECISION ACTUALLY USED, plus what identifies
    the pull. Deliberately NOT the whole CreditReport (serving/bureau.py).

    Field names are byte-aligned to CreditReport's own -- a client-facing SUBSET
    of that model, never a re-spelling of it. tests/test_serving.py asserts
    CREDIT_REPORT_KEYS <= set(CreditReport.model_fields), so a field name
    invented here cannot survive.

    ------------------------------------------------------------------------
    `dti_n` IS ABSENT, AND THAT IS THE POINT.

    CreditReport HAS a dti_n. The scoring path does not use it: _to_raw_frame
    (serving/app.py) takes dti_n from the REQUEST and leaves `report.dti_n`
    unread. Putting the pulled dti_n inside a block labelled "credit report"
    would show a client a bureau DTI sitting beside a decision computed from the
    applicant's self-reported one -- under MockBureau those are wildly different
    numbers (see docs/data-decisions.md's "move dti_n to the bureau" entry, which
    measures the mock's dti_n as uniform over [0, 1000)). That is a decision
    described one way and made another, which is the single thing this repo
    exists to prevent. So the block carries the credit value the model consumed
    (fico_n) and nothing it did not.

    `applicant_id` and `inquiry_window_days` are absent for smaller reasons.
    The client sent applicant_id; the response echoes no other request field, and
    echoing this one would imply the bureau resolved an identity it did not.
    inquiry_window_days is a disclosed-invented mock placeholder
    (_MOCK_INQUIRY_WINDOW_DAYS's comment, serving/bureau.py: "carries no
    authority beyond 'a plausible mock value'") that feeds nothing -- serving it
    would dress an armchair number as a bureau fact.
    ------------------------------------------------------------------------

    `pulled_at`, not `credit_report_pulled_at`. That prefix existed only to
    disambiguate the field at the TOP LEVEL of ScoreResponse, where it sat beside
    model_trained_at and calibrator_trained_at. Nested here the prefix would
    stutter (credit_report.credit_report_pulled_at) and, worse, would stop
    matching CreditReport's own field name -- which is what the subset invariant
    above is asserted against. This is a WIRE-FORMAT RENAME, not a pure move: a
    client reading credit_report_pulled_at will not find it.
    """

    # The one model input in this block: the value CreditBureau.fetch() returned
    # and _to_raw_frame merged into the frame the booster scored. Data, not
    # provenance -- which is why no constant naming this set may be called
    # "provenance" (see CREDIT_REPORT_KEYS, tests/test_serving.py).
    fico_n: float

    # The three that identify WHICH pull produced the fico_n above.
    bureau: str
    fico_version: str
    pulled_at: datetime


class ScoreResponse(BaseModel):
    """
    Mirrors explain_applicants()'s returned dict key-for-key (explain.py), plus
    EXACTLY ONE key of its own: `credit_report`, a ScoredCreditReport that /score
    fills in from the CreditReport it fetched. explain_applicants() knows nothing
    about the bureau layer and never returns that key itself.

    The bureau's contribution used to be three loose top-level fields (bureau,
    fico_version, credit_report_pulled_at). It is now one nested object, and the
    fetched fico_n -- the credit value the model actually scored on -- is inside
    it. Two things forced the nesting. The mirror invariant below was guarded by a
    constant naming the permitted bureau keys, and a fourth loose key would have
    made that constant name a set three-quarters provenance and one-quarter data
    (fico_n being data) -- no honest name exists for such a set, so it was removed
    rather than stretched; tests/test_serving.py now carries
    BUREAU_SOURCED_TOP_LEVEL_KEY and CREDIT_REPORT_KEYS in its place, each
    describing exactly what it holds. And the pull is one thing: a fico_n and the
    identity of the report it came from belong together, because either without
    the other is a claim nobody can audit.

    Every other field is not a reshaping of explain_applicants()'s dict, not a
    subset of it, not a renaming of it. tests/test_serving.py asserts
    set(ScoreResponse.model_fields) == set(explain_applicants()'s keys) |
    {"credit_report"} -- an ADDITIVE form, stronger than the subtraction it
    replaced, which would have silently tolerated a fourth bureau key appearing
    in both the model and the constant that excused it.

    There is no `contribution_to_probability` key, not even set to None.
    docs/explainability.md Section 5 proves percentage-point attribution is
    undefined under the shipped isotonic calibrator -- a 52-level step function
    has zero slope across 99.31% of the reject region, so "this feature added N
    points of default probability" has no value to compute. A key set to None
    would tell the next reader that a value belongs there and someone merely has
    not computed it yet. Nothing belongs there.

    ------------------------------------------------------------------------
    An EMPTY `reason_codes` list means no risk-increasing factor could be named:
    every SHAP contribution was <= 0 and the base value alone cleared the
    decision boundary. Per docs/design.md Section 6, that applicant must route
    to human review regardless of `p_calibrated`, because an adverse-action
    notice cannot list principal reasons when there are none.

    Serving does NOT make that routing decision, and there is deliberately no
    `requires_review` field. The empty list is the fact. A boolean would convert
    "we have no reasons to give" into "we decided to send this to review" -- a
    policy, and not one an API should embed. The rule lives in design.md; the
    API returns the fact it keys off.
    ------------------------------------------------------------------------

    `model_trained_at` collides with pydantic v2's reserved `model_` namespace,
    hence protected_namespaces=(). The key is not renamed: it is
    explain_applicants()'s key, and mirroring is the point.
    """

    model_config = ConfigDict(protected_namespaces=())

    scale: str                                  # == explain.CONTRIBUTION_SCALE
    p_raw: float                                # sigmoid(margin); calibrator INPUT
    p_calibrated: float                         # the decided quantity
    threshold: float
    decision: Literal["REJECT", "APPROVE"]
    base_value_log_odds: float
    raw_margin_log_odds: float                  # == base_value + sum(contributions)
    contributions_log_odds: dict[str, float]
    reason_codes: list[ReasonCode]
    model_trained_at: str | None
    calibrator_trained_at: str | None
    # The pull this decision was made on -- filled in by /score from the
    # CreditReport it fetched (serving/bureau.py), not by explain_applicants(),
    # which knows nothing about the bureau layer. Same rationale as
    # model_trained_at / calibrator_trained_at above, carried one step further:
    # a decision that used a bureau-sourced fico_n is a claim about an
    # applicant-and-report PAIR, so it ships the value it scored on together
    # with the identity of the report that supplied it. See ScoredCreditReport
    # on why the pulled dti_n is not in there.
    credit_report: ScoredCreditReport


class HealthResponse(BaseModel):
    """Readiness, plus the identity of what is loaded."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_trained_at: str | None
    calibrator_trained_at: str | None
    threshold: float


class CalibratorResponse(BaseModel):
    """
    The SHIPPED isotonic calibrator's own shape, read off the loaded artifact.

    This exists so a client can draw the step function the service actually
    decides with, rather than a snapshot of it copied into the client at some
    point in the past. The arrays below are `IsotonicRegression.X_thresholds_`
    and `.y_thresholds_` verbatim off the calibrator in ArtifactBundle
    (serving/artifacts.py) -- the same object /score composes with the booster,
    loaded once at startup through the same load_bundle() gates (which include
    load_calibrator's model-binding check, calibrate.py). There is no second
    read of the pickle and no separate path that could go stale against the
    first.

    `threshold` is bundle.threshold -- SELECTED_THRESHOLD (serving/config.py),
    the value /score decides at. It is NOT src.explain's
    DEFAULT_EXPLAIN_THRESHOLD, which is the literal 0.25 and a different float.
    It is returned here for one reason: a client drawing the reject boundary
    must draw it where the service actually puts it. A client that hardcoded
    0.25 would be drawing a line the service does not use.

    `n_distinct_y` is len(set(y_thresholds)) -- the number of levels in the step
    function, 52 for the shipped calibrator against 104 knots. The knots come in
    equal-valued PAIRS: each pair spans one flat block, and the ramp between two
    blocks is the interval between one pair's last x and the next pair's first.
    That two-to-one ratio is the whole reason percentage-point attribution is
    undefined here -- see docs/explainability.md Section 4 -- and it is a fact
    about the artifact, so it is served from the artifact rather than asserted
    in prose a client would have to trust.
    """

    x_thresholds: list[float]
    y_thresholds: list[float]
    x_min: float
    x_max: float
    n_knots: int
    n_distinct_y: int
    threshold: float
