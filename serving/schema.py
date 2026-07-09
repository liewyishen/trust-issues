"""
The request contract and the response contract.

The request contract is NOT LOAN_SCHEMA. LOAN_SCHEMA (data_validation.py:114-188)
is the TRAINING contract: it requires `Default` and `addr_state` as non-nullable
columns, and a live request has neither. It is correct as it is and is not
touched. What the two share -- every bound, every category set -- is IMPORTED
from data_validation and features, never retyped. A second copy of a category
list is the drift features.py's feature contract exists to prevent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data_validation import (
    DTI_MAX_REAL,
    DTI_SENTINEL,
    FICO_MAX,
    FICO_MIN,
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
# (features.py:118: `out["emp_length"].eq("NI")`).
# ---------------------------------------------------------------------------
EMP_LENGTH_NOT_DISCLOSED = "NI"

# The closed set of emp_length values the model was fit on: the eleven ordinal
# strings in features.emp_order (features.py:24-28) plus "NI". Measured against
# Train (n=453,804): exactly 12 distinct values, zero nulls.
VALID_EMP_LENGTH: frozenset[str] = frozenset(emp_order) | {EMP_LENGTH_NOT_DISCLOSED}


class ScoreRequest(BaseModel):
    """
    One applicant. Seven raw application-time fields.

    Feature engineering turns these seven into the eight the model scores
    (features.py: emp_length yields both emp_length_ord and the explicit
    emp_length_missing flag). The service accepts the seven; it never accepts
    the eight, because a client that can set emp_length_missing independently
    of emp_length can describe an applicant who does not exist.

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
       (data_validation.py:186). A training frame legitimately carries id,
       issue_d, zip_code. A REQUEST that carries an eighth field is a client
       that believes it is sending something the model reads. It is not.

    2. An unseen `purpose` is rejected, where _to_lgb_frame degrades it to NaN
       and scores it (train.py:151-159, which calls that degradation
       deliberate: "a malformed/missing feature value at inference time
       shouldn't take down serving"). This overrides that, at the HTTP boundary
       only, on the following grounds. purpose in training is nullable=False,
       isin(VALID_PURPOSE) (data_validation.py:169-173), and category_maps is
       derived from Train alone (train.py:132-145). No training row therefore
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

    revenue: float = Field(strict=True, ge=REVENUE_MIN)
    dti_n: float = Field(strict=True)
    loan_amnt: float = Field(strict=True, ge=LOAN_MIN, le=LOAN_MAX)
    fico_n: float = Field(strict=True, ge=FICO_MIN, le=FICO_MAX)
    emp_length: str | None
    purpose: str
    home_ownership_n: str

    @field_validator("dti_n")
    @classmethod
    def _dti_in_band_or_sentinel(cls, v: float) -> float:
        """
        Mirrors LOAN_SCHEMA's dti_n check (data_validation.py:123-137).

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
    One principal adverse factor, as _rank_adverse emits it (explain.py:369-375).

    `value` is a string for every feature, including numeric ones: _rank_adverse
    writes `str(v)` at explain.py:373. An "NI" applicant's emp_length_ord
    therefore arrives as the string "nan", not as a float NaN, and no NaN ever
    reaches the JSON encoder -- which is fortunate, because JSON has no NaN
    literal. This model does not "fix" that by retyping the field; it records
    why the field is what it is.

    `contribution_log_odds` is on the RAW LOG-ODDS MARGIN axis, never on the
    probability axis. See ScoreResponse.
    """

    rank: int          # 1-based (explain.py:375, enumerate(..., start=1))
    feature: str
    value: str
    contribution_log_odds: float


class ScoreResponse(BaseModel):
    """
    Mirrors explain_applicants()'s dict key-for-key (explain.py:443-458).

    Not a reshaping of it, not a subset of it, not a renaming of it.
    tests/test_serving.py asserts the two key sets are equal, so this model
    cannot drift from the function it serializes.

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


class HealthResponse(BaseModel):
    """Readiness, plus the identity of what is loaded."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_trained_at: str | None
    calibrator_trained_at: str | None
    threshold: float
