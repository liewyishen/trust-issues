"""
The credit-bureau data contract, and a deterministic mock implementation.

This module is now wired into scoring: serving/app.py imports it, keeps a
CreditBureau on app.state (lifespan constructs MockBureau; create_app()
accepts an injected one; get_bureau() resolves it per request), and the
/score handler calls bureau.fetch(applicant_id) before scoring. `fico_n` is
bureau-sourced -- ScoreRequest no longer carries the field, and ScoreResponse
returns the pull under a single nested `credit_report` key
(serving/schema.py's ScoredCreditReport: the fetched fico_n, plus the bureau,
fico_version and pulled_at that identify which pull produced it). `dti_n`
stays applicant-reported this round: report.dti_n is fetched but deliberately
unused (see docs/data-decisions.md's "Phase 1 bureau wiring" entry), and for
that same reason it is deliberately absent from ScoredCreditReport -- a block
labelled "credit report" must not show a DTI the decision did not use.
src/explain.py's explain_applicants() remains untouched -- app.py's
_to_raw_frame() hands it the same seven-field raw frame as before, now
assembled from two sources.

Two things this module refuses to do, on purpose:

  - It does not compute dti_n. Reconnaissance on this repo confirmed dti_n is
    a raw, pre-computed field supplied by the data source (see
    src/data_validation.py's dti_n Column and its 495-row 2016+ high-DTI
    investigation) -- there is no monthly_debt or income-ratio field anywhere
    in this codebase to derive it from. A real credit bureau pulls dti_n the
    same way: pre-computed, from the applicant's tradeline history, not
    computed by the lender's own model-serving code. CreditReport.dti_n is
    therefore modeled as a bureau-supplied value, passed through unchanged,
    exactly like features.py's NUMERIC passthroughs (revenue, loan_amnt,
    fico_n) are passed through unchanged rather than re-derived.

  - It does not invent new numeric constants for fields the shipped model
    already constrains. fico_n's and dti_n's bounds are imported from
    src/data_validation.py (FICO_MIN, FICO_MAX, DTI_MAX_REAL, DTI_SENTINEL) --
    the constants ScoreRequest bound fico_n to before Phase 1 moved the field
    here, and still binds dti_n to (serving/schema.py) -- so a bureau-supplied
    value and an applicant-reported one are held to the identical domain
    contract. Fields with no existing analog in this repo
    (inquiry_window_days, and the mock's synthetic pulled_at) are called out
    below as new to this module -- invented values must be disclosed as
    invented, this repo's "don't armchair a number silently" discipline.

Only the two credit fields the current 8-feature model actually consumes
(fico_n, dti_n) are modeled. A real bureau report carries dozens more fields
(delinq_2yrs, open_acc, revol_util, ...) -- none of them feed the shipped
model, so adding them here would be dead weight with no consumer.
docs/data-decisions.md's "Phase 1 credit-bureau foundation" entry records
which fields were considered and why they stayed out.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data_validation import DTI_MAX_REAL, DTI_SENTINEL, FICO_MAX, FICO_MIN

# ---------------------------------------------------------------------------
# MockBureau's fixed reference timestamp for pulled_at.
#
# New to this module -- there is no existing constant to import for this.
# MockBureau.fetch(applicant_id) must be deterministic: the same applicant_id
# returns a byte-identical report every call, forever, the same reproducibility
# discipline data_loader.py's RANDOM_SEED = 42 and temporal_split()'s fixed
# shuffle already hold elsewhere in this repo. datetime.now() would break that
# on its first tick past midnight. A single fixed constant sidesteps the
# conflict entirely: every mock report shares one pulled_at, since MockBureau
# has no real pull to timestamp and manufacturing a fake-but-varying date per
# applicant would only be reproducibility theater. Timezone-aware (UTC), to
# match ScoreResponse's other timestamp-shaped fields being ISO strings from
# datetime.now(timezone.utc) in src/train.py.
# ---------------------------------------------------------------------------
_MOCK_PULLED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)

# The mock's fixed FICO model version string. Deliberately named so it cannot
# be mistaken for a real vendor's version string (e.g. "FICO 8", "FICO 9").
_MOCK_FICO_VERSION = "mock-fico-v1"

# New to this module -- no existing constant covers "how many days of inquiry
# history a bureau pull represents." 180 is a placeholder chosen because
# inquiries are commonly scored over roughly a 6-month window in FICO-style
# models; it is not sourced from any constraint elsewhere in this repo and
# carries no authority beyond "a plausible mock value." A real bureau
# integration would receive this from the bureau's response, not from a
# repo-wide constant.
_MOCK_INQUIRY_WINDOW_DAYS = 180


class CreditReport(BaseModel):
    """
    One applicant's credit-bureau pull.

    Two field groups, kept visually separate below:

      (a) Provenance metadata -- describes WHERE this report came from and
          WHEN. Never fed to the model; exists so a caller can log/audit which
          bureau, which FICO version, and how stale a report is, the same way
          serving/schema.py's ScoreResponse carries model_trained_at and
          calibrator_trained_at for provenance rather than silently trusting
          "the currently loaded artifact."

      (b) Credit data fields -- the values the scoring path reads from a
          pull. /score consumes fico_n today; dti_n is modeled but stays
          applicant-reported this round. These are the only two
          credit-bureau-shaped fields the shipped 8-feature model actually
          consumes (src/features.py's NUMERIC). Their bounds are imported from
          src/data_validation.py, not redeclared, so a bureau-supplied value
          and an applicant-reported ScoreRequest value (dti_n today) are held
          to the identical domain contract.
    """

    model_config = ConfigDict(extra="forbid")

    # --- (a) Provenance metadata -- not model inputs ------------------------
    applicant_id: str = Field(min_length=1)
    bureau: Literal["equifax", "experian", "transunion", "mock"]
    pulled_at: datetime
    fico_version: str = Field(min_length=1)
    # New to this module (see _MOCK_INQUIRY_WINDOW_DAYS above): no existing
    # constraint to import. A window can't be zero or negative by definition,
    # so gt=0 is basic domain sanity, not a borrowed business threshold.
    inquiry_window_days: int = Field(gt=0)

    # --- (b) Credit data fields -- model inputs (fico_n live today) ---------
    # The bound ScoreRequest.fico_n carried before Phase 1 moved the field
    # here: same imported FICO_MIN/FICO_MAX, same strict float. /score now
    # reads fico_n from this report; ScoreRequest no longer accepts the field
    # at all (serving/schema.py, extra="forbid").
    fico_n: float = Field(strict=True, ge=FICO_MIN, le=FICO_MAX)

    # Bureau-supplied, not computed here -- see the module docstring. Passed
    # through unchanged, the same way features.py's add_features() passes
    # dti_n through unmodified rather than deriving it.
    dti_n: float = Field(strict=True)

    @field_validator("dti_n")
    @classmethod
    def _dti_in_band_or_sentinel(cls, v: float) -> float:
        """
        The same band-or-sentinel logic as ScoreRequest._dti_in_band_or_sentinel
        (serving/schema.py), duplicated rather than imported (a Phase 1
        constraint -- schema.py was out of scope the round this module was
        created). Both validators are built from the same imported
        DTI_MAX_REAL / DTI_SENTINEL constants, so they cannot silently drift
        apart on WHAT the bound is, only potentially on wording -- worth
        consolidating into a shared validator in a future round.
        """
        if (0.0 <= v <= DTI_MAX_REAL) or v == DTI_SENTINEL:
            return v
        raise ValueError(
            f"dti_n outside real [0,{DTI_MAX_REAL}] band and not the known "
            f"{DTI_SENTINEL} sentinel"
        )


@runtime_checkable
class CreditBureau(Protocol):
    """
    What the /score path depends on: this interface, not a concrete
    vendor SDK. MockBureau is the only implementation today; a real
    Equifax/Experian/TransUnion client would implement this same method and
    be swappable without touching any caller.
    """

    def fetch(self, applicant_id: str) -> CreditReport:
        """Return applicant_id's credit report. Raises on a failed pull."""
        ...


class MockBureau:
    """
    A CreditBureau that never calls a real vendor.

    Deterministic by construction: fetch(applicant_id) hashes applicant_id
    into a seed, so the same applicant_id always returns a byte-identical
    CreditReport. This is the same discipline as data_loader.py's
    RANDOM_SEED = 42 -- a fixed, reproducible mapping rather than
    call-order-dependent global RNG state -- applied per-applicant instead of
    per-run, since two different tests (or two different calls in the same
    test) must see the same report for the same applicant_id without
    coordinating a shared seed.

    mean_fico / std_fico are a DRIFT-DEMO KNOB, not applicant data. One
    MockBureau instance models one snapshot of the applicant population's
    FICO distribution; constructing MockBureau(mean_fico=650) and scoring a
    batch through it is how a population-level credit-quality shift gets
    simulated, without touching a single real bureau record. 700.0 / 50.0 are
    demo assumptions about a "normal" population -- NOT measured from
    src/data_validation.py's real FICO distribution (that module's own
    comment describes it as "a tight, sane band (~612-847)") or from any
    dataset in this repo. They are chosen to be a plausible default
    center/spread for a mock, nothing more. Flagged explicitly per this
    project's "don't armchair a number silently" discipline: these ARE
    armchair values, and this docstring is the disclosure, not a citation.
    """

    def __init__(self, mean_fico: float = 700.0, std_fico: float = 50.0) -> None:
        self.mean_fico = mean_fico
        self.std_fico = std_fico

    def fetch(self, applicant_id: str) -> CreditReport:
        # SHA-256 over applicant_id, not Python's built-in hash(): hash() is
        # randomized per-process (PYTHONHASHSEED) unless disabled, which would
        # make "the same applicant_id" NOT reproduce across two test runs or
        # two processes -- exactly the failure mode determinism exists to rule
        # out. digest() gives 32 deterministic bytes to carve two independent
        # draws out of.
        digest = hashlib.sha256(applicant_id.encode("utf-8")).digest()

        # fico_n: seed a numpy Generator from the first 8 digest bytes, so the
        # SAME applicant_id always draws the SAME Normal(mean_fico, std_fico)
        # deviate. Determinism is preserved end to end -- only the SHAPE of
        # the draw changed (uniform -> normal) and its center/spread became
        # the mean_fico/std_fico knob instead of being pinned across the full
        # [FICO_MIN, FICO_MAX] band.
        fico_seed = int.from_bytes(digest[:8], "big")
        fico_rng = np.random.default_rng(fico_seed)
        fico_raw = fico_rng.normal(self.mean_fico, self.std_fico)

        # A Normal(mean, std) draw can land outside [FICO_MIN, FICO_MAX].
        # CreditReport.fico_n's pydantic constraint (ge=FICO_MIN, le=FICO_MAX,
        # imported from src/data_validation.py) would reject such a draw
        # outright and turn an unlucky applicant_id into a ValidationError.
        # Clipping keeps every applicant_id fetchable while still letting the
        # tail pile up at the boundary -- exactly the behavior a drift demo
        # wants to see once mean_fico is pushed toward FICO_MIN.
        fico_n = float(np.clip(fico_raw, FICO_MIN, FICO_MAX))

        # dti_n: UNCHANGED by this round -- still a uniform draw from the
        # hash, still bureau-supplied/passed-through rather than computed
        # (see the module docstring). This round's mandate is fico_n's
        # distribution only.
        dti_unit = int.from_bytes(digest[8:16], "big") / 2**64  # in [0, 1)
        dti_n = dti_unit * DTI_MAX_REAL

        return CreditReport(
            applicant_id=applicant_id,
            bureau="mock",
            pulled_at=_MOCK_PULLED_AT,
            fico_version=_MOCK_FICO_VERSION,
            inquiry_window_days=_MOCK_INQUIRY_WINDOW_DAYS,
            fico_n=fico_n,
            dti_n=dti_n,
        )
