"""
Tests for serving/

Doesn't re-test scoring, calibration or attribution -- tests/test_explain.py
owns those, and serving/ reaches all three THROUGH explain_applicants(). What's
locked down here is the SERVING CONTRACT, the things that are true of the
HTTP boundary and of nothing beneath it:

  1. The threshold is the one select_threshold() chose (0.25000000000000006),
     not explain.py's literal 0.25, and the two decide differently at p_cal
     exactly 0.25.
  2. ScoreResponse mirrors explain_applicants()'s dict key-for-key, PLUS exactly
     one key of its own -- `credit_report`, a ScoredCreditReport that
     explain_applicants() knows nothing about. This is the test that makes
     serving/ an adapter instead of a second implementation, for everything
     explain_applicants() is responsible for. The invariant is ADDITIVE (== the
     explain keys | {"credit_report"}), not a subtraction of permitted extras,
     so a second bureau key cannot be smuggled in alongside a constant that
     excuses it.
  3. JSON null on emp_length IS "NI" -- byte-identical responses -- and an
     unmapped string is a 422, not a silent collapse onto null's encoding.
  4. No request that validates can produce the off-manifold state
     (emp_length_ord=NaN, emp_length_missing=0), which occurs in 0 of Train's
     453,804 rows.
  5. An unseen `purpose` is rejected, not scored from LightGBM's untrained NaN
     bin.
  6. A missing field is a 422; so is an extra one (extra="forbid", the
     deliberate asymmetry with LOAN_SCHEMA's strict=False) -- including a
     client-submitted fico_n, which is real and model-consumed but no longer
     a request field at all.
  7. Floats are strict: int and float are accepted, a numeric string is not.
     Verified behavior, not assumed -- see the test.
  8. Missing artifacts at startup raise. lifespan does not swallow them. Same
     for a missing bureau client -- mirrors the artifact case exactly.
  9. A feature-contract mismatch fails closed THROUGH serving/, and so does a
     category set that has drifted from the shipped model's.
 10. A stale calibrator fails closed through serving/.
 11. The additivity guard runs on the SERVING path: corrupted contributions
     produce a 500, never a 200 with a decision. This is what proves score()
     did not take explain_applicants()'s precomputed escape hatch, which
     bypasses _shap_matrix and therefore bypasses the guard.
 12. No explainer is cached: one TreeExplainer is constructed per request.
     The concurrency hazard is invisible to a single-threaded test client, so
     the construction count is the only thing that can hold the decision in
     place.
 13. fico_n comes from CreditBureau.fetch(applicant_id), not from the
     request: the bureau is called with exactly the submitted applicant_id, the
     pull it returns surfaces in the response under `credit_report` -- the
     fetched fico_n itself, plus the bureau/fico_version/pulled_at that identify
     which pull produced it -- and the same applicant_id scores reproducibly
     across independent /score calls, the same determinism MockBureau itself
     guarantees (tests/test_bureau.py), now proven through the HTTP layer.
     The pulled dti_n is deliberately NOT in that block: the decision did not
     use it, so a block labelled "credit report" must not show it.
 14. GET /calibrator serves the shipped calibrator off the bundle /score decides
     with, threshold included, so a client draws the reject line where the
     service actually puts it.
 15. CORS admits the enumerated dev origins and nothing else -- never "*".
 16. POST /drift returns pipelines/drift_check.py's OWN PSI/KS/alarms -- the
     monitor that actually runs -- never a second implementation of them. The
     FICO knob at 700 leaves fico_n quiet and at 650 makes it fire, while dti_n
     (which the knob does not touch) stays quiet in BOTH: the negative control
     is what makes the alarm mean something.
 17. `import serving.app` pulls in NO mlflow, NO metaflow and NO pipelines.
     pyproject.toml has always CLAIMED this -- it is why the serving image can
     skip the training dependency group and stay ~937MB -- and nothing enforced
     it. /drift is the first route whose machinery lives behind that line, so
     this is where the line gets a test.

Plus, carried across the HTTP boundary from tests/test_explain.py: no
probability-scale contribution leaks into the response.

Run:  pytest tests/test_serving.py -v
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib
import shap
from fastapi.testclient import TestClient

# The drift monitor's OWN thresholds. Imported, never retyped -- the tests below
# assert "over / under the line the monitor actually uses", which is the same
# discipline tests/test_demo_drift.py holds. A hardcoded 0.25 here would be a
# second copy of the alarm rule inside the test that exists to prove there is
# only one.
from pipelines.drift_check import DEFAULT_ALARM_THRESHOLDS
import src.explain as src_explain
from src.data_validation import FICO_MAX, FICO_MIN, VALID_HOME_OWNERSHIP, VALID_PURPOSE
# The fairness audit's OWN constants -- the 0.80 rule, the bootstrap count, the
# ablation's deliberately-not-the-operating-point threshold. Imported, never
# retyped, for the same reason DEFAULT_ALARM_THRESHOLDS is: a hardcoded 0.80
# here would be a second copy of the rule, inside the test that exists to prove
# the client is handed the real one.
from src.fairness import (
    ABLATION_THRESHOLD,
    EO_THRESHOLD,
    MIN_N,
    N_BOOT,
    SWEEP_THRESHOLDS,
    WATCH_STATES,
)
from src.features import CATEGORICAL, FEATURES, TARGET, emp_order
from src.model_io import _x, load_model_artifact, train_lgb
from src.calibrate import DEFAULT_MODEL_PATH, calibrate_model
from src.explain import DEFAULT_EXPLAIN_THRESHOLD, explain_applicants

from serving.app import DRIFT_DEMO_AVAILABLE, create_app
from serving.artifacts import load_bundle
from serving.bureau import CreditBureau, CreditReport, MockBureau
from serving.config import FAIRNESS_AUDIT_PATH, SELECTED_THRESHOLD
from serving.fairness import (
    FairnessAudit,
    audit_model_trained_at,
    is_stale,
    load_fairness_audit,
)
from serving.schema import (
    EMP_LENGTH_NOT_DISCLOSED,
    VALID_EMP_LENGTH,
    ScoredCreditReport,
    ScoreRequest,
    ScoreResponse,
)

REASON_CODE_KEYS = {"rank", "feature", "value", "contribution_log_odds"}

# The ONE key /score adds to explain_applicants()'s dict. Everything the bureau
# contributes to the response lives under it (serving/schema.py's ScoreResponse).
BUREAU_SOURCED_TOP_LEVEL_KEY = "credit_report"

# ScoredCreditReport's full key set: one model input (fico_n) plus the three
# fields that identify WHICH pull produced it.
#
# Deliberately NOT named "...PROVENANCE_KEYS" -- which is exactly what the three
# loose top-level fields it replaces WERE called. fico_n is DATA, the value the
# booster actually scored on; a set holding both data and provenance cannot
# honestly be named after either half. The old constant would have had to stretch
# to cover a fourth key it does not describe, so it was removed rather than
# stretched.
CREDIT_REPORT_KEYS = {"fico_n", "bureau", "fico_version", "pulled_at"}

# One applicant that scores cleanly. Every test below mutates a copy of this.
# This is the REQUEST shape: applicant_id + six applicant-reported fields.
# fico_n is bureau-sourced now (serving/bureau.py) -- it is NOT here, and a
# client that submits its own is a 422 (test_extra_field_is_422).
GOOD = {
    "applicant_id": "applicant-0001",
    "revenue": 60_000.0,
    "dti_n": 18.0,
    "loan_amnt": 10_000.0,
    "emp_length": "5 years",
    "purpose": "debt_consolidation",
    "home_ownership_n": "RENT",
}

# The RAW MODEL INPUT shape -- what explain_applicants()/add_features()/_x()
# actually consume. Diverges from GOOD as of the bureau wiring: it carries
# fico_n (a real model feature, bureau-sourced at the HTTP layer) and drops
# applicant_id (not a model feature, only a bureau lookup key). Tests that
# call _x()/explain_applicants() directly -- bypassing the HTTP layer and its
# bureau fetch -- use this instead of GOOD.
GOOD_RAW = {k: v for k, v in GOOD.items() if k != "applicant_id"}
GOOD_RAW["fico_n"] = 700.0


def _make_split(n, rng):
    """Synthetic rows covering EVERY purpose and EVERY home_ownership value.

    test_explain.py's fixture draws 3 purposes, which is fine for explaining but
    not here: load_bundle() asserts the model's category_maps equals the request
    schema's VALID_PURPOSE (14) and VALID_HOME_OWNERSHIP (4). A 3-purpose model
    would (correctly) fail that gate. Coverage is forced, not left to sampling.
    """
    purposes, homes = sorted(VALID_PURPOSE), sorted(VALID_HOME_OWNERSHIP)
    emp_lengths = sorted(VALID_EMP_LENGTH)
    fico = rng.uniform(620, 820, n)
    dti = rng.uniform(0, 40, n)
    z = 0.05 * (dti - 20) - 0.03 * (fico - 700)
    p_default = 0.5 / (1 + np.exp(-z)) + 0.5 * 0.15
    # First len(cats) rows deterministically cover each category; rest are random.
    purpose_col = [purposes[i % len(purposes)] if i < len(purposes) else rng.choice(purposes) for i in range(n)]
    home_col = [homes[i % len(homes)] if i < len(homes) else rng.choice(homes) for i in range(n)]
    return pd.DataFrame({
        "revenue": rng.uniform(20_000, 150_000, n),
        "dti_n": dti,
        "loan_amnt": rng.uniform(1_000, 35_000, n),
        "fico_n": fico,
        "emp_length": rng.choice(emp_lengths, n),
        "purpose": purpose_col,
        "home_ownership_n": home_col,
        "addr_state": rng.choice(["CA", "TX", "NY"], n),
        "Default": rng.binomial(1, p_default),
    })


@pytest.fixture
def synthetic_splits():
    rng = np.random.default_rng(13)
    return {k: _make_split(n, rng) for k, n in
            [("train", 900), ("val", 250), ("calib", 500), ("test", 200)]}


@pytest.fixture
def artifact(synthetic_splits):
    booster, best_iteration, _m, category_maps = train_lgb(
        synthetic_splits, use_spw=False, num_boost_round=60, early_stopping_rounds=10,
    )
    return {
        "model": booster, "features": FEATURES, "categorical": CATEGORICAL,
        "category_maps": category_maps, "best_iteration": best_iteration,
        "params": {}, "trained_at": "test-fixture",
    }


@pytest.fixture
def model_path(artifact, tmp_path):
    path = tmp_path / "model.pkl"
    joblib.dump(artifact, path)
    return path


@pytest.fixture
def calibrator_path(synthetic_splits, model_path, tmp_path):
    path = tmp_path / "isotonic_calibrator.pkl"
    calibrate_model(model_path=model_path, splits=synthetic_splits, calibrator_path=path)
    return path


@pytest.fixture
def bundle(model_path, calibrator_path):
    return load_bundle(model_path=model_path, calibrator_path=calibrator_path)


@pytest.fixture
def bureau():
    return MockBureau()


@pytest.fixture
def client(bundle, bureau):
    return TestClient(create_app(bundle=bundle, bureau=bureau))


# ---------------------------------------------------------------------------
# 1. The threshold is the SELECTED one. explain.py's constant is a different
#    float, and at p_cal exactly 0.25 they decide differently.
# ---------------------------------------------------------------------------
def test_selected_threshold_is_not_the_explain_constant():
    assert SELECTED_THRESHOLD == float(np.arange(0.05, 0.95, 0.01)[20])
    assert SELECTED_THRESHOLD != DEFAULT_EXPLAIN_THRESHOLD
    assert DEFAULT_EXPLAIN_THRESHOLD == 0.25

    # The disagreement, spelled out. An applicant at p_cal exactly 0.25 is
    # REJECTED by explain.py's default and APPROVED by the threshold the
    # pipeline actually selected. Decision is `p_cal >= threshold`.
    assert 0.25 >= DEFAULT_EXPLAIN_THRESHOLD          # REJECT
    assert not (0.25 >= SELECTED_THRESHOLD)           # APPROVE


def test_service_scores_with_the_selected_threshold(client):
    body = client.post("/score", json=GOOD).json()
    assert body["threshold"] == SELECTED_THRESHOLD
    assert body["threshold"] != DEFAULT_EXPLAIN_THRESHOLD
    assert client.get("/healthz").json()["threshold"] == SELECTED_THRESHOLD


def test_decision_is_reject_iff_p_cal_at_or_above_threshold(client, synthetic_splits):
    """Both sides, over many applicants: the comparison is >= on p_calibrated.

    fico_n is bureau-sourced now, not read off the synthetic row -- the row's
    own fico_n is irrelevant to what actually gets scored, and this assertion
    doesn't need it: it only checks the response's own decision/p_calibrated/
    threshold are mutually consistent, which holds regardless of where fico_n
    came from.
    """
    seen_reject = seen_approve = False
    for i, (_, row) in enumerate(synthetic_splits["test"].head(25).iterrows()):
        payload = {k: row[k] for k in GOOD if k != "applicant_id"}
        payload["applicant_id"] = f"synthetic-{i:03d}"
        payload["revenue"] = float(payload["revenue"])
        payload["dti_n"] = float(payload["dti_n"])
        payload["loan_amnt"] = float(payload["loan_amnt"])
        body = client.post("/score", json=payload).json()
        expected = "REJECT" if body["p_calibrated"] >= body["threshold"] else "APPROVE"
        assert body["decision"] == expected
        seen_reject |= body["decision"] == "REJECT"
        seen_approve |= body["decision"] == "APPROVE"
    assert seen_reject and seen_approve, "fixture exercised only one branch"


# ---------------------------------------------------------------------------
# 2. The response cannot drift from the function it serializes.
# ---------------------------------------------------------------------------
def test_response_mirrors_explain_applicants_key_for_key(bundle, model_path, calibrator_path):
    """GOOD_RAW, not GOOD: explain_applicants() consumes the RAW model-input
    shape (has fico_n, no applicant_id), which is no longer identical to the
    REQUEST shape now that fico_n is bureau-sourced.

    ADDITIVE, not subtractive. The old form of this test subtracted a set of
    permitted bureau keys from ScoreResponse and compared the remainder -- which
    would have passed just as happily if a FOURTH bureau key were added to both
    the model and the constant excusing it. This form pins the count: everything
    on ScoreResponse is explain_applicants()'s, except exactly one key.
    """
    direct = explain_applicants(
        pd.DataFrame([GOOD_RAW]), model_path=model_path,
        calibrator_path=calibrator_path, threshold=bundle.threshold,
    )[0]
    assert set(ScoreResponse.model_fields) == set(direct) | {BUREAU_SOURCED_TOP_LEVEL_KEY}


def test_scored_credit_report_shape_is_exact():
    assert set(ScoredCreditReport.model_fields) == CREDIT_REPORT_KEYS


def test_scored_credit_report_invents_no_field_names():
    """Every key served under `credit_report` is a real field on the CreditReport
    the bureau actually returns (serving/bureau.py). This is what makes "a
    client-facing SUBSET of CreditReport" a fact rather than a claim in a
    docstring -- a field renamed or invented here cannot pass.

    It is also what forced `pulled_at` over the old top-level spelling
    `credit_report_pulled_at`: that name is not a CreditReport field, so under
    this assertion it would fail. The rename is not cosmetic; it is the price of
    being able to state the subset invariant at all.
    """
    assert CREDIT_REPORT_KEYS <= set(CreditReport.model_fields)


def test_scored_credit_report_does_not_carry_dti_n():
    """The pulled dti_n is fetched and deliberately NOT scored: _to_raw_frame
    (serving/app.py) takes dti_n from the REQUEST and never reads report.dti_n.

    Serving it inside a block labelled "credit report" would show a client a
    bureau DTI beside a decision computed from the applicant's self-reported one
    -- under MockBureau, a number drawn uniformly from [0, 1000) beside a
    decision made on a self-reported 18 (docs/data-decisions.md's "move dti_n to
    the bureau" entry measures the mock's distribution).

    The second assertion is the load-bearing one: dti_n EXISTS on CreditReport,
    so its absence from ScoredCreditReport is provably a choice and not an
    oversight. If someone ever drops dti_n from CreditReport entirely, this test
    goes red and forces them to read why it was omitted here.
    """
    assert "dti_n" not in ScoredCreditReport.model_fields
    assert "dti_n" in CreditReport.model_fields


def test_reason_code_key_set_is_exact(client):
    for rc in client.post("/score", json=GOOD).json()["reason_codes"]:
        assert set(rc) == REASON_CODE_KEYS


# ---------------------------------------------------------------------------
# 3. JSON null IS "NI". An unmapped string is not.
# ---------------------------------------------------------------------------
def test_null_emp_length_is_identical_to_not_disclosed(client):
    as_null = client.post("/score", json={**GOOD, "emp_length": None})
    as_ni = client.post("/score", json={**GOOD, "emp_length": EMP_LENGTH_NOT_DISCLOSED})
    assert as_null.status_code == as_ni.status_code == 200
    assert as_null.json() == as_ni.json()


def test_null_emp_length_differs_from_a_disclosed_tenure(client):
    """Normalizing null to "NI" is not a no-op: it changes the score."""
    ni = client.post("/score", json={**GOOD, "emp_length": None}).json()
    disclosed = client.post("/score", json={**GOOD, "emp_length": "5 years"}).json()
    assert ni["p_raw"] != disclosed["p_raw"]


def test_unmapped_emp_length_is_rejected_not_collapsed(client):
    """"bogus" encodes identically to an un-normalized null. It must not arrive."""
    assert client.post("/score", json={**GOOD, "emp_length": "bogus"}).status_code == 422
    assert client.post("/score", json={**GOOD, "emp_length": "5 years"}).status_code == 200


def test_normalization_happens_before_the_frame_is_built(client):
    assert ScoreRequest(**{**GOOD, "emp_length": None}).emp_length == EMP_LENGTH_NOT_DISCLOSED


# ---------------------------------------------------------------------------
# 4. The off-manifold state is unreachable from any request that validates.
#    (emp_length_ord=NaN, emp_length_missing=0) occurs in 0 of Train's 453,804
#    rows; an un-normalized null and any unmapped string both produce it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("emp", [None, *sorted(VALID_EMP_LENGTH)])
def test_no_validated_request_can_encode_off_manifold(emp):
    """request.model_dump() no longer carries fico_n (bureau-sourced now), but
    _x() needs it -- merge in a fixed valid value, mirroring what /score's
    _to_raw_frame does with a real bureau fetch."""
    request = ScoreRequest(**{**GOOD, "emp_length": emp})
    raw = {**request.model_dump(exclude={"applicant_id"}), "fico_n": 700.0}
    engineered = _x(pd.DataFrame([raw]))
    ord_, missing = engineered.iloc[0]["emp_length_ord"], engineered.iloc[0]["emp_length_missing"]
    assert not (pd.isna(ord_) and missing == 0), f"{emp!r} encoded off-manifold"
    # And the only way to get ord=NaN is to have declared non-disclosure.
    if pd.isna(ord_):
        assert missing == 1


def test_the_off_manifold_state_is_what_we_claim_it_is():
    """Guard the premise: an UNVALIDATED null really does encode (NaN, 0).

    GOOD_RAW, not GOOD: this calls _x() directly, which needs the RAW
    model-input shape (has fico_n), not the request shape.
    """
    engineered = _x(pd.DataFrame([{**GOOD_RAW, "emp_length": None}]))
    assert pd.isna(engineered.iloc[0]["emp_length_ord"])
    assert engineered.iloc[0]["emp_length_missing"] == 0
    # ... and so does an arbitrary string. Indistinguishable.
    bogus = _x(pd.DataFrame([{**GOOD_RAW, "emp_length": "bogus"}]))
    assert pd.isna(bogus.iloc[0]["emp_length_ord"])
    assert bogus.iloc[0]["emp_length_missing"] == 0


# ---------------------------------------------------------------------------
# 5. An unseen `purpose` is rejected, not scored from an untrained NaN bin.
# ---------------------------------------------------------------------------
def test_unseen_purpose_is_422(client):
    assert client.post("/score", json={**GOOD, "purpose": "cryptocurrency_yolo"}).status_code == 422


def test_every_known_purpose_is_200(client):
    for purpose in sorted(VALID_PURPOSE):
        assert client.post("/score", json={**GOOD, "purpose": purpose}).status_code == 200


def test_unknown_home_ownership_is_422(client):
    assert client.post("/score", json={**GOOD, "home_ownership_n": "SQUAT"}).status_code == 422
    assert client.post("/score", json={**GOOD, "home_ownership_n": "OWN"}).status_code == 200


# ---------------------------------------------------------------------------
# 6. Missing field -> 422. Extra field -> 422. The second is the deliberate
#    asymmetry with LOAN_SCHEMA's strict=False (data_validation.py).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dropped", sorted(GOOD))
def test_missing_field_is_422(client, dropped):
    assert client.post("/score", json={k: v for k, v in GOOD.items() if k != dropped}).status_code == 422


def test_extra_field_is_422(client):
    assert client.post("/score", json={**GOOD, "zip_code": "021xx"}).status_code == 422
    assert client.post("/score", json={**GOOD, "addr_state": "CA"}).status_code == 422
    # fico_n is real and model-consumed, but no longer a request field: the
    # service sources it from the bureau now, and a client that submits its
    # own gets the same 422 as any other unrecognized field.
    assert client.post("/score", json={**GOOD, "fico_n": 700.0}).status_code == 422
    assert client.post("/score", json=GOOD).status_code == 200


# ---------------------------------------------------------------------------
# 7. Floats are strict. The behavior is NAMED for all three inputs, because it
#    was measured (pydantic 2.13.4) rather than assumed.
# ---------------------------------------------------------------------------
def test_strict_floats_name_the_behavior_for_all_three_inputs(client):
    """fico_n moved to the bureau and can no longer carry this contract test
    (it isn't a request field at all). revenue is still applicant-reported
    and still Field(strict=True), so it takes over as the strict-float
    example."""
    assert client.post("/score", json={**GOOD, "revenue": 60_000.0}).status_code == 200  # float
    assert client.post("/score", json={**GOOD, "revenue": 60_000}).status_code == 200    # int, widened
    assert client.post("/score", json={**GOOD, "revenue": "60000"}).status_code == 422   # str, rejected


def test_strict_floats_reject_bool_which_lax_mode_would_call_1_point_0(client):
    assert client.post("/score", json={**GOOD, "revenue": True}).status_code == 422


def test_out_of_band_numerics_are_422(client):
    """fico_n's out-of-band case moved out of this test: it is no longer a
    request field, so a client can no longer submit an out-of-band value for
    it at all (test_extra_field_is_422 covers the now-relevant behavior --
    submitting fico_n in any form is 422, regardless of value)."""
    assert client.post("/score", json={**GOOD, "loan_amnt": 1e9}).status_code == 422   # > LOAN_MAX
    assert client.post("/score", json={**GOOD, "dti_n": -1.0}).status_code == 422      # negative
    assert client.post("/score", json={**GOOD, "dti_n": 999.0}).status_code == 200     # the sentinel


# ---------------------------------------------------------------------------
# 8. Missing artifacts at startup raise. Nothing swallows them.
# ---------------------------------------------------------------------------
def test_missing_artifacts_raise_at_load(tmp_path, calibrator_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(model_path=tmp_path / "absent.pkl", calibrator_path=calibrator_path)


def test_present_artifacts_load(model_path, calibrator_path):
    assert load_bundle(model_path=model_path, calibrator_path=calibrator_path) is not None


def test_score_is_503_when_the_bundle_is_absent(bundle):
    """Reachable only without startup -- which is exactly this. See serving/errors.py.

    bureau is injected here so the ONLY missing piece is the bundle -- without
    it, app.state.bureau would also be unset (create_app(bundle=bundle) alone
    skips lifespan entirely), and the 503 below would be ambiguous about which
    dependency actually caused it.
    """
    app = create_app(bundle=bundle, bureau=MockBureau())
    app.state.bundle = None
    unloaded = TestClient(app)
    assert unloaded.post("/score", json=GOOD).status_code == 503
    assert unloaded.get("/healthz").status_code == 503


def test_score_is_503_when_the_bureau_is_absent(bundle):
    """The bureau side of the same reachable-only-in-tests unloaded state."""
    app = create_app(bundle=bundle, bureau=MockBureau())
    app.state.bureau = None
    unloaded = TestClient(app)
    assert unloaded.post("/score", json=GOOD).status_code == 503


# ---------------------------------------------------------------------------
# 9. A feature-contract mismatch fails closed through serving/. So does a
#    category set that has drifted from the shipped model's.
# ---------------------------------------------------------------------------
def test_feature_contract_mismatch_fails_closed(artifact, calibrator_path, tmp_path):
    bad = dict(artifact)
    bad["features"] = artifact["features"] + ["addr_state"]
    path = tmp_path / "mismatched.pkl"
    joblib.dump(bad, path)
    with pytest.raises(ValueError, match="[Cc]ontract mismatch"):
        load_bundle(model_path=path, calibrator_path=calibrator_path)


def test_category_drift_fails_closed(artifact, calibrator_path, tmp_path):
    """The model knows 13 purposes, the schema admits 14 -> refuse to serve."""
    drifted = dict(artifact)
    maps = dict(artifact["category_maps"])
    maps["purpose"] = pd.Index(sorted(VALID_PURPOSE)[:-1])
    drifted["category_maps"] = maps
    path = tmp_path / "drifted.pkl"
    joblib.dump(drifted, path)
    with pytest.raises(ValueError, match="category mismatch"):
        load_bundle(model_path=path, calibrator_path=calibrator_path)


def test_matching_categories_do_not_raise(model_path, calibrator_path):
    assert load_bundle(model_path=model_path, calibrator_path=calibrator_path) is not None


# ---------------------------------------------------------------------------
# 10. A stale calibrator fails closed through serving/.
# ---------------------------------------------------------------------------
def test_stale_calibrator_fails_closed(artifact, model_path, calibrator_path, tmp_path):
    stale = dict(artifact)
    stale["trained_at"] = "a-different-model"
    path = tmp_path / "retrained.pkl"
    joblib.dump(stale, path)
    with pytest.raises(ValueError, match="[Ss]tale calibrator"):
        load_bundle(model_path=path, calibrator_path=calibrator_path)


def test_matching_calibrator_does_not_raise(model_path, calibrator_path):
    assert load_bundle(model_path=model_path, calibrator_path=calibrator_path) is not None


# ---------------------------------------------------------------------------
# 11. The additivity guard runs on the SERVING path. Corrupted contributions
#     produce a 500, never a 200 with a decision.
#
#     This is what proves score() did not take explain_applicants()'s
#     precomputed escape hatch (explain.py), which skips _shap_matrix and
#     therefore skips _assert_additivity entirely.
# ---------------------------------------------------------------------------
_REAL_TREE_EXPLAINER = shap.TreeExplainer


class _CorruptingExplainer:
    """Real contributions, one cell perturbed. base + sum(sv) != margin."""

    def __init__(self, booster):
        self._inner = _REAL_TREE_EXPLAINER(booster)

    @property
    def expected_value(self):
        return self._inner.expected_value

    def shap_values(self, X, **kw):
        sv = np.array(self._inner.shap_values(X, **kw), dtype=float)
        sv[0, 0] += 1.0
        return sv


def test_corrupted_contributions_return_500_not_a_decision(client, monkeypatch):
    monkeypatch.setattr(shap, "TreeExplainer", _CorruptingExplainer)
    response = client.post("/score", json=GOOD)
    assert response.status_code == 500
    assert "decision" not in response.json()


def test_honest_contributions_return_200(client):
    """The other side: unpatched, the same request scores."""
    response = client.post("/score", json=GOOD)
    assert response.status_code == 200
    assert response.json()["decision"] in {"REJECT", "APPROVE"}


def test_the_500_body_leaks_nothing_about_the_model(client, monkeypatch):
    monkeypatch.setattr(shap, "TreeExplainer", _CorruptingExplainer)
    detail = client.post("/score", json=GOOD).json()["detail"]
    for leak in ("margin", "contribution", "atol", "1e-09", "log_odds"):
        assert leak not in detail.lower()


def test_without_the_guard_the_same_corruption_returns_a_wrong_decision(client, monkeypatch):
    """The counterfactual, so the 500 above cannot pass for the wrong reason.

    Neuter _assert_additivity and the corrupted explainer scores happily: 200,
    with raw_margin_log_odds shifted by exactly the +1.0 injected into the
    contributions, and a decision derived from it. Nothing else in the stack
    notices. The guard is not one check among several -- it is the only thing
    between a corrupted explanation and an applicant's answer.
    """
    clean = client.post("/score", json=GOOD).json()["raw_margin_log_odds"]

    monkeypatch.setattr(shap, "TreeExplainer", _CorruptingExplainer)
    monkeypatch.setattr(src_explain, "_assert_additivity", lambda *a, **k: None)

    response = client.post("/score", json=GOOD)
    assert response.status_code == 200
    assert response.json()["raw_margin_log_odds"] == pytest.approx(clean + 1.0)


# ---------------------------------------------------------------------------
# 12. No explainer is cached. One construction per request.
#
#     expected_value is instance state mutated by shap_values() (inside
#     TreeExplainer.shap_values). The hazard is invisible to a single-threaded test client,
#     so the construction COUNT is the only thing that can hold this decision
#     in place against a future "obvious" optimization.
# ---------------------------------------------------------------------------
def test_one_explainer_is_constructed_per_request(client, monkeypatch):
    constructions = []

    def _counting(booster, *args, **kwargs):
        constructions.append(booster)
        return _REAL_TREE_EXPLAINER(booster, *args, **kwargs)

    monkeypatch.setattr(shap, "TreeExplainer", _counting)
    for _ in range(3):
        assert client.post("/score", json=GOOD).status_code == 200
    assert len(constructions) == 3


# ---------------------------------------------------------------------------
# 13. No probability-scale contribution leaks across the HTTP boundary.
#     docs/explainability.md Section 5 proves percentage-point attribution is
#     undefined under the shipped calibrator. tests/test_explain.py keeps that
#     proof load-bearing in the library; this keeps it load-bearing on the wire.
# ---------------------------------------------------------------------------
def test_no_probability_scale_contribution_leaks(client):
    body = client.post("/score", json=GOOD).json()
    assert "contribution_to_probability" not in body
    assert body["scale"] == "log_odds_margin"
    for rc in body["reason_codes"]:
        assert "contribution_to_probability" not in rc
        assert set(rc) == REASON_CODE_KEYS


def test_probability_fields_that_do_belong_are_predictions_not_attributions(client):
    body = client.post("/score", json=GOOD).json()
    assert {"p_raw", "p_calibrated"} <= set(body)
    assert all(k.endswith("_log_odds") for k in
               ("base_value_log_odds", "raw_margin_log_odds", "contributions_log_odds"))


# ---------------------------------------------------------------------------
# 14. fico_n comes from CreditBureau.fetch(applicant_id), not the request.
#     The bureau is called with the submitted applicant_id, its provenance
#     surfaces in the response, and the same applicant_id scores
#     reproducibly across independent /score calls.
# ---------------------------------------------------------------------------
class _SpyBureau:
    """Wraps MockBureau, recording every applicant_id fetch() was called with."""

    def __init__(self):
        self._inner = MockBureau()
        self.calls: list[str] = []

    def fetch(self, applicant_id: str) -> CreditReport:
        self.calls.append(applicant_id)
        return self._inner.fetch(applicant_id)


def test_bureau_is_called_with_the_requests_applicant_id(bundle):
    spy = _SpyBureau()
    client = TestClient(create_app(bundle=bundle, bureau=spy))
    assert client.post("/score", json=GOOD).status_code == 200
    assert spy.calls == [GOOD["applicant_id"]]


def test_response_carries_the_pull_it_scored_on(client, bureau):
    """Renamed from test_response_carries_bureau_provenance: it now guards more
    than provenance. The fetched fico_n -- the credit value the booster actually
    consumed -- travels with the three fields identifying which pull supplied it,
    and the whole thing arrives under one nested key.

    The fico_n assertion is the new one, and it is the reason the nesting exists:
    before this, a client was told WHICH report was pulled but never what was IN
    it, so it could not check the decision against the data the decision used.
    """
    report = bureau.fetch(GOOD["applicant_id"])
    cr = client.post("/score", json=GOOD).json()["credit_report"]

    assert cr["fico_n"] == report.fico_n
    assert cr["bureau"] == report.bureau == "mock"
    assert cr["fico_version"] == report.fico_version
    # datetime round-trips through JSON as an ISO string (pydantic v2 emits a
    # "Z" suffix for UTC rather than "+00:00" -- parse both back to compare
    # the actual instant rather than the string spelling of it).
    assert datetime.fromisoformat(cr["pulled_at"]) == report.pulled_at


def test_same_applicant_id_scores_reproducibly_through_http(client):
    """The determinism MockBureau itself guarantees (tests/test_bureau.py),
    proven end to end through the HTTP layer: two independent /score calls
    for the same applicant_id must be byte-identical."""
    first = client.post("/score", json=GOOD).json()
    second = client.post("/score", json=GOOD).json()
    assert first == second


def test_different_applicant_ids_can_score_differently(client):
    """Sanity that fico_n genuinely varies with applicant_id through the HTTP
    layer, not a constant silently plumbed through regardless of the bureau
    call. Not a claim about specific values -- just that ten distinct
    applicants don't all collapse to one margin."""
    margins = {
        client.post(
            "/score", json={**GOOD, "applicant_id": f"applicant-{i:04d}"},
        ).json()["raw_margin_log_odds"]
        for i in range(10)
    }
    assert len(margins) > 1


def test_blank_applicant_id_is_422(client):
    assert client.post("/score", json={**GOOD, "applicant_id": ""}).status_code == 422


# ---------------------------------------------------------------------------
# The shipped artifacts. Skipped, not failed, when they are absent.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="shipped model artifact absent")
def test_the_shipped_artifact_passes_every_startup_gate():
    """The 14 purposes and 4 ownership values the schema admits are exactly the
    ones the shipped model was fit on. Nothing had ever compared them."""
    shipped = load_bundle()
    assert set(shipped.category_maps["purpose"]) == set(VALID_PURPOSE)
    assert set(shipped.category_maps["home_ownership_n"]) == set(VALID_HOME_OWNERSHIP)
    assert shipped.threshold == SELECTED_THRESHOLD
    assert shipped.model_trained_at is not None


# ---------------------------------------------------------------------------
# 14. GET /calibrator serves the SHIPPED calibrator's own shape, off the bundle
#     /score already decides with -- not a second read, and not a snapshot.
#
#     The point of the endpoint is that a client drawing the step function draws
#     THE one in production. So the tests that matter are identity tests (served
#     arrays ARE bundle.calibrator's) and threshold-agreement tests (the line the
#     client draws is the line /score decides at), not shape trivia.
# ---------------------------------------------------------------------------
def test_calibrator_serves_the_loaded_calibrator_itself(client, bundle):
    """Served arrays are the loaded IsotonicRegression's, value for value.

    Holds for ANY bundle, including this fixture's synthetic one -- which is the
    reason it is asserted against `bundle` rather than against 104/52. A test
    that hardcoded the shipped numbers here would pass for the wrong reason the
    day the fixture's synthetic calibrator happened to land on them.
    """
    body = client.get("/calibrator").json()
    cal = bundle.calibrator

    assert body["x_thresholds"] == cal.X_thresholds_.tolist()
    assert body["y_thresholds"] == cal.y_thresholds_.tolist()
    assert body["x_min"] == float(cal.X_min_)
    assert body["x_max"] == float(cal.X_max_)
    assert body["n_knots"] == len(cal.X_thresholds_)
    assert body["n_distinct_y"] == len(set(cal.y_thresholds_.tolist()))


def test_calibrator_threshold_is_the_one_score_decides_at(client):
    """The reject line the client draws is the line the service actually uses.

    /calibrator's threshold must equal the threshold /score puts in its own
    response -- not DEFAULT_EXPLAIN_THRESHOLD (explain.py), which is the literal
    0.25 and a different float (SELECTED_THRESHOLD's comment, serving/config.py).
    A frontend that guessed 0.25 would draw a boundary the service does not
    decide at; this is the test that makes guessing unnecessary.
    """
    drawn = client.get("/calibrator").json()["threshold"]
    decided = client.post("/score", json=GOOD).json()["threshold"]
    assert drawn == decided


def test_calibrator_is_503_when_the_bundle_is_absent(bundle):
    """Depends(get_bundle) is load-bearing: no bundle, no curve. Same reachable-
    only-in-tests unloaded state as /score and /healthz (serving/errors.py)."""
    app = create_app(bundle=bundle, bureau=MockBureau())
    app.state.bundle = None
    assert TestClient(app).get("/calibrator").status_code == 503


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="shipped model artifact absent")
def test_the_shipped_calibrator_is_a_52_level_step_function_over_104_knots():
    """The numbers docs/explainability.md Section 4 reasons from, asserted against
    the artifact the service would actually load -- so the document and the
    shipped pickle cannot drift apart silently.

    n_knots == 2 * n_distinct_y is the structure, not a coincidence: the knots
    come in equal-valued PAIRS, one pair per flat block.
    """
    shipped = load_bundle()
    body = TestClient(create_app(bundle=shipped, bureau=MockBureau())).get("/calibrator").json()

    assert body["n_knots"] == 104
    assert body["n_distinct_y"] == 52
    assert body["n_knots"] == 2 * body["n_distinct_y"]
    assert body["x_min"] == float(shipped.calibrator.X_min_)
    assert body["x_max"] == float(shipped.calibrator.X_max_)
    assert body["threshold"] == SELECTED_THRESHOLD


# ---------------------------------------------------------------------------
# 15. CORS admits the browser frontend, and ONLY the enumerated dev origins.
#     "*" is not a wildcard this service is allowed to ship (CORS_ALLOW_ORIGINS,
#     serving/app.py).
# ---------------------------------------------------------------------------
def test_cors_admits_the_vite_dev_origin(client):
    r = client.options(
        "/score",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_does_not_admit_an_unlisted_origin(client):
    """The test that would fail if anyone "simplified" allow_origins to ["*"]."""
    r = client.options(
        "/score",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in r.headers


# ---------------------------------------------------------------------------
# 16. POST /drift -- the drift demo's FICO knob, over HTTP.
#
#     The endpoint's entire claim is that it is a WRAPPER: the PSI on the wire is
#     the PSI pipelines/drift_check.py computes, not one the serving layer worked
#     out for itself. So the tests that matter are (a) the numbers are the
#     monitor's, proven by identity and by interception, and (b) the demo's
#     BEHAVIOR survives the HTTP boundary -- quiet at 700, firing at 650, with
#     dti_n silent throughout.
#
#     The thresholds are never restated here. Every assertion below is tested
#     against DEFAULT_ALARM_THRESHOLDS, imported from the monitor, for the same
#     reason tests/test_demo_drift.py does it: an exact PSI hardcoded into a test
#     is a number that has to be re-typed the day the mock changes, and re-typing
#     is how the test and the thing it tests stop agreeing.
# ---------------------------------------------------------------------------
def test_drift_at_700_leaves_fico_quiet(client):
    body = client.post("/drift", json={"mean_fico": 700.0}).json()
    fico = body["features"]["fico_n"]
    assert fico["psi"] < DEFAULT_ALARM_THRESHOLDS["psi"]
    assert fico["ks"] < DEFAULT_ALARM_THRESHOLDS["ks"]
    assert fico["alarmed"] is False


def test_drift_at_650_makes_fico_fire(client):
    body = client.post("/drift", json={"mean_fico": 650.0}).json()
    fico = body["features"]["fico_n"]
    assert fico["psi"] > DEFAULT_ALARM_THRESHOLDS["psi"]
    assert fico["ks"] > DEFAULT_ALARM_THRESHOLDS["ks"]
    assert fico["alarmed"] is True


def test_dti_is_an_unmoved_negative_control_at_both_settings(client):
    """The one that makes the other two mean anything.

    MockBureau seeds dti_n off the applicant_id hash and never off mean_fico, and
    both settings draw the same applicant ids -- so dti_n is byte-identical at 700
    and at 650. It must therefore report the SAME psi/ks at both, and stay quiet
    at both. A monitor that lit up on dti_n too would not be detecting drift; it
    would be detecting that something, somewhere, changed.

    This is also the test that would have caught the tripwire trap. drift_check's
    dti tripwire alarm names "dti_n" inside its explanatory prose, so deriving
    `alarmed` by searching the alarm text for the column name would report the
    negative control as FIRING at every setting of the knob -- and the demo's
    central contrast would quietly invert.
    """
    quiet = client.post("/drift", json={"mean_fico": 700.0}).json()["features"]["dti_n"]
    shifted = client.post("/drift", json={"mean_fico": 650.0}).json()["features"]["dti_n"]

    assert quiet == shifted                    # the knob did not touch it, at all
    for dti in (quiet, shifted):
        assert dti["psi"] < DEFAULT_ALARM_THRESHOLDS["psi"]
        assert dti["ks"] < DEFAULT_ALARM_THRESHOLDS["ks"]
        assert dti["alarmed"] is False


def test_the_same_knob_setting_returns_the_same_numbers(client):
    """Deterministic end to end: fixed applicant ids, hash-seeded MockBureau, no
    RNG anywhere on the path. Two identical requests must be byte-identical
    responses -- which is what lets a client drag a slider and get a curve rather
    than a shimmer, and what lets a demo be repeated exactly."""
    first = client.post("/drift", json={"mean_fico": 650.0}).json()
    second = client.post("/drift", json={"mean_fico": 650.0}).json()
    assert first == second


def test_the_endpoint_computes_nothing_it_calls_the_real_monitor(client, monkeypatch):
    """THE test. Everything else here would still pass against a serving-layer
    reimplementation of PSI that happened to agree; this one would not.

    Two claims, in order. First, the functions the demo module holds ARE
    drift_check's -- identity, not same-named copies that could drift apart.
    Second, the endpoint actually goes THROUGH them: drift_metrics is wrapped in a
    spy, and if the response's PSI can be produced without that spy being called,
    then some other code computed it.
    """
    import pipelines.drift_check as monitor
    import scripts.demo_drift as demo

    assert demo.drift_metrics is monitor.drift_metrics
    assert demo.evaluate_alarms is monitor.evaluate_alarms

    calls: list[tuple] = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return monitor.drift_metrics(*args, **kwargs)

    monkeypatch.setattr(demo, "drift_metrics", spy)
    body = client.post("/drift", json={"mean_fico": 650.0}).json()

    assert calls, "POST /drift produced a PSI without calling drift_check.drift_metrics"
    assert body["features"]["fico_n"]["alarmed"] is True


def test_the_wire_numbers_are_the_demo_scripts_numbers(client):
    """The endpoint and `uv run python scripts/demo_drift.py` cannot disagree,
    because they are one code path. Asserted against drift_report() directly --
    the same function the handler calls -- so a divergence could only come from
    the HTTP layer reshaping something on the way out.
    """
    from scripts.demo_drift import drift_report

    body = client.post("/drift", json={"mean_fico": 650.0}).json()
    report = drift_report(650.0)

    assert body["metrics"] == pytest.approx(report["metrics"])
    assert body["alarms"] == report["alarms"]
    assert body["features"]["fico_n"]["psi"] == pytest.approx(report["features"]["fico_n"]["psi"])


def test_the_per_feature_view_is_a_rekeying_of_the_raw_metrics(client):
    """`features` is a convenience, and the raw `metrics` it was re-keyed FROM
    ships beside it -- which is what makes the convenience checkable instead of
    trusted. If they ever disagreed, the flat dict is the one drift_check
    actually produced."""
    body = client.post("/drift", json={"mean_fico": 650.0}).json()
    year = body["current_year"]

    for column, feature in body["features"].items():
        assert feature["psi"] == pytest.approx(body["metrics"][f"psi_{column}_{year}"])
        assert feature["ks"] == pytest.approx(body["metrics"][f"ks_{column}_{year}"])


def test_the_thresholds_are_the_monitors_own_and_arrive_unfiltered(client):
    """A client draws the alarm line where the monitor actually trips -- the same
    discipline that makes /calibrator ship the decision threshold rather than let
    a client guess 0.25.

    Unfiltered, too: the dti-specific sentinel/tripwire/calib_gap thresholds are
    returned alongside psi/ks, so the client sees the whole rule set the monitor
    ran under and not just the two lines that flatter the demo.
    """
    body = client.post("/drift", json={"mean_fico": 700.0}).json()
    assert body["thresholds"] == pytest.approx(DEFAULT_ALARM_THRESHOLDS)


def test_the_dti_tripwire_artifact_is_disclosed_not_filtered_out(client):
    """The alarm that embarrasses the demo is left in the response.

    MockBureau's dti_n is a uniform draw over [0, 1000), so ~90% of it lands in
    drift_check's (100, 1000] tripwire band and the tripwire alarm fires at EVERY
    setting of the knob. It is a mock artifact, not drift -- it reads identically
    in the control and the downturn, which is exactly why it is not what the FICO
    knob moves. Dropping it from `alarms` would make the demo look cleaner and
    would be the one dishonest edit available here.
    """
    for mean_fico in (700.0, 650.0):
        alarms = client.post("/drift", json={"mean_fico": mean_fico}).json()["alarms"]
        assert any(a.startswith("tripwire_share_") for a in alarms)


def test_the_knob_is_bounded_to_the_fico_band(client):
    """mean_fico is the centre of a FICO distribution, so it is bounded by the
    FICO band -- imported from src/data_validation.py, not retyped. Outside it
    MockBureau's clipping piles the whole population onto one boundary value, and
    the PSI computed off that spike would look like a dramatic finding and mean
    nothing."""
    assert client.post("/drift", json={"mean_fico": FICO_MIN - 1}).status_code == 422
    assert client.post("/drift", json={"mean_fico": FICO_MAX + 1}).status_code == 422
    assert client.post("/drift", json={"mean_fico": "650"}).status_code == 422   # strict float
    assert client.post("/drift", json={"mean_fico": 650.0, "n": 10}).status_code == 422  # forbid


def test_drift_is_absent_not_broken_when_the_demo_is_not_mounted(bundle):
    """The production container's shape: scripts/ and pipelines/ are not in the
    image, so DRIFT_DEMO_AVAILABLE is False there and /drift simply is not a route.

    404, not 500. A route that exists and cannot work is worse than a route that
    does not exist -- the first is a service lying about what it is.
    """
    app = create_app(bundle=bundle, bureau=MockBureau(), drift_demo=False)
    assert TestClient(app).post("/drift", json={"mean_fico": 650.0}).status_code == 404


# ---------------------------------------------------------------------------
# 17. The serving import graph. pyproject.toml claims mlflow, metaflow and
#     seaborn "never appear in sys.modules after `import serving.app`". Nothing
#     enforced it. This does.
#
#     WHAT THIS IS: a regression roster. Six names, each one a real leak that
#     really happened, kept so it cannot happen twice --
#
#         mlflow, metaflow, seaborn  the training group (341418f, 79572b1)
#         pipelines, scripts.demo_drift  /drift's handler import
#         src.fairness               GET /fairness
#
#     WHAT THIS IS NOT: a bound on the image. An earlier version of this comment
#     said the sys.modules claim "is why the image is ~937MB instead of ~2.6GB."
#     That conflates two properties. The image's size is decided by what
#     `uv sync --frozen --no-dev --no-group training` INSTALLS (Dockerfile) and
#     what the COPY lines put in it. This test measures what `import serving.app`
#     LOADS. The second is why the container does not die at boot; the first is
#     why it is 937MB. Both are real, they are not the same, and this test bounds
#     only the second.
#
#     That distinction is not pedantry -- it is the gap, and it is the one that
#     matters next. A dependency can be DECLARED in pyproject.toml and imported
#     only inside a route handler: `uv sync` installs it, the image grows, and
#     this graph does not move by a single byte. That is not a hypothetical, it
#     is /drift's own pattern (serving/app.py's handler-level import), built on
#     purpose and defended below. So this test is structurally blind to that
#     shape -- BY DESIGN, and it is not a defect in the test. It is the reason
#     the test's name is what it is: no TRAINING DEPENDENCY, not "serving stays
#     lean." Anyone reading it as the latter is reading the comment, not the
#     assertion. Two of us did. See docs/data-decisions.md's entry on the 26
#     reached-and-tolerated packages for what that misreading cost and what the
#     graph actually contains.
#
#     /drift is the first route whose machinery lives on the far side of that
#     line (scripts/demo_drift.py -> pipelines/drift_check.py -> mlflow, and
#     -> pipelines/training_flow.py -> metaflow). Importing drift_report at module
#     scope, or even inside create_app(), would drag all three in -- `app =
#     create_app()` runs at import. Measured: it does, and it costs 1.3s. The
#     handler-level import in serving/app.py is what keeps this test true, and
#     this test is what stops someone "tidying" that import up to the top of the
#     file and silently breaking the container six weeks later.
#
#     A subprocess, because sys.modules in-process is already polluted by every
#     other test that imported pipelines. A fresh interpreter is the only place
#     the question can be asked honestly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _training_group() -> frozenset[str]:
    """The training group, DERIVED from pyproject.toml -- never transcribed.

    This function exists because the transcription was wrong and stayed wrong.
    The watched list used to be a literal tuple carrying three of this group's
    four members. The fourth, matplotlib, was the one that leaks -- so the test
    named "pulls in no training dependency" passed while a training dependency
    was in sys.modules. A hand-copied set cannot go stale loudly: it is only as
    honest as the day someone typed it, and nothing tells you the day it stopped
    being true. Reading the group is what makes the name checkable at all.

    Same move as tests/test_readme.py pinning the live collected count instead of
    a literal, one layer over: iterate the definition, do not restate it.

    Distribution name == module name for all four of today's members. That
    coincidence is doing real work here and it is not guaranteed in general
    (scikit-learn imports as sklearn). If a future member breaks it, this watches
    a name that can never appear and passes vacuously -- the failure would be
    silent, which is the same shape of defect this function was written to kill.
    Left underived on purpose: the fix is importlib.metadata.packages_distributions(),
    which answers only for INSTALLED packages, so it would make the guard depend
    on the environment it runs in. Noted, not built.
    """
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        requirements = tomllib.load(fh)["dependency-groups"]["training"]
    return frozenset(re.split(r"[<>=!~\[; ]", req)[0].strip() for req in requirements)


# The first-party three are transcribed and stay transcribed: `pipelines`,
# `src.fairness` and `scripts.demo_drift` are import paths in this repo, not
# entries in a dependency table. There is no definition to iterate -- the
# directory listing is not one, since watching every first-party package would
# watch `src` and `serving` themselves. Each is here because it leaked once.
_FIRST_PARTY_LEAKS = frozenset({"pipelines", "src.fairness", "scripts.demo_drift"})

# THE EXCEPTION, WRITTEN WHERE IT RUNS.
#
# matplotlib is in the training group AND in serving's import graph. Both facts
# are true, and together they are not a contradiction -- they are a tolerance:
#
#     serving/app.py -> serving/artifacts.py -> lightgbm -> lightgbm.compat
#     lightgbm.compat:  try: import matplotlib / except ImportError:
#                       MATPLOTLIB_INSTALLED = False   (degrades to no plotting)
#
# So serving REACHES matplotlib here, where the training group is installed, and
# TOLERATES its absence in the image, where it is not. lightgbm runs either way,
# and nothing on the scoring path plots. Reached and tolerated -- not unreached.
# That is why matplotlib stays out of [project] dependencies while mlflow,
# metaflow and seaborn stay out of the graph entirely: a different reason for the
# same shelf.
#
# This is the FIRST TIME that exception exists anywhere that executes. It was
# already written in prose five times over, in four files, and not one of them
# runs: pyproject.toml:28-34 and again at :79-84, docs/design.md's serving-layer
# section, docs/data-decisions.md's "Why 'serving never reaches matplotlib' was
# false" entry, and README.md's serving-layer section. All five are correct. All
# five are narration. The transcription this replaces was green for exactly as
# long as the tolerance lived only in prose, because prose cannot be the thing a
# subprocess disagrees with.
#
# Hand-written on purpose, and it cannot be derived: "which imports does lightgbm
# wrap in try/except" is a fact about lightgbm's source, not a row in any table
# this repo owns. A decision, not a definition. So it gets the treatment a
# transcription cannot have -- it is checked for staleness below. If matplotlib
# ever stops being reached (lightgbm drops the import, or artifacts.py stops
# importing lightgbm), this set is a lie about a package nobody pulls, and the
# test says so instead of quietly excusing a name that is no longer there. An
# exception that cannot expire is how the last one survived being wrong.
_TOLERATED = frozenset({"matplotlib"})


def test_importing_serving_app_pulls_in_no_untolerated_training_dependency():
    """`untolerated` is in the name because the old name was false.

    It read "pulls in no training dependency" while matplotlib -- a training
    dependency by this test's OWN derived definition of the term, read live out
    of [dependency-groups] training -- sat in sys.modules, and the test was
    green. "No X" is a universal, matplotlib was a counterexample, and declaring
    the exception in _TOLERATED does not make the universal true. It makes the
    BODY honest. The name has to say what the body checks, and the body checks
    `present - _TOLERATED`.

    The charitable reading -- "training dependency" means "a package serving
    actually needs the training group for," which matplotlib is not -- is
    available, and it is how the name survived a review that was hunting for
    exactly this. It does not survive the derivation. Once the test reads the
    group instead of remembering it, the group's members are what the name is
    quantified over, and matplotlib is in it.

    Same fix as 88301a1's `_in_one_process`: the assertion was right, the
    quantifier reached past it, so the quantifier gets the qualifier. Ugly is
    fine. False is not.

    Under-claims on purpose: this also asserts _TOLERATED has not gone stale,
    which the name does not mention. A name narrower than its body is safe --
    it is the wide ones that lie.
    """
    # src.fairness and scripts.demo_drift join the list with GET /fairness.
    # serving/fairness.py is a JSON reader and a string comparison; it must
    # never reach for the audit itself, which calls load_raw() and expects the
    # 167 MB assessment CSV -- a file .dockerignore excludes on purpose.
    # Importing src.fairness into the serving process would be the first step
    # toward a route that cannot run in the image it ships in.
    #
    # `scripts.demo_drift`, NOT `scripts`. The bare `scripts` namespace package
    # IS in sys.modules after this import, legitimately and unavoidably:
    # DRIFT_DEMO_AVAILABLE probes find_spec("scripts.demo_drift"), and finding a
    # spec for a submodule requires importing its parent to look inside it.
    # There is no scripts/__init__.py -- it is an empty namespace package, so
    # that import executes nothing and pulls in nothing. Watching `scripts` here
    # would fail on a fact about how find_spec works rather than on a dependency
    # leak, and the only way to make it pass would be to delete the honest
    # find_spec probe that keeps /drift out of the image.
    training = _training_group()
    watched = tuple(sorted(training | _FIRST_PARTY_LEAKS))
    probe = (
        "import sys; import serving.app; "
        f"print(','.join(m for m in {watched!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=PROJECT_ROOT, check=True,
    )
    present = {m for m in result.stdout.strip().split(",") if m}

    leaked = sorted(present - _TOLERATED)
    assert leaked == [], (
        f"`import serving.app` now pulls in: {', '.join(leaked)}. Those are "
        "training-group dependencies the serving image does not install "
        "(Dockerfile: uv sync --no-group training) and, for pipelines/, does not "
        "even copy (.dockerignore). The container would die at boot. If this broke "
        "because a drift import moved to module scope, move it back into the "
        "handler (serving/app.py). If the import is genuinely guarded by a "
        "try/except ImportError in third-party code -- the way lightgbm guards "
        "matplotlib -- then it is tolerated, not leaked, and belongs in _TOLERATED "
        "with the guard named. Prove the guard first; do not add it to keep this "
        "green."
    )

    stale = sorted(_TOLERATED - present)
    assert stale == [], (
        f"_TOLERATED excuses {', '.join(stale)}, which `import serving.app` no "
        "longer reaches. The tolerance is now a claim about a package nothing "
        "pulls. Delete the entry -- an exception that outlives its reason is how "
        "the transcription this test replaced stayed green while it was false."
    )


def test_the_drift_route_is_mounted_here():
    """DRIFT_DEMO_AVAILABLE is a probe, not a constant: it asks whether the demo's
    machinery is importable from where the app is being built. In this repo it is.
    find_spec answers that WITHOUT importing anything, which is what keeps the
    test above true.

    THE NAME USED TO END "_and_would_not_be_in_the_image". Nothing below asserts
    that, and nothing anywhere else does either. The second clause is true, and it
    rests entirely on three build facts no test reads:

        Dockerfile:51-53   COPY src/ serving/ models/ -- and nothing else, so
                           scripts/ and pipelines/ never enter the image. By
                           OMISSION from a list, which is all that holds it up.
        .dockerignore:18   pipelines/ (belt and braces; scripts/ is not listed
                           here at all -- it is kept out by the COPY line alone)
        Dockerfile:44      uv sync --frozen --no-dev --no-group training, so no
                           mlflow for pipelines/drift_check.py to reach anyway

    Add `COPY scripts/` to the Dockerfile and the old name is false with this test
    still green. The suite cannot see it: every mention of Dockerfile or
    .dockerignore in tests/ is a comment or a failure message. Zero assertions
    read either file.

    NOT FIXED HERE, on purpose. A MetaPathFinder can simulate the modules being
    absent and prove find_spec then returns None -- but that tests find_spec,
    and find_spec is not what holds the clause up; the COPY line is. A guard that
    checks the wrong one of two things reads as coverage and is worse than the
    honest gap. Building the image is the only thing that would actually answer
    it, and that is not a unit test.

    So the sentence shrinks to what is proven, the way 6927fa2 shrank
    ApplicationForm.tsx's determinism claim. test_drift_is_absent_not_broken_when
    _the_demo_is_not_mounted covers the other half of the IF: given
    drift_demo=False, /drift is a 404. That the image IMPLIES drift_demo=False is
    the part nobody asserts.
    """
    assert DRIFT_DEMO_AVAILABLE is True
    assert importlib.util.find_spec("scripts.demo_drift") is not None




# ---------------------------------------------------------------------------
# GET /fairness -- the frozen audit, and the gate that binds it to the model.
#
# tests/test_fairness.py owns the audit's correctness. What is tested here is
# the one thing that can rot: a JSON file on disk claiming to describe a
# booster. /fairness is the only route in this service whose payload is not
# read live off the bundle, so it is the only one that can go stale -- and the
# gate is the entire reason it is allowed to exist at all.
#
# The synthetic `bundle` fixture is NOT usable for the happy path: it is a
# throwaway model trained in tmp_path, so the committed audit is honestly stale
# against it. These tests take the SHIPPED bundle, and skip where the rest of
# the file skips -- models/*.pkl is gitignored, so a fresh clone has the audit
# (committed) but not the model (not committed).
# ---------------------------------------------------------------------------
SHIPPED = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="shipped model artifact absent (models/ is gitignored)",
)

# Numbers that must never appear in a 409 body. The point of refusing a stale
# audit is that the client is not handed ratios it could draw.
FAIRNESS_RATIOS = ("0.7448", "0.9879", "0.6690", "0.6654")


@pytest.fixture(scope="module")
def shipped_bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def shipped_audit():
    return load_fairness_audit()


@pytest.fixture(scope="module")
def fairness_client(shipped_bundle, shipped_audit):
    return TestClient(
        create_app(bundle=shipped_bundle, bureau=MockBureau(), audit=shipped_audit)
    )


@SHIPPED
def test_the_committed_audit_describes_the_model_we_actually_ship(
    shipped_bundle, shipped_audit
):
    """
    THE test. The others check that the gate WORKS; this checks that the gate is
    currently OPEN -- that models/fairness_audit.json is about the booster in
    models/lgbm_model.pkl, and not about one from three retrains ago.

    If this goes red, the fix is `uv run python scripts/audit_fairness.py`, not
    an edit to this assertion. A failure here means the repo is asserting a
    fairness claim about a model it is not serving -- the exact defect the
    artifact-plus-gate design exists to make impossible to ship quietly.
    """
    assert shipped_audit.available, shipped_audit.unavailable_reason
    assert not is_stale(shipped_audit.audit, shipped_bundle), (
        "models/fairness_audit.json was run against "
        f"{audit_model_trained_at(shipped_audit.audit)}, but the shipped model was "
        f"trained at {shipped_bundle.model_trained_at}. "
        "Re-run scripts/audit_fairness.py."
    )


@SHIPPED
def test_fairness_shows_the_provenance_match_rather_than_asking_for_trust(
    fairness_client, shipped_bundle
):
    body = fairness_client.get("/fairness").json()

    # The client can SEE the binding hold, instead of inferring it from the
    # absence of a 409.
    assert body["model"]["trained_at"] == body["shipped_model_trained_at"]
    assert body["shipped_model_trained_at"] == shipped_bundle.model_trained_at

    # The fairness conclusion, executed and checkable: the model that made
    # these decisions has no addr_state to lean on.
    assert body["model"]["includes_addr_state"] is False
    assert "addr_state" not in body["model"]["features"]


@SHIPPED
def test_the_endpoint_reports_the_artifact_it_computes_nothing(fairness_client):
    """
    Every number on the wire already existed in the JSON on disk. /fairness is a
    file reader; if it were quietly deriving anything, the two would diverge.
    """
    body = fairness_client.get("/fairness").json()
    on_disk = json.loads(FAIRNESS_AUDIT_PATH.read_text())

    for key in ("schema_version", "generated_at", "model", "constants",
                "layer1", "layer2", "layer3"):
        assert body[key] == on_disk[key]

    # shipped_model_trained_at is the ONLY key the route adds, and it is read
    # off the live bundle rather than copied out of the file.
    assert set(body) - set(on_disk) == {"shipped_model_trained_at"}


@SHIPPED
def test_the_audits_constants_are_the_audits_own_not_retyped_in_serving(fairness_client):
    """
    A client draws the 0.80 line, and it must draw it where src/fairness.py puts
    it -- the same discipline /drift follows with DEFAULT_ALARM_THRESHOLDS.
    """
    constants = fairness_client.get("/fairness").json()["constants"]

    assert constants["eo_threshold"] == EO_THRESHOLD
    assert constants["min_n"] == MIN_N
    assert constants["n_boot"] == N_BOOT
    assert constants["ablation_threshold"] == ABLATION_THRESHOLD
    assert constants["watch_states"] == list(WATCH_STATES)

    # The sweep is the ONE constant the artifact does not ship verbatim, and the
    # difference is deliberate: scripts/audit_fairness.py adds the operating
    # threshold, because src/fairness.py's notebook-era list does not contain it
    # and a sweep that omits the point we decide at cannot answer "what do we
    # approve at the point we decide at". So: a superset, and specifically THAT
    # superset -- shipping the module constant here while having swept something
    # else would be the same say != do in miniature.
    assert constants["sweep_thresholds"] == sorted(
        set(SWEEP_THRESHOLDS) | {SELECTED_THRESHOLD}
    )


@SHIPPED
def test_layer1_audits_the_shipped_model_at_the_point_score_decides_at(
    fairness_client, shipped_bundle
):
    """
    Auditing at any other cutoff answers a question nobody asked. And the value
    is SELECTED_THRESHOLD (0.25000000000000006), not the literal 0.25 -- a real,
    different float (serving/config.py).
    """
    layer1 = fairness_client.get("/fairness").json()["layer1"]

    assert layer1["threshold"] == SELECTED_THRESHOLD
    assert layer1["threshold"] == shipped_bundle.threshold


@SHIPPED
def test_every_reported_interval_brackets_its_own_point_estimate(fairness_client):
    """
    The CI is the whole reason Layer 3 now returns its Test frames. An interval
    that did not contain its own ratio would make the chart drawn from it
    fiction.
    """
    body = fairness_client.get("/fairness").json()

    for row in body["layer1"]["states"]:
        assert row["ci_low"] <= row["eo_ratio"] <= row["ci_high"], row["state"]

    for row in body["layer3"]["states"]:
        for side in ("with_state", "no_state"):
            assert (
                row[f"ci_low_{side}"] <= row[f"eo_ratio_{side}"] <= row[f"ci_high_{side}"]
            ), (row["state"], side)
        # Same Test set, same y_true: the ablation toggles a feature, not a
        # population. A free invariant, so assert it.
        assert row["n_good_with_state"] == row["n_good_no_state"]


@SHIPPED
def test_the_ablation_is_the_evidence_the_readme_claims_it_is(fairness_client):
    """
    The repo's loudest fairness claim, checked against the artifact that is
    actually shipped: MS is CONFIRMED (whole CI below the 0.80 line) with the
    state label, and clear without it -- and no state is confirmed once
    addr_state is gone.

    Deliberately asserted as an INTERVAL claim, not "0.744 -> 0.988". The point
    of returning Layer 3's frames was that two bare point estimates cannot tell
    a real shift from sampling noise. A test that pinned the point estimates
    would be re-committing the error the CIs were added to fix.
    """
    body = fairness_client.get("/fairness").json()
    eo_threshold = body["constants"]["eo_threshold"]
    states = {row["state"]: row for row in body["layer3"]["states"]}

    ms = states["MS"]
    assert ms["ci_high_with_state"] < eo_threshold      # whole interval below 0.80
    assert ms["ci_high_no_state"] > eo_threshold        # recovers past it
    assert ms["ci_low_no_state"] > ms["ci_high_with_state"]  # intervals disjoint

    confirmed_with = {s for s, r in states.items()
                      if r["verdict_with_state"].startswith("confirmed")}
    confirmed_without = {s for s, r in states.items()
                         if r["verdict_no_state"].startswith("confirmed")}
    assert confirmed_with == {"MS"}
    assert confirmed_without == set()

    # And the price, which the README quotes: dropping the feature costs AUC.
    assert body["layer3"]["auc_no_state"] < body["layer3"]["auc_with_state"]


@SHIPPED
def test_a_stale_audit_is_refused_and_not_one_ratio_is_sent(shipped_bundle, shipped_audit):
    """
    The 409 carries BOTH timestamps and NO numbers.

    Sending the ratios with a warning attached is not a middle ground: a client
    handed numbers will draw them. Withholding them is the only thing that
    reliably stops a stale ratio being rendered as a current one.
    """
    relabelled = dataclasses.replace(
        shipped_bundle, model_trained_at="2099-01-01T00:00:00+00:00"
    )
    client = TestClient(
        create_app(bundle=relabelled, bureau=MockBureau(), audit=shipped_audit)
    )
    response = client.get("/fairness")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["audit_model_trained_at"] == shipped_bundle.model_trained_at
    assert detail["shipped_model_trained_at"] == "2099-01-01T00:00:00+00:00"

    for ratio in FAIRNESS_RATIOS:
        assert ratio not in response.text, f"a stale {ratio} escaped in the 409 body"


def test_an_audit_that_cannot_name_its_model_is_stale_by_default(bundle):
    """
    Absence of provenance is not evidence of a match. An artifact that cannot
    say which model it audited has not earned the benefit of the doubt.
    """
    nameless = FairnessAudit({"schema_version": 1, "model": {}}, None)
    assert is_stale(nameless.audit, bundle)


def test_fairness_is_absent_not_broken_when_there_is_no_artifact(bundle):
    missing = load_fairness_audit(Path("models/no_such_audit.json"))
    assert not missing.available

    client = TestClient(create_app(bundle=bundle, bureau=MockBureau(), audit=missing))
    response = client.get("/fairness")

    assert response.status_code == 404
    assert "scripts/audit_fairness.py" in response.json()["detail"]


def test_an_unknown_schema_version_is_refused_rather_than_read_halfway(tmp_path):
    """
    A future artifact shape must not be read partially: absent keys would surface
    as nulls on the wire and render as blank chart axes instead of as an error.
    """
    future = tmp_path / "fairness_audit.json"
    future.write_text(json.dumps({"schema_version": 99, "model": {"trained_at": "x"}}))

    loaded = load_fairness_audit(future)

    assert not loaded.available
    assert "schema_version" in loaded.unavailable_reason


def test_a_broken_fairness_artifact_never_takes_scoring_down(bundle):
    """
    Fail-closed on the numbers, fail-OPEN on the service.

    The audit is a REPORTING signal (blue in docs/architecture.html), not a
    gate -- the same policy training_flow.py's explain step applies. The
    synthetic `bundle` here is genuinely a different model from the one the
    committed audit describes, so this is a real staleness, not a simulated one.
    """
    cases = {
        "stale": load_fairness_audit(),
        "absent": load_fairness_audit(Path("models/no_such_audit.json")),
    }
    for label, audit in cases.items():
        client = TestClient(create_app(bundle=bundle, bureau=MockBureau(), audit=audit))
        assert client.get("/fairness").status_code in (404, 409), label
        # ...and the service goes right on scoring applicants.
        assert client.post("/score", json=GOOD).status_code == 200, label
        assert client.get("/healthz").status_code == 200, label


@SHIPPED
def test_the_sweep_contains_the_operating_point_exactly_not_nearly(fairness_client):
    """
    The Layer-2 sweep must hold a row AT the threshold /score decides at, matching
    on ==, not "within a tolerance".

    Two things conspire here and both were live defects:

      1. src/fairness.py's SWEEP_THRESHOLDS is the notebook's list and does not
         contain 0.25000000000000006. scripts/audit_fairness.py adds the operating
         point to the sweep rather than letting a client round to the nearest row
         (0.26) and call it the operating one.

      2. pandas' to_json defaults to double_precision=10 and silently rounds
         0.25000000000000006 to 0.25 -- a genuinely different float. The row was
         being written, and the lookup for it still failed. scripts/audit_fairness.py's
         _records() avoids to_json for exactly this reason.

    Neither is visible by reading the JSON: 0.25 and 0.25000000000000006 print the
    same at any sane precision. Only the == lookup catches it, which is why this
    asserts the lookup.
    """
    body = fairness_client.get("/fairness").json()
    operating = body["layer1"]["threshold"]

    assert operating == SELECTED_THRESHOLD
    assert operating in body["constants"]["sweep_thresholds"]

    rows = [r for r in body["layer2"]["rows"] if r["threshold"] == operating]
    assert len(rows) == 1, (
        "No sweep row at the operating threshold. A client asking 'what does the "
        "shipped model approve at the cutoff it actually uses?' would have to round "
        "to the nearest swept row and report that as the operating point."
    )
    # And it is a real approval rate, not a placeholder.
    assert 0.0 < rows[0]["national_good_approval_rate"] <= 1.0
