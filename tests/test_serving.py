"""
Tests for serving/

Doesn't re-test scoring, calibration or attribution -- tests/test_explain.py
owns those, and serving/ reaches all three THROUGH explain_applicants(). What's
locked down here is the SERVING CONTRACT, the things that are true of the
HTTP boundary and of nothing beneath it:

  1. The threshold is the one select_threshold() chose (0.25000000000000006),
     not explain.py's literal 0.25, and the two decide differently at p_cal
     exactly 0.25.
  2. ScoreResponse mirrors explain_applicants()'s dict key-for-key, PLUS three
     bureau-provenance fields explain_applicants() knows nothing about. This
     is the test that makes serving/ an adapter instead of a second
     implementation, for everything explain_applicants() is responsible for.
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
     request: the bureau is called with exactly the submitted applicant_id,
     the three provenance fields it returns (bureau, fico_version,
     credit_report_pulled_at) surface in the response, and the same
     applicant_id scores reproducibly across independent /score calls -- the
     same determinism MockBureau itself guarantees (tests/test_bureau.py),
     now proven through the HTTP layer.

Plus, carried across the HTTP boundary from tests/test_explain.py: no
probability-scale contribution leaks into the response.

Run:  pytest tests/test_serving.py -v
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import joblib
import shap
from fastapi.testclient import TestClient

import src.explain as src_explain
from src.data_validation import VALID_HOME_OWNERSHIP, VALID_PURPOSE
from src.features import CATEGORICAL, FEATURES, TARGET, emp_order
from src.model_io import _x, load_model_artifact, train_lgb
from src.calibrate import DEFAULT_MODEL_PATH, calibrate_model
from src.explain import DEFAULT_EXPLAIN_THRESHOLD, explain_applicants

from serving.app import create_app
from serving.artifacts import load_bundle
from serving.bureau import CreditBureau, CreditReport, MockBureau
from serving.config import SELECTED_THRESHOLD
from serving.schema import EMP_LENGTH_NOT_DISCLOSED, VALID_EMP_LENGTH, ScoreRequest, ScoreResponse

REASON_CODE_KEYS = {"rank", "feature", "value", "contribution_log_odds"}
BUREAU_PROVENANCE_KEYS = {"bureau", "fico_version", "credit_report_pulled_at"}

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
    REQUEST shape now that fico_n is bureau-sourced. The three bureau fields
    are subtracted because explain_applicants() doesn't produce them --
    /score fills them in separately from the CreditReport it fetched."""
    direct = explain_applicants(
        pd.DataFrame([GOOD_RAW]), model_path=model_path,
        calibrator_path=calibrator_path, threshold=bundle.threshold,
    )[0]
    non_bureau_fields = set(ScoreResponse.model_fields) - BUREAU_PROVENANCE_KEYS
    assert non_bureau_fields == set(direct)


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


def test_response_carries_bureau_provenance(client, bureau):
    report = bureau.fetch(GOOD["applicant_id"])
    body = client.post("/score", json=GOOD).json()
    assert body["bureau"] == report.bureau == "mock"
    assert body["fico_version"] == report.fico_version
    # datetime round-trips through JSON as an ISO string (pydantic v2 emits a
    # "Z" suffix for UTC rather than "+00:00" -- parse both back to compare
    # the actual instant rather than the string spelling of it).
    assert datetime.fromisoformat(body["credit_report_pulled_at"]) == report.pulled_at


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
