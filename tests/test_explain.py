"""
Tests for src/explain.py

Doesn't chase the real dataset's SHAP values -- those only mean something
against the shipped artifact, and run_explanation()'s docstring records them
from an actual run. What's locked down here is the EXPLANATION CONTRACT:

  1. Additivity on the raw axis: base_value + sum(contributions) IS the margin
     the model scores with, and sigmoid of it IS Booster.predict().
  2. Decision consistency: REJECT iff p_cal >= threshold, at the exact float
     the test names.
  3. Fail-closed: a mismatched feature contract or a stale calibrator raises
     THROUGH this module's entry point, not merely inside calibrate.py.
  4. The scale is declared, and
  5. no probability-scale contribution can leak into the output -- the key set
     of a reason code is asserted exactly. docs/explainability.md Section 5
     proves percentage-point attribution is undefined under the shipped
     calibrator; test 5 is what keeps that proof load-bearing.
  6. Sign convention: only risk-INCREASING factors are denial reasons, and an
     applicant with no risk-increasing factor gets an empty list, not a
     manufactured reason.
  7. Rank order is a fact about the returned list.
  8. The precomputed escape hatch touches no disk, and is all-or-nothing.
  9. Explanation never reads the target -- proven by passing a frame that has
     no target column at all. This is the serving-path guarantee.
 10. shap's return shapes are normalized in exactly one place, including the
     one that bites: explainer.expected_value is MUTATED by shap_values().

Run:  pytest tests/test_explain.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib
from scipy.special import expit

from src.features import CATEGORICAL, FEATURES, TARGET
from src.train import _to_lgb_frame, _x, train_lgb
from src.calibrate import calibrate_model
from src.explain import (
    CONTRIBUTION_SCALE,
    DEFAULT_EXPLAIN_THRESHOLD,
    _rank_adverse,
    _shap_matrix,
    explain_applicants,
    global_importance,
)

# The exact float evaluate.select_threshold() returns on the real dataset:
# np.arange(0.05, 0.95, 0.01)[20], logged by MLflow run cca4c361 as
# best_threshold. It is NOT 0.25. See DEFAULT_EXPLAIN_THRESHOLD's comment.
SELECTED_THRESHOLD = float(np.arange(0.05, 0.95, 0.01)[20])

REASON_CODE_KEYS = {"rank", "feature", "value", "contribution_log_odds"}


def _make_split(n, purposes, homes, states, emp_lengths, rng):
    fico = rng.uniform(620, 820, n)
    dti = rng.uniform(0, 40, n)
    z = 0.05 * (dti - 20) - 0.03 * (fico - 700)
    p_default = 0.5 / (1 + np.exp(-z)) + 0.5 * 0.15
    return pd.DataFrame({
        "revenue": rng.uniform(20_000, 150_000, n),
        "dti_n": dti,
        "loan_amnt": rng.uniform(1_000, 35_000, n),
        "fico_n": fico,
        "emp_length": rng.choice(emp_lengths, n),
        "purpose": rng.choice(purposes, n),
        "home_ownership_n": rng.choice(homes, n),
        "addr_state": rng.choice(states, n),
        "Default": rng.binomial(1, p_default),
    })


@pytest.fixture
def synthetic_splits() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(13)
    purposes = ["debt_consolidation", "credit_card", "other"]
    homes = ["MORTGAGE", "RENT", "OWN"]
    emp_lengths = ["< 1 year", "5 years", "10+ years", "NI"]
    states = ["CA", "TX", "NY"]
    return {
        "train": _make_split(800, purposes, homes, states, emp_lengths, rng),
        "val": _make_split(200, purposes, homes, states, emp_lengths, rng),
        "calib": _make_split(500, purposes, homes, states, emp_lengths, rng),
        "test": _make_split(200, purposes, homes, states, emp_lengths, rng),
    }


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
def applicants(synthetic_splits):
    """Unlabeled, exactly as a serving request arrives: no TARGET column."""
    return synthetic_splits["test"].head(40).drop(columns=[TARGET]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Additivity on the raw axis. SHAP explains the margin the model actually
#    scores with -- exactly, not approximately -- and p_raw is sigmoid of it.
# ---------------------------------------------------------------------------
def test_additivity_reconstructs_the_raw_margin(applicants, artifact, model_path, calibrator_path):
    out = explain_applicants(applicants, model_path=model_path, calibrator_path=calibrator_path)

    X_lgb = _to_lgb_frame(_x(applicants), artifact["category_maps"])
    margin = artifact["model"].predict(
        X_lgb, num_iteration=artifact["best_iteration"], raw_score=True,
    )

    for i, o in enumerate(out):
        recon = o["base_value_log_odds"] + sum(o["contributions_log_odds"].values())
        assert recon == pytest.approx(margin[i], abs=1e-9)
        assert o["raw_margin_log_odds"] == pytest.approx(margin[i], abs=1e-9)


def test_additivity_holds_for_a_single_row(applicants, artifact, model_path, calibrator_path):
    """The serving case. One row, double brackets."""
    one = applicants.iloc[[0]]
    out = explain_applicants(one, model_path=model_path, calibrator_path=calibrator_path)
    assert len(out) == 1

    X_lgb = _to_lgb_frame(_x(one), artifact["category_maps"])
    margin = artifact["model"].predict(
        X_lgb, num_iteration=artifact["best_iteration"], raw_score=True,
    )[0]
    assert out[0]["raw_margin_log_odds"] == pytest.approx(margin, abs=1e-9)


def test_p_raw_is_exactly_sigmoid_of_the_margin(applicants, artifact, model_path, calibrator_path):
    """objective="binary" means Booster.predict() IS sigmoid(margin). explain.py
    derives p_raw that way instead of predicting twice; this pins the identity."""
    out = explain_applicants(applicants, model_path=model_path, calibrator_path=calibrator_path)

    X_lgb = _to_lgb_frame(_x(applicants), artifact["category_maps"])
    p_lgb = artifact["model"].predict(X_lgb, num_iteration=artifact["best_iteration"])

    for i, o in enumerate(out):
        assert o["p_raw"] == pytest.approx(p_lgb[i], abs=1e-9)
        assert o["p_raw"] == pytest.approx(expit(o["raw_margin_log_odds"]), abs=1e-12)


def test_shap_explains_only_the_trees_the_model_scores_with(applicants, artifact):
    """explain.py forwards tree_limit=best_iteration so it explains exactly the
    trees Booster.predict(num_iteration=best_iteration) uses. The shipped
    booster carries no trees past best_iteration, so this is currently a no-op
    -- but a booster saved without truncation would silently break additivity,
    and the forwarding is what makes that impossible rather than merely
    unlikely. Asserted through a recording stub, since a real explainer cannot
    report what it was asked."""
    seen = {}

    class _RecordingExplainer:
        expected_value = np.array([0.0])

        def shap_values(self, X, tree_limit=None, **kw):
            seen["tree_limit"] = tree_limit
            type(self).expected_value = np.float64(0.0)
            return np.zeros((len(X), len(FEATURES)))

    X_lgb = _to_lgb_frame(_x(applicants), artifact["category_maps"])
    _shap_matrix(_RecordingExplainer(), X_lgb, tree_limit=artifact["best_iteration"])
    assert seen["tree_limit"] == artifact["best_iteration"]


# ---------------------------------------------------------------------------
# 2. Decision consistency: REJECT iff p_cal >= threshold. Both sides, and the
#    boundary row states WHICH float it is testing against -- 0.25 and the real
#    selected threshold 0.25000000000000006 disagree on exactly that row.
# ---------------------------------------------------------------------------
def test_decision_is_reject_iff_p_cal_at_or_above_threshold(applicants):
    n = len(applicants)
    p_cal = np.linspace(0.05, 0.60, n)
    out = explain_applicants(
        applicants, threshold=0.25,
        shap_values=np.zeros((n, len(FEATURES))), base_value=0.0, p_cal=p_cal,
    )
    for o, p in zip(out, p_cal):
        assert o["decision"] == ("REJECT" if p >= 0.25 else "APPROVE")
    assert {o["decision"] for o in out} == {"REJECT", "APPROVE"}   # both sides exercised


def test_p_cal_exactly_0_25_flips_between_the_constant_and_the_selected_threshold(applicants):
    """A row at p_cal == 0.25 EXACTLY.

    Against DEFAULT_EXPLAIN_THRESHOLD (0.25):            0.25 >= 0.25  -> REJECT
    Against the threshold select_threshold() returns
    on the real dataset (0.25000000000000006):  0.25 >= 0.2500...06  -> APPROVE

    One ULP, two different letters in an adverse-action notice. This test names
    both floats rather than trusting either."""
    assert DEFAULT_EXPLAIN_THRESHOLD == 0.25
    assert SELECTED_THRESHOLD == 0.25000000000000006
    assert SELECTED_THRESHOLD != 0.25

    one = applicants.iloc[[0]]
    kw = dict(shap_values=np.zeros((1, len(FEATURES))), base_value=0.0,
              p_cal=np.array([0.25]))

    assert explain_applicants(one, threshold=DEFAULT_EXPLAIN_THRESHOLD, **kw)[0]["decision"] == "REJECT"
    assert explain_applicants(one, threshold=SELECTED_THRESHOLD, **kw)[0]["decision"] == "APPROVE"


# ---------------------------------------------------------------------------
# 3. Fail-closed, both guards, THROUGH explain.py's own entry point. The guards
#    existing in train.py / calibrate.py proves nothing about whether this
#    module arms them.
# ---------------------------------------------------------------------------
def test_feature_contract_mismatch_fails_closed(applicants, artifact, tmp_path, calibrator_path):
    bad = dict(artifact, features=FEATURES + ["ghost_feature"])
    bad_path = tmp_path / "bad_model.pkl"
    joblib.dump(bad, bad_path)

    with pytest.raises(ValueError, match="contract mismatch"):
        explain_applicants(applicants, model_path=bad_path, calibrator_path=calibrator_path)


def test_stale_calibrator_fails_closed(applicants, artifact, model_path, calibrator_path, tmp_path):
    """explain.py must pass model_artifact= into load_calibrator. If it ever
    stops, this is the test that notices."""
    cal = joblib.load(calibrator_path)
    cal["model_trained_at"] = "a-different-model"
    stale_path = tmp_path / "stale_calibrator.pkl"
    joblib.dump(cal, stale_path)

    with pytest.raises(ValueError, match="[Ss]tale calibrator"):
        explain_applicants(applicants, model_path=model_path, calibrator_path=stale_path)


def test_matching_artifacts_do_not_raise(applicants, model_path, calibrator_path):
    """The negative side of both guards."""
    out = explain_applicants(applicants, model_path=model_path, calibrator_path=calibrator_path)
    assert len(out) == len(applicants)
    assert out[0]["model_trained_at"] == "test-fixture"
    assert out[0]["calibrator_trained_at"] is not None


# ---------------------------------------------------------------------------
# 4. The scale is declared, and it is the raw log-odds margin.
# ---------------------------------------------------------------------------
def test_scale_field_is_present_and_correct(applicants, model_path, calibrator_path):
    out = explain_applicants(applicants, model_path=model_path, calibrator_path=calibrator_path)
    assert CONTRIBUTION_SCALE == "log_odds_margin"
    for o in out:
        assert o["scale"] == CONTRIBUTION_SCALE
        for rc in o["reason_codes"]:
            assert "contribution_log_odds" in rc


# ---------------------------------------------------------------------------
# 5. No probability-scale contribution leaks. Three assertions, because one is
#    dodgeable by a rename. docs/explainability.md Section 5 is what this
#    defends: percentage-point attribution is UNDEFINED under the shipped
#    isotonic calibrator, so no key may imply it -- not even set to None.
# ---------------------------------------------------------------------------
def test_no_probability_scale_contribution_leaks(applicants, model_path, calibrator_path):
    out = explain_applicants(applicants, model_path=model_path, calibrator_path=calibrator_path)
    assert any(o["reason_codes"] for o in out), "fixture must produce some reason codes"

    for o in out:
        assert "contribution_to_probability" not in o
        for rc in o["reason_codes"]:
            assert "contribution_to_probability" not in rc
            assert not any("prob" in k for k in rc), rc
            assert set(rc) == REASON_CODE_KEYS, rc


def test_probability_fields_that_DO_belong_are_predictions_not_attributions(
    applicants, model_path, calibrator_path,
):
    """p_raw and p_calibrated are probabilities and they are supposed to be
    here: one is the calibrator's input, the other is the decided quantity. The
    forbidden thing is a per-feature CONTRIBUTION in probability units. This
    test states that distinction so test 5 above is not mistaken for a ban on
    the word 'probability'."""
    o = explain_applicants(applicants, model_path=model_path, calibrator_path=calibrator_path)[0]
    assert 0.0 <= o["p_raw"] <= 1.0
    assert 0.0 <= o["p_calibrated"] <= 1.0
    assert set(o["contributions_log_odds"]) == set(FEATURES)


# ---------------------------------------------------------------------------
# 6. Sign convention. Positive SHAP pushes toward default; a risk-DECREASING
#    factor is not a denial reason. Both sides.
# ---------------------------------------------------------------------------
def test_every_reason_code_is_risk_increasing(applicants, model_path, calibrator_path):
    out = explain_applicants(applicants, model_path=model_path, calibrator_path=calibrator_path)
    for o in out:
        for rc in o["reason_codes"]:
            assert rc["contribution_log_odds"] > 0


def test_all_negative_applicant_gets_an_empty_reason_list(applicants):
    """The named edge case. An applicant can be rejected with every contribution
    <= 0, if base_value alone lands above the boundary. explain.py returns [] and
    does not manufacture a reason. This test PINS today's behavior; what an
    adverse-action notice should say here is not decided (see run_explanation)."""
    one = applicants.iloc[[0]]
    out = explain_applicants(
        one, threshold=0.25,
        shap_values=-np.abs(np.arange(1, len(FEATURES) + 1, dtype=float)).reshape(1, -1),
        base_value=5.0, p_cal=np.array([0.9]),
    )[0]
    assert out["decision"] == "REJECT"
    assert out["reason_codes"] == []


def test_zero_contribution_is_not_a_denial_reason():
    """Strictly positive, not non-negative: a feature that did nothing did not
    deny anyone."""
    values = pd.Series([1.0] * len(FEATURES), index=FEATURES)
    contribs = np.zeros(len(FEATURES))
    contribs[0] = 0.5
    codes = _rank_adverse(FEATURES, contribs, values, max_reasons=4)
    assert len(codes) == 1
    assert codes[0]["feature"] == FEATURES[0]


# ---------------------------------------------------------------------------
# 7. Rank order is a fact about the list, not a promise about how it was built.
# ---------------------------------------------------------------------------
def test_reason_codes_are_rank_ordered_and_capped(applicants, model_path, calibrator_path):
    out = explain_applicants(
        applicants, model_path=model_path, calibrator_path=calibrator_path, max_reasons=3,
    )
    for o in out:
        codes = o["reason_codes"]
        assert len(codes) <= 3
        assert [rc["rank"] for rc in codes] == list(range(1, len(codes) + 1))
        vals = [rc["contribution_log_odds"] for rc in codes]
        assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# 8. The precomputed escape hatch touches no disk, and is all-or-nothing.
# ---------------------------------------------------------------------------
def test_escape_hatch_never_loads_an_artifact(applicants, tmp_path):
    """Structural, in the manner of test_evaluate's split-key proof: the model
    path does not exist, so any attempt to load it would raise."""
    n = len(applicants)
    out = explain_applicants(
        applicants,
        model_path=tmp_path / "does_not_exist.pkl",
        calibrator_path=tmp_path / "also_missing.pkl",
        shap_values=np.zeros((n, len(FEATURES))),
        base_value=0.0,
        p_cal=np.full(n, 0.1),
    )
    assert len(out) == n
    assert out[0]["decision"] == "APPROVE"
    assert out[0]["model_trained_at"] is None      # honest: nothing was loaded


@pytest.mark.parametrize("supplied", ["shap_values", "base_value", "p_cal"])
def test_partial_escape_hatch_raises(applicants, supplied):
    """A half-populated hatch would load an artifact and discard half of it --
    a silent way to explain one model and decide with another."""
    n = len(applicants)
    kw = {
        "shap_values": np.zeros((n, len(FEATURES))),
        "base_value": 0.0,
        "p_cal": np.full(n, 0.1),
    }
    with pytest.raises(ValueError, match="all-or-nothing"):
        explain_applicants(applicants, **{supplied: kw[supplied]})


# ---------------------------------------------------------------------------
# 9. Explanation NEVER reads the target. This is the serving-path guarantee,
#    proven the way test_evaluate proves its split discipline: by removing the
#    column entirely, so any read raises. It is also the test that would have
#    caught _xy (which indexes engineered[TARGET]) being on this path.
# ---------------------------------------------------------------------------
def test_explanation_requires_no_target_column(synthetic_splits, model_path, calibrator_path):
    unlabeled = synthetic_splits["test"].head(10).drop(columns=[TARGET])
    assert TARGET not in unlabeled.columns

    out = explain_applicants(unlabeled, model_path=model_path, calibrator_path=calibrator_path)
    assert len(out) == 10


def test_global_importance_requires_no_target_column(synthetic_splits, model_path):
    unlabeled = synthetic_splits["test"].head(50).drop(columns=[TARGET])
    imp = global_importance(unlabeled, model_path=model_path)
    assert list(imp.columns) == ["feature", "mean_abs_shap"]
    assert set(imp["feature"]) == set(FEATURES)
    assert (imp["mean_abs_shap"] >= 0).all()
    assert imp["mean_abs_shap"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# 10. A single applicant is a one-row FRAME. A Series is not silently accepted.
# ---------------------------------------------------------------------------
def test_single_row_frame_yields_exactly_one_explanation(applicants, model_path, calibrator_path):
    out = explain_applicants(
        applicants.iloc[[3]], model_path=model_path, calibrator_path=calibrator_path,
    )
    assert len(out) == 1


def test_a_series_is_rejected_not_silently_mishandled(applicants, model_path, calibrator_path):
    """df.iloc[3] is a Series. add_features()/_to_lgb_frame() are frame
    operations; a Series must fail, not produce a plausible wrong answer.

    It fails inside add_features(), with AttributeError: 'str' object has no
    attribute 'map' -- a true failure with an unhelpful message. Pinned to the
    specific exception rather than a bare Exception, so that if explain.py ever
    grows a friendlier up-front type check, this test fails and gets updated
    instead of silently passing on the new error."""
    with pytest.raises(AttributeError, match="has no attribute 'map'"):
        explain_applicants(
            applicants.iloc[3], model_path=model_path, calibrator_path=calibrator_path,
        )


# ---------------------------------------------------------------------------
# 11. shap's return shapes are normalized in exactly one place. The hazard that
#     matters: explainer.expected_value is MUTATED BY shap_values(). At
#     construction it is a length-1 ndarray; after the call it is a scalar, and
#     after a tree_limit call it is that subset's base value. _shap_matrix must
#     read it AFTER. (notebooks/analysis.ipynb reads it after too -- but by
#     accident of line order. Read first, its expected_value[1] would have
#     raised IndexError on a length-1 array.)
# ---------------------------------------------------------------------------
class _StubExplainer:
    """Reproduces shap's actual mutation: expected_value starts as a length-1
    array and is replaced by the post-call scalar."""

    def __init__(self, n, k, as_list=False, post_base=1.25):
        self.expected_value = np.array([-99.0])   # the pre-call value, wrong on purpose
        self._n, self._k, self._as_list, self._post = n, k, as_list, post_base

    def shap_values(self, X, tree_limit=None, **kw):
        self.expected_value = np.float64(self._post)
        block = np.ones((self._n, self._k))
        return [block * -1, block] if self._as_list else block


def test_shap_matrix_reads_expected_value_after_the_call():
    X = pd.DataFrame(np.zeros((5, len(FEATURES))), columns=FEATURES)
    sv, base = _shap_matrix(_StubExplainer(5, len(FEATURES)), X)
    assert base == 1.25          # the post-call value, not the -99.0 pre-call one
    assert sv.shape == (5, len(FEATURES))


def test_shap_matrix_takes_the_positive_class_from_a_list_return():
    """shap has historically returned [class_0, class_1] for binary boosters."""
    X = pd.DataFrame(np.zeros((5, len(FEATURES))), columns=FEATURES)
    sv, _base = _shap_matrix(_StubExplainer(5, len(FEATURES), as_list=True), X)
    assert (sv == 1.0).all()     # class_1, not the -1.0 class_0 block


def test_shap_matrix_takes_the_positive_class_from_a_two_element_base():
    class _TwoClassBase(_StubExplainer):
        def shap_values(self, X, tree_limit=None, **kw):
            self.expected_value = np.array([0.4, 0.6])
            return np.ones((self._n, self._k))

    X = pd.DataFrame(np.zeros((3, len(FEATURES))), columns=FEATURES)
    _sv, base = _shap_matrix(_TwoClassBase(3, len(FEATURES)), X)
    assert base == pytest.approx(0.6)


def test_shap_matrix_raises_loudly_on_an_unrecognized_shape():
    """A silent axis reduction would produce a plausible, wrong explanation."""
    class _WrongShape(_StubExplainer):
        def shap_values(self, X, tree_limit=None, **kw):
            self.expected_value = np.float64(0.0)
            return np.ones((self._n, self._k + 3))

    X = pd.DataFrame(np.zeros((3, len(FEATURES))), columns=FEATURES)
    with pytest.raises(ValueError, match="Unrecognized shap_values shape"):
        _shap_matrix(_WrongShape(3, len(FEATURES)), X)


def test_shap_matrix_raises_loudly_on_an_unrecognized_base_size():
    class _WrongBase(_StubExplainer):
        def shap_values(self, X, tree_limit=None, **kw):
            self.expected_value = np.array([0.1, 0.2, 0.3])
            return np.ones((self._n, self._k))

    X = pd.DataFrame(np.zeros((3, len(FEATURES))), columns=FEATURES)
    with pytest.raises(ValueError, match="Unrecognized expected_value"):
        _shap_matrix(_WrongBase(3, len(FEATURES)), X)


# ---------------------------------------------------------------------------
# 12. An unseen category degrades to NaN for the MODEL (train.py's graceful
#     inference path) but the reason code still reports the string the
#     applicant supplied. "Denied because purpose = nan" is not actionable.
# ---------------------------------------------------------------------------
def test_unseen_category_keeps_its_human_readable_value(
    applicants, artifact, model_path, calibrator_path,
):
    row = applicants.iloc[[0]].copy()
    row["purpose"] = "a_purpose_never_seen_in_training"

    # The MODEL sees NaN -- train.py's deliberate graceful-degradation path.
    assert _to_lgb_frame(_x(row), artifact["category_maps"])["purpose"].isna().all()

    out = explain_applicants(row, model_path=model_path, calibrator_path=calibrator_path)[0]
    assert np.isfinite(out["raw_margin_log_odds"])

    # Unconditional, not `if "purpose" in codes`: on this seeded fixture purpose
    # carries a positive contribution, so it MUST appear. A conditional
    # assertion here would pass silently the day it stops appearing.
    assert out["contributions_log_odds"]["purpose"] > 0
    codes = {rc["feature"]: rc["value"] for rc in out["reason_codes"]}
    assert codes["purpose"] == "a_purpose_never_seen_in_training"
    assert "nan" not in [rc["value"] for rc in out["reason_codes"]]
