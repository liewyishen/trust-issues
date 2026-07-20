"""
Tests for serving/render.py

Doesn't re-test scoring, attribution or the HTTP boundary -- test_explain.py
and test_serving.py own those. What's locked down here is the RENDERING
CONTRACT: the renderer is now the thing that must not lie, so these are the
guards that make its fixed strings checkable rather than merely readable.

  1. Closed enumerations, code-owned, derived not retyped. Every FEATURES
     member, every VALID_PURPOSE / VALID_HOME_OWNERSHIP value, both Literal
     decisions and the shipped CONTRIBUTION_SCALE need an approved phrase.
     Each list is read from its own source of truth, so adding a category in
     data_validation.py reddens a test here instead of surfacing as a
     RenderError on a live applicant.
  2. The schema partition. set(ScoreResponse.model_fields) == RENDERED |
     NOT_RENDERED, and the same for ScoredCreditReport. A field added to
     either schema lands in neither set and reddens. NOT_RENDERED is a record
     of refusals WITH REASONS, not a tolerance list -- the distinction is the
     one _TOLERATED got wrong.
  3. Ordered fragment containment. Each fact renders to a fixed string and
     every fragment must appear IN RANK ORDER. Measured: set containment
     catches 16 of 17 facts and misses rank inversion; ordered containment
     catches 17 of 17. Rank order is the regulated artifact (explain.py cites
     ECOA / Regulation B on principal reasons in rank order), not a
     presentational choice -- so the weaker check is not good enough, and
     test_set_containment_would_miss_rank_inversion pins exactly that gap so
     nobody weakens it back.
  4. The renderer states NO quantity of risk factors. This is the guard for a
     lie the first draft of render.py actually told: it said "Four factors
     pushed toward risk" for an applicant with SIX risk-increasing
     contributions, because _rank_adverse filters `s > 0` and THEN truncates
     to max_reasons. Every number in that sentence was a correct payload leaf;
     the falsehood was a cardinality the payload does not carry. The
     field-level partition of (2) is structurally blind to it -- reason_codes
     was rendered, the partition was green.
  5. Direction is ONE clause, not one per factor. _rank_adverse drops
     contribution <= 0, so "points toward risk" is a property of the LIST.
     Emitting it per factor would derive a constant N times: N chances to
     derive it wrong, which is a measured failure mode of the LLM rewrite this
     module replaces.
  6. The comparison word is COMPUTED. explain.py decides
     `"REJECT" if p_cal >= threshold else "APPROVE"`, so p_cal == threshold is
     a REJECT that is not "above" the threshold. No output of the shipped
     52-step calibrator equals it (0 of 52), which makes "above" true by
     ARTIFACT and not by construction. The test drives the boundary directly
     rather than waiting for a retrain to reach it.
  7. Empty reason_codes is its own clause, not a degenerate loop.
  8. Unmapped input RAISES. A notice that says "emp_length_ord" has explained
     nothing, and degrading to the raw column name is how an unmapped
     category reaches an applicant.
  9. No phrase asserts a CAUSE. "renting rather than owning your home" names
     what the model saw; "renting signals instability" invents a mechanism the
     model never expressed and was an observed failure of the rewrite
     approach. This is checked against the frozen content itself.

There is no network here and no model. The renderer is pure: same
ScoreResponse in, same bytes out, which is what lets its output be asserted
byte-for-byte in a diff a human reviews.
"""

import re
import typing

import pytest

from serving.render import (
    _DECISION_PHRASES,
    _FACTORS_HEADER,
    _FEATURE_PHRASES,
    _SCALE_PHRASES,
    _VALUE_PHRASES,
    CONTRIBUTION_PLACES,
    NOT_RENDERED_CREDIT_REPORT_FIELDS,
    NOT_RENDERED_FIELDS,
    PROBABILITY_PLACES,
    RENDERED_CREDIT_REPORT_FIELDS,
    RENDERED_FIELDS,
    VALUE_KEYED_FEATURES,
    RenderError,
    _first_out_of_order,
    explanation_fragments,
    render_explanation,
)
from serving.schema import ScoredCreditReport, ScoreResponse
from src.calibrate import DEFAULT_MODEL_PATH
from src.data_validation import VALID_HOME_OWNERSHIP, VALID_PURPOSE
from src.explain import CONTRIBUTION_SCALE
from src.features import FEATURES

# A real /score response, captured from the shipped artifacts (applicant-0001:
# revenue 60000, dti_n 18, loan_amnt 10000, emp_length "5 years",
# debt_consolidation, RENT). Written out rather than fetched so these tests
# stay pure unit tests of the renderer -- test_serving.py already owns the
# question of whether /score produces this.
APPROVED_PAYLOAD = {
    "scale": "log_odds_margin",
    "p_raw": 0.1311402470298053,
    "p_calibrated": 0.14722930674704382,
    "threshold": 0.25000000000000006,
    "decision": "APPROVE",
    "base_value_log_odds": -1.6992794393197017,
    "raw_margin_log_odds": -1.8909143847093923,
    "contributions_log_odds": {
        "revenue": 0.023536244530427276,
        "dti_n": -0.007036216183172891,
        "loan_amnt": -0.1360267307367919,
        "fico_n": -0.11354341259071304,
        "emp_length_ord": -0.040521649015302384,
        "emp_length_missing": -0.0022705860555225048,
        "purpose": 0.015894765631362995,
        "home_ownership_n": 0.0683326390300218,
    },
    "reason_codes": [
        {"rank": 1, "feature": "home_ownership_n", "value": "RENT",
         "contribution_log_odds": 0.0683326390300218},
        {"rank": 2, "feature": "revenue", "value": "60000.0",
         "contribution_log_odds": 0.023536244530427276},
        {"rank": 3, "feature": "purpose", "value": "debt_consolidation",
         "contribution_log_odds": 0.015894765631362995},
    ],
    "model_trained_at": "2026-07-11T05:42:41.754251+00:00",
    "calibrator_trained_at": "2026-07-11T05:42:45.938093+00:00",
    "credit_report": {
        "fico_n": 705.4484698638636,
        "bureau": "mock",
        "fico_version": "mock-fico-v1",
        "pulled_at": "2026-01-01T00:00:00Z",
    },
}


@pytest.fixture
def approved() -> ScoreResponse:
    return ScoreResponse(**APPROVED_PAYLOAD)


def _with(response: ScoreResponse, **overrides) -> ScoreResponse:
    """A copy with fields replaced. model_copy keeps validation honest."""
    return response.model_copy(update=overrides)


# ---------------------------------------------------------------------------
# 1. Closed enumerations, derived from their own source of truth
# ---------------------------------------------------------------------------

def test_every_feature_has_a_phrase():
    """
    Read from features.FEATURES, not retyped. A feature added there without a
    phrase reddens here rather than raising on a live applicant.
    """
    covered = set(_FEATURE_PHRASES) | VALUE_KEYED_FEATURES
    missing = set(FEATURES) - covered
    assert not missing, (
        f"features.FEATURES members with no approved phrase: {sorted(missing)}. "
        "Add a column phrase to _FEATURE_PHRASES, or list the feature in "
        "VALUE_KEYED_FEATURES and give every allowed value its own phrase."
    )


def test_no_phrase_is_declared_for_a_feature_that_does_not_exist():
    """
    The other direction. A phrase for a dropped feature is dead content that
    reads as approved, and would sit in the diff looking maintained.
    """
    declared = set(_FEATURE_PHRASES) | VALUE_KEYED_FEATURES
    stale = declared - set(FEATURES)
    assert not stale, f"phrases for non-features: {sorted(stale)}"


def test_every_categorical_value_has_a_phrase():
    """
    VALID_PURPOSE and VALID_HOME_OWNERSHIP are read from data_validation, the
    same sets serving/schema.py validates requests against. A fifteenth
    purpose reddens here.
    """
    expected = {
        "home_ownership_n": VALID_HOME_OWNERSHIP,
        "purpose": VALID_PURPOSE,
        # add_features derives this as `eq("NI").astype(int)`, so the value
        # reaching the wire through str(v) is exactly "0" or "1".
        "emp_length_missing": {"0", "1"},
    }
    assert set(expected) == VALUE_KEYED_FEATURES, (
        "VALUE_KEYED_FEATURES changed without this test's expectations "
        "following it; a value-keyed feature with no enumerated value set "
        "would be checked by nothing."
    )
    for feature, values in expected.items():
        missing = {v for v in values if (feature, v) not in _VALUE_PHRASES}
        assert not missing, (
            f"{feature} values with no approved phrase: {sorted(missing)}"
        )


def test_every_decision_value_has_a_phrase():
    """
    Derived from the Literal on the schema field, not from a retyped pair.
    A third decision value cannot arrive by silence.
    """
    literal = ScoreResponse.model_fields["decision"].annotation
    values = set(typing.get_args(literal))
    assert values, "decision is no longer a Literal; this test is now vacuous"
    assert values == set(_DECISION_PHRASES), (
        f"schema decisions {sorted(values)} vs phrases "
        f"{sorted(_DECISION_PHRASES)}"
    )


def test_shipped_contribution_scale_has_a_phrase():
    """
    If explain.CONTRIBUTION_SCALE changes, this module is describing the wrong
    axis. That is the leak test_no_probability_scale_contribution_leaks
    closes on the wire; this keeps it closed in prose.
    """
    assert CONTRIBUTION_SCALE in _SCALE_PHRASES


# ---------------------------------------------------------------------------
# 2. The schema partition
# ---------------------------------------------------------------------------

def test_schema_partition_is_total():
    """
    Every ScoreResponse field on exactly one side. This is the
    _training_group() shape: derived membership rather than a hand-copied
    list, so a new field lands in neither set and reddens.
    """
    fields = set(ScoreResponse.model_fields)
    overlap = RENDERED_FIELDS & set(NOT_RENDERED_FIELDS)
    assert not overlap, f"fields claimed by both sides: {sorted(overlap)}"
    unclassified = fields - RENDERED_FIELDS - set(NOT_RENDERED_FIELDS)
    assert not unclassified, (
        f"ScoreResponse fields in neither partition: {sorted(unclassified)}. "
        "Render it, or record a REASON for refusing to -- silence is how a "
        "fact gets dropped without anyone deciding to drop it."
    )
    invented = (RENDERED_FIELDS | set(NOT_RENDERED_FIELDS)) - fields
    assert not invented, f"partition names non-fields: {sorted(invented)}"


def test_credit_report_partition_is_total():
    """
    credit_report is rendered, so its own fields need the same treatment --
    otherwise the partition stops one level above where the facts live.
    """
    fields = set(ScoredCreditReport.model_fields)
    overlap = RENDERED_CREDIT_REPORT_FIELDS & set(NOT_RENDERED_CREDIT_REPORT_FIELDS)
    assert not overlap, f"fields claimed by both sides: {sorted(overlap)}"
    unclassified = (
        fields - RENDERED_CREDIT_REPORT_FIELDS
        - set(NOT_RENDERED_CREDIT_REPORT_FIELDS)
    )
    assert not unclassified, (
        f"ScoredCreditReport fields in neither partition: {sorted(unclassified)}"
    )
    invented = (
        RENDERED_CREDIT_REPORT_FIELDS | set(NOT_RENDERED_CREDIT_REPORT_FIELDS)
    ) - fields
    assert not invented, f"partition names non-fields: {sorted(invented)}"


def test_every_refusal_carries_a_reason():
    """
    NOT_RENDERED is a record of refusals, not a tolerance list. An entry with
    an empty reason is a toleration wearing a refusal's clothes.
    """
    for name, reason in {**NOT_RENDERED_FIELDS,
                         **NOT_RENDERED_CREDIT_REPORT_FIELDS}.items():
        assert reason and len(reason) > 40, (
            f"{name} is refused without an argument: {reason!r}"
        )


# ---------------------------------------------------------------------------
# 3. Ordered fragment containment
# ---------------------------------------------------------------------------

def test_rendered_output_contains_every_fragment_in_order(approved):
    text = render_explanation(approved)
    assert _first_out_of_order(text, explanation_fragments(approved)) is None


def test_set_containment_would_miss_rank_inversion(approved):
    """
    The measured gap, pinned so it cannot be weakened back.

    Swapping two ranked factors leaves every fragment PRESENT -- set
    containment is green -- and breaks the order ECOA cares about. This test
    asserts both halves: that the weaker check passes the mutant, and that the
    shipped check fails it.
    """
    text = render_explanation(approved)
    fragments = explanation_fragments(approved)

    lines = text.split("\n")
    first = next(i for i, ln in enumerate(lines) if ln.startswith("  1. "))
    lines[first], lines[first + 1] = lines[first + 1], lines[first]
    swapped = "\n".join(lines)

    assert all(f in swapped for f in fragments), (
        "the mutant was supposed to keep every fragment present; if it does "
        "not, this test is no longer measuring set-vs-ordered containment"
    )
    assert _first_out_of_order(swapped, fragments) is not None, (
        "ordered containment did not catch a rank inversion, which is the one "
        "fact of seventeen that set containment misses"
    )


def test_renderer_rejects_output_that_lost_a_fact(approved):
    """
    render_explanation self-checks before returning. Drive the check directly
    with a text that dropped a factor: the fragment list still names it.
    """
    fragments = explanation_fragments(approved)
    text = render_explanation(approved)
    without_third = "\n".join(
        ln for ln in text.split("\n") if not ln.startswith("  3. ")
    )
    assert _first_out_of_order(without_third, fragments) is not None


# ---------------------------------------------------------------------------
# 4. No quantity of risk factors -- the tenth-blind-spot guard
# ---------------------------------------------------------------------------

_CARDINALS = {"zero", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine", "ten"}


def test_renderer_states_no_quantity_of_risk_factors(approved):
    """
    The payload carries len(reason_codes) -- the count AFTER _rank_adverse
    truncates to max_reasons -- and never the count of risk-increasing
    contributions. This applicant has 3 of each; the REJECT case measured 4
    rendered against 6 risk-positive. Any sentence naming a quantity asserts
    the one the payload does not carry.

    Rank digits ("  1.", "  2.") are payload facts and are permitted; an
    English cardinal is not, because it can only be a count.
    """
    text = render_explanation(approved).lower()
    found = {w for w in _CARDINALS if re.search(rf"\b{w}\b", text)}
    assert not found, (
        f"rendered output names a quantity: {sorted(found)}. The payload does "
        "not carry the number of risk-increasing factors -- only the number "
        "of rows that survived truncation."
    )


def test_factors_header_line_is_exactly_the_frozen_constant(approved):
    """
    Byte-equality on the header LINE, not a search for forbidden words.

    The word-based test above was mutation-tested and found insufficient: a
    count written as a DIGIT ("3 factors pushed toward risk") passed all
    twenty-nine tests, because rank digits are legitimate elsewhere in the
    output so digits cannot simply be banned. That failure is the same shape
    as every other blind spot in this repo -- the right instrument aimed at an
    enumeration, here the enumeration of English number words.

    An equality check is not an enumeration. Any header mutation reddens:
    word, digit, roman numeral, or a phrasing nobody has thought of.
    """
    lines = render_explanation(approved).split("\n")
    first_factor = next(i for i, ln in enumerate(lines) if re.match(r"^  \d+\. ", ln))
    header = lines[first_factor - 1]
    assert header == f"{_FACTORS_HEADER} {_SCALE_PHRASES[approved.scale]}:", (
        f"the line introducing the ranked factors is not the frozen header: "
        f"{header!r}"
    )
    assert not any(c.isdigit() for c in _FACTORS_HEADER), _FACTORS_HEADER


# ---------------------------------------------------------------------------
# 5. Direction is one clause, not one per factor
# ---------------------------------------------------------------------------

def test_direction_is_stated_once_not_per_factor(approved):
    """
    _rank_adverse filters `s > 0`, so direction is a property of the LIST.
    Stating it per factor derives a constant N times -- N chances to state it
    wrong, which is measured failure #1 of the rewrite approach (an income
    factor the payload ranks as risk-INCREASING described as favourable).
    """
    text = render_explanation(approved)
    assert text.count("pushed toward risk") == 2, (
        "expected exactly two mentions -- the section header and the closing "
        "clause -- and never one per listed factor"
    )
    factor_lines = [ln for ln in text.split("\n") if re.match(r"^  \d+\. ", ln)]
    assert len(factor_lines) == len(approved.reason_codes)
    for line in factor_lines:
        assert "toward risk" not in line, line
        assert "away from risk" not in line, line


def test_the_one_sided_clause_is_unconditional(approved):
    """
    It appears for both decisions and for the empty list. An earlier draft
    made the framing clause conditional on APPROVE, which is a branch, and a
    branch is a place the framing can be wrong for one of its values.
    """
    rejected = _with(approved, decision="REJECT", p_calibrated=0.5588235294117647)
    empty = _with(approved, reason_codes=[])
    for response in (approved, rejected, empty):
        assert (
            "Factors that pushed away from risk are not listed here."
            in render_explanation(response)
        )


# ---------------------------------------------------------------------------
# 6. The comparison word is computed
# ---------------------------------------------------------------------------

def test_comparison_word_at_the_threshold_is_not_above(approved):
    """
    explain.py: `"REJECT" if p_cal >= threshold else "APPROVE"`. At equality
    the decision is REJECT and the probability is NOT above the threshold, so
    "above" would be a false sentence. No output of the shipped 52-step
    calibrator hits this (0 of 52 -- nearest are 0.2414 and 0.2648), which
    makes the correct word true by ARTIFACT today. The boundary is driven
    directly so a retrain cannot quietly make the renderer lie.
    """
    at_boundary = _with(
        approved, decision="REJECT", p_calibrated=approved.threshold
    )
    text = render_explanation(at_boundary)
    assert "at or above the decision threshold" in text
    assert "which is above" not in text


def test_comparison_word_below_and_above(approved):
    assert "which is below the decision threshold" in render_explanation(approved)
    above = _with(approved, decision="REJECT", p_calibrated=0.5588235294117647)
    assert "at or above the decision threshold" in render_explanation(above)


# ---------------------------------------------------------------------------
# 7. Empty reason_codes
# ---------------------------------------------------------------------------

def test_empty_reason_codes_is_its_own_clause(approved):
    """
    ScoreResponse's docstring records the empty list as a real case. A
    cardinality-driven template renders "Zero factors ... strongest first:"
    followed by nothing -- clauses that are individually true and together
    read as a formatting failure.
    """
    text = render_explanation(_with(approved, reason_codes=[]))
    assert "No factor pushed toward risk." in text
    assert _FACTORS_HEADER not in text
    assert "strongest first" not in text
    assert not [ln for ln in text.split("\n") if re.match(r"^  \d+\. ", ln)]


def test_empty_reason_codes_does_not_route_or_editorialise(approved):
    """
    design.md Section 6 routes an empty-reason applicant to human review, and
    serving deliberately does not decide that (no `requires_review` field, by
    ScoreResponse's own docstring). The renderer must not announce a policy
    the API refuses to embed.
    """
    text = render_explanation(_with(approved, reason_codes=[])).lower()
    for leak in ("review", "manual", "referred", "escalat"):
        assert leak not in text, leak


# ---------------------------------------------------------------------------
# 8. Unmapped input raises
# ---------------------------------------------------------------------------

def test_unmapped_category_raises_rather_than_leaking_the_raw_value(approved):
    codes = [approved.reason_codes[0].model_copy(update={"value": "HOUSEBOAT"})]
    with pytest.raises(RenderError, match="home_ownership_n"):
        render_explanation(_with(approved, reason_codes=codes))


def test_unmapped_scale_raises(approved):
    with pytest.raises(RenderError, match="scale"):
        render_explanation(_with(approved, scale="probability"))


def test_unmapped_decision_raises(approved):
    """
    model_copy bypasses validation, which is what makes this reachable at all
    -- and is exactly the state a schema change could create.
    """
    with pytest.raises(RenderError, match="decision"):
        render_explanation(_with(approved, decision="REFER"))


# ---------------------------------------------------------------------------
# 9. The frozen content itself
# ---------------------------------------------------------------------------

def test_no_phrase_asserts_a_cause():
    """
    Phrases name what the model saw. "renting signals instability" invents a
    mechanism the model never expressed -- an observed failure of the rewrite
    approach this module replaces -- and an invented cause in an adverse
    notice is a claim nobody can audit.
    """
    causal = ("because", "signals", "indicates", "means that", "which is why",
              "due to", "shows that", "proves")
    for key, phrase in {**{k: v for k, v in _FEATURE_PHRASES.items()},
                        **{f"{k[0]}={k[1]}": v for k, v in _VALUE_PHRASES.items()}}.items():
        low = phrase.lower()
        for word in causal:
            assert word not in low, f"{key}: {phrase!r} asserts a cause"


def test_no_phrase_editorialises_about_the_applicant():
    """
    Measured failure #5 of the rewrite approach was invented character
    judgement ("you've shown you can manage credit responsibly"). Frozen
    content is where that would now live, so it is checked here.
    """
    judgemental = ("responsib", "unfortunate", "sadly", "good", "bad", "poor",
                   "excellent", "risky", "unstable", "concerning", "healthy")
    for key, phrase in {**{k: v for k, v in _FEATURE_PHRASES.items()},
                        **{f"{k[0]}={k[1]}": v for k, v in _VALUE_PHRASES.items()}}.items():
        low = phrase.lower()
        for word in judgemental:
            assert word not in low, f"{key}: {phrase!r} editorialises"


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------

def test_scale_is_rendered_exactly_when_it_has_a_referent(approved):
    """
    `scale` is in RENDERED_FIELDS but names the axis the CONTRIBUTIONS live
    on, so with an empty reason_codes list it qualifies nothing. Printing an
    axis for numbers that are not shown would be a fact stated about nothing.
    The partition test cannot see this -- it works at field granularity -- so
    the condition is pinned here.
    """
    assert "log-odds margin" in render_explanation(approved)
    assert "log-odds margin" not in render_explanation(_with(approved, reason_codes=[]))


# Same predicate and same reason string as test_serving.py's SHIPPED marker,
# on purpose: one condition phrased twice is a second thing to keep in step.
@pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="shipped model artifact absent (models/ is gitignored)",
)
def test_the_captured_payload_still_matches_a_live_score(approved):
    """
    APPROVED_PAYLOAD is hand-copied, which is the staleness shape this repo
    keeps fixing -- a fact transcribed out of its source of truth drifts
    silently. This is the only test here that touches the shipped artifacts,
    and it exists so the other twenty-seven can stay pure unit tests without
    the fixture quietly becoming fiction.
    """
    from fastapi.testclient import TestClient

    from serving.app import create_app
    from serving.artifacts import load_bundle
    from serving.bureau import MockBureau

    client = TestClient(create_app(bundle=load_bundle(), bureau=MockBureau()))
    live = client.post("/score", json={
        "applicant_id": "applicant-0001", "revenue": 60000.0, "dti_n": 18.0,
        "loan_amnt": 10000.0, "emp_length": "5 years",
        "purpose": "debt_consolidation", "home_ownership_n": "RENT",
    })
    assert live.status_code == 200, live.text
    assert render_explanation(ScoreResponse(**live.json())) == render_explanation(approved)


def test_render_is_deterministic(approved):
    """
    Same response in, same bytes out. This is what lets the output be reviewed
    in a diff and asserted byte-for-byte -- and it is only true because there
    is no model, no clock and no randomness anywhere in the path.
    """
    assert render_explanation(approved) == render_explanation(approved)


def test_rendered_numbers_are_the_payload_rounded_to_declared_precision(approved):
    """
    Rounding is truthful AT the declared precision. The test re-derives the
    same rounding from the payload rather than matching a literal, so a change
    to PROBABILITY_PLACES / CONTRIBUTION_PLACES cannot silently desynchronise
    the output from what the constants claim.
    """
    text = render_explanation(approved)
    assert f"{approved.p_calibrated:.{PROBABILITY_PLACES}f}" in text
    assert f"{approved.threshold:.{PROBABILITY_PLACES}f}" in text
    for code in approved.reason_codes:
        assert f"{code.contribution_log_odds:+.{CONTRIBUTION_PLACES}f}" in text


def test_no_refused_field_value_appears_in_the_output(approved):
    """
    The partition says these are not rendered. This checks the renderer agrees
    -- p_raw in particular, because it is a probability that sits one line away
    from p_calibrated and would read as an alternative answer.
    """
    text = render_explanation(approved)
    assert f"{approved.p_raw:.{PROBABILITY_PLACES}f}" not in text
    assert f"{approved.base_value_log_odds:+.{CONTRIBUTION_PLACES}f}" not in text
    assert f"{approved.raw_margin_log_odds:+.{CONTRIBUTION_PLACES}f}" not in text
