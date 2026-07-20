"""
Render a ScoreResponse as plain-language prose. Pure, deterministic, no model.

WHY THIS IS CODE AND NOT AN LLM CALL
------------------------------------------------------------------------
This module is the measured conclusion of an investigation, not a preference.
Eight real model calls were given a fixed fact-string and asked to rewrite it.
Across those eight the model never changed a number, never flipped the
decision, and never relabelled the scale -- and still produced a direction
flip ("your income makes you a good candidate" for a factor the payload lists
as risk-INCREASING), dropped every contribution in 6 of 8, dropped the
probability and the threshold entirely in 2 of 8, and attached policy promises
("you can expect approval again next year"). A number-membership guard passed
8 of 8 and caught none of it.

Eleven single-fact mutants of the one faithful rewrite -- relabel, direction
reversal, decision flip, omission, rank inversion, invented causation -- were
also passed 11/11 by that guard. The reason is structural, not a bug: a
relabel changes an English noun and leaves every checked quantity identical.

The guard that DOES work is ordered fragment containment (17/17 of the fact
set), and it only works on fragments the code itself wrote. So the code writes
them. What was left for a model afterwards was connectives, ordering and
register -- and since every guardable fact had by then been moved into the
template, 100% of that residue sits in the region no guard covers. Moving the
boundary did not reduce the model's risk; it concentrated the model into the
unguarded part. There is no runtime model here.

The plain-language phrasing below is design-time content: written once,
frozen as fixed strings, reviewed in the diff by a human. That review is the
approval step. Nothing generates it at request time.

WHAT THIS MODULE MUST NOT DO
------------------------------------------------------------------------
Rendering facts moves the say-equals-do burden off a filter and onto this
file. A fixed string looks like code and reads like a fact, and no guard
checks it at runtime unless one is written. A first draft of this renderer
said "Four factors pushed toward risk" for an applicant with SIX
risk-increasing contributions, because _rank_adverse (explain.py) filters
`s > 0` and THEN truncates to max_reasons -- the payload carries the rendered
count, never the risk-positive count. Every number in that draft was a
correct payload leaf. The lie was a cardinality the payload does not carry.

Hence: this renderer states NO quantity of risk factors. See _FACTORS_HEADER.
"""

from __future__ import annotations

from serving.schema import ScoreResponse
from src.explain import CONTRIBUTION_SCALE

# Rendering precision. Rounding to a fixed number of places is a truthful
# statement AT that precision -- "0.147" is a correct 3-place rendering of
# 0.14722930674704382 -- which is why the guard below re-derives the same
# rounding rather than matching raw repr. Three constants because three
# quantities on three different axes: a contribution is an order of magnitude
# smaller than a probability and 3 places would flatten +0.0159 and +0.0235
# toward each other, and a FICO score is not a probability at all.
PROBABILITY_PLACES = 3
CONTRIBUTION_PLACES = 4
FICO_PLACES = 3


# ---------------------------------------------------------------------------
# The frozen content. Design-time generation, human-approved in code review.
#
# No phrase asserts a CAUSE. "renting rather than owning your home" names what
# the model saw; "renting signals instability" would invent a mechanism the
# model never expressed, and was an observed failure mode of the rewrite
# approach this module replaces. Adverse-action language names the factor, not
# a theory of the applicant.
# ---------------------------------------------------------------------------

# Features whose plain-language phrase depends on the VALUE, not just the
# column. Every allowed value needs its own entry. The coverage test reads
# VALID_PURPOSE / VALID_HOME_OWNERSHIP from data_validation rather than
# retyping them, so a category added there reddens
# test_every_categorical_value_has_a_phrase instead of reaching an applicant
# as a RenderError at request time.
VALUE_KEYED_FEATURES = frozenset({"home_ownership_n", "purpose", "emp_length_missing"})

_VALUE_PHRASES: dict[tuple[str, str], str] = {
    # home_ownership_n -- all four of VALID_HOME_OWNERSHIP
    ("home_ownership_n", "RENT"): "renting rather than owning your home",
    ("home_ownership_n", "OWN"): "owning your home outright",
    ("home_ownership_n", "MORTGAGE"): "holding a mortgage on your home",
    ("home_ownership_n", "OTHER"): "a housing arrangement recorded as neither "
                                   "rent, ownership nor mortgage",
    # purpose -- all fourteen of VALID_PURPOSE
    ("purpose", "debt_consolidation"): "consolidating existing debt",
    ("purpose", "credit_card"): "paying down credit card balances",
    ("purpose", "home_improvement"): "home improvement work",
    ("purpose", "major_purchase"): "a major purchase",
    ("purpose", "medical"): "medical expenses",
    ("purpose", "small_business"): "funding a small business",
    ("purpose", "car"): "a car purchase",
    ("purpose", "vacation"): "a vacation",
    ("purpose", "moving"): "moving costs",
    ("purpose", "house"): "a house purchase",
    ("purpose", "wedding"): "wedding costs",
    ("purpose", "renewable_energy"): "a renewable energy project",
    ("purpose", "educational"): "educational costs",
    ("purpose", "other"): "a purpose recorded as other",
    # emp_length_missing -- the 0/1 flag add_features derives (features.py).
    # 1 means emp_length was "NI". The docstring there is explicit that this is
    # kept as a feature rather than imputed away "because declining to disclose
    # tenure may itself carry a risk signal", so the phrase names the
    # non-disclosure and stops there.
    ("emp_length_missing", "1"): "your employment tenure not being stated",
    ("emp_length_missing", "0"): "your employment tenure being stated",
    # Two value-specific OVERRIDES on otherwise column-keyed features. Both
    # exist because the raw value is a sentinel, and pairing a sentinel with
    # the column's ordinary phrase would read as a real measurement.
    #
    # dti_n == 999 is the documented missing-value sentinel (DTI_SENTINEL,
    # data_validation.py), not a debt-to-income ratio of 999.
    ("dti_n", "999.0"): "a debt-to-income ratio that was not available",
    # emp_length_ord is NaN for "NI" applicants and reaches the wire as the
    # string "nan" (ReasonCode's docstring records exactly this).
    ("emp_length_ord", "nan"): "an employment history whose length was not stated",
}

# Features whose phrase depends only on the column. The raw value is printed
# beside the phrase, so the phrase names the quantity and never restates it --
# "the length of your employment history" rather than "10 years", because
# emp_order maps "10+ years" to 10 (features.py) and "10 years" would be false
# at the top of the scale.
_FEATURE_PHRASES: dict[str, str] = {
    "revenue": "your reported annual income",
    "dti_n": "your debt-to-income ratio",
    "loan_amnt": "the loan amount requested",
    "fico_n": "your FICO score",
    "emp_length_ord": "the length of your employment history",
}

# The `scale` field is rendered from a map rather than hardcoded so that a
# change to explain.CONTRIBUTION_SCALE reddens a test instead of leaving this
# module quietly describing the wrong axis.
_SCALE_PHRASES: dict[str, str] = {
    "log_odds_margin": "the log-odds margin scale",
}

# `decision` is a Literal["REJECT", "APPROVE"] on the wire and those are the
# bytes a client parses, so they are not renamed -- they are conjugated. The
# coverage test reads typing.get_args off the schema annotation rather than
# retyping the pair, so a third decision value cannot arrive by silence.
_DECISION_PHRASES: dict[str, str] = {
    "APPROVE": "APPROVED",
    "REJECT": "REJECTED",
}


# ---------------------------------------------------------------------------
# The schema partition. Every ScoreResponse field is on exactly one side.
#
# This is the _training_group() shape: derived membership, not a hand-copied
# list. A field ADDED to ScoreResponse lands in neither set and reddens
# test_schema_partition_is_total, so it cannot be dropped by silence.
#
# NOT_RENDERED is a record of REFUSALS WITH REASONS, not a tolerance list.
# The distinction matters: a tolerance list grows whenever something is
# inconvenient, a refusal list requires an argument per entry.
# ---------------------------------------------------------------------------
RENDERED_FIELDS = frozenset(
    {"decision", "p_calibrated", "threshold", "scale", "reason_codes", "credit_report"}
)

NOT_RENDERED_FIELDS: dict[str, str] = {
    "p_raw": "The calibrator's INPUT, not the decided quantity. Rendering it "
             "beside p_calibrated invites the reader to treat whichever is "
             "more favourable as the answer.",
    "base_value_log_odds": "The population baseline. It is not a fact about "
                           "this applicant, and it is required for the "
                           "additivity identity, not for a notice.",
    "raw_margin_log_odds": "Derived: base + sum(contributions). Rendering it "
                           "invites arithmetic across two scales, which "
                           "docs/explainability.md Section 5 proves is "
                           "undefined under the shipped isotonic calibrator.",
    "contributions_log_odds": "All eight features, including the five that "
                              "pushed AWAY from risk. Rendering them beside "
                              "reason_codes would present protective factors "
                              "in a block whose header says 'toward risk'. "
                              "The closing clause states their existence "
                              "instead, without ranking them -- _rank_adverse "
                              "never ranked them, and inventing an order here "
                              "would be this module deciding something the "
                              "model did not.",
    "model_trained_at": "Provenance. Belongs in an audit view, not in a "
                        "notice to an applicant.",
    "calibrator_trained_at": "Provenance. Identifies which calibrator fit "
                             "produced p_calibrated, which an auditor needs "
                             "and an applicant cannot act on.",
}

# credit_report is RENDERED, but only one of its four fields is a fact about
# the applicant; the other three identify the pull. Partitioned separately so
# that a field added to ScoredCreditReport also cannot arrive by silence.
RENDERED_CREDIT_REPORT_FIELDS = frozenset({"fico_n", "bureau", "fico_version"})

NOT_RENDERED_CREDIT_REPORT_FIELDS: dict[str, str] = {
    "pulled_at": "Under MockBureau this is a frozen constant "
                 "(_MOCK_PULLED_AT, serving/bureau.py). Printing a fixed "
                 "timestamp as if it dated this applicant's pull would dress "
                 "a constant as a fact.",
}


# ---------------------------------------------------------------------------
# Fixed fragments that carry framing rather than data.
# ---------------------------------------------------------------------------

# States NO quantity. The payload carries len(reason_codes) -- the count of
# rows AFTER _rank_adverse truncates to max_reasons -- and does not carry the
# count of risk-increasing contributions. Those are different facts and a
# header naming either one would assert the other. See the module docstring.
_FACTORS_HEADER = "Factors that pushed toward risk, strongest first, on"

# Unconditional. An earlier draft appended "this list is not a set of reasons
# the application was approved" for APPROVE only -- true, necessary, and a
# BRANCH on `decision`, which is a place the framing can be wrong for one of
# its two values. The header above already says what the list IS ("factors
# that pushed toward risk") rather than what it explains, and the clause below
# tells the reader the list is one-sided. Together they hold the line for both
# decisions with no branch. explain.py's module docstring cites ECOA /
# Regulation B for why reason codes are rank-ordered principal factors; that
# makes them the principal reasons on a REJECT and NOT a rationale on an
# APPROVE, and neither reading survives if this clause is dropped.
_ONE_SIDED_CLAUSE = "Factors that pushed away from risk are not listed here."

_ALL_LISTED_TOWARD_RISK = "Every factor listed above pushed toward risk."

# Its own clause, not a degenerate loop. ScoreResponse's docstring records
# that an empty reason_codes list is a real case, not a defensive branch. A
# cardinality-driven template would emit "Zero factors ... strongest first:"
# followed by nothing, three individually-true clauses that read as a
# formatting failure.
_NO_FACTORS_CLAUSE = "No factor pushed toward risk."


class RenderError(RuntimeError):
    """
    The renderer could not state something truthfully, so it stated nothing.

    Raised rather than returning partial prose. serving/errors.py already
    fixes this posture for the explanation path: "It must not be downgraded to
    a warning and it must not return a decision without an explanation. Fail
    closed." A half-rendered notice is the failure mode this whole module
    exists to prevent.
    """


def _phrase_for(feature: str, value: str) -> str:
    """
    Plain-language phrase for one (feature, value) pair.

    Value-keyed lookup FIRST, so the two sentinel overrides (dti_n == 999.0,
    emp_length_ord == "nan") win over their column's ordinary phrase. Raises
    rather than falling back to the raw column name: a notice that says
    "emp_length_ord" has not explained anything, and silently degrading to it
    would let an unmapped category reach an applicant.
    """
    phrase = _VALUE_PHRASES.get((feature, value))
    if phrase is not None:
        return phrase
    if feature in VALUE_KEYED_FEATURES:
        raise RenderError(
            f"No approved phrase for {feature} = {value!r}. Value-keyed "
            "features need one phrase per allowed value; add it to "
            "_VALUE_PHRASES rather than letting a raw category reach a notice."
        )
    phrase = _FEATURE_PHRASES.get(feature)
    if phrase is None:
        raise RenderError(
            f"No approved phrase for feature {feature!r}. Every member of "
            "features.FEATURES needs one; see test_every_feature_has_a_phrase."
        )
    return phrase


def _decision_sentence(decision: str) -> str:
    """
    The opening sentence. Raises on an unmapped decision rather than emitting
    the raw wire token, which would read as "This application was APPROVE."
    """
    phrase = _DECISION_PHRASES.get(decision)
    if phrase is None:
        raise RenderError(
            f"No approved phrase for decision {decision!r}; "
            "see test_every_decision_value_has_a_phrase."
        )
    return f"This application was {phrase}."


def _comparison_word(p_calibrated: float, threshold: float) -> str:
    """
    "below" or "at or above", computed -- never hardcoded per decision.

    explain.py decides `"REJECT" if p_cal >= threshold else "APPROVE"`. The
    boundary case p_cal == threshold is therefore a REJECT that is NOT above
    the threshold, so the word "above" would be false there. No output of the
    shipped 52-step calibrator equals the threshold (measured: 0 of 52, the
    nearest are 0.2414... and 0.2648...), which makes "above" true BY ARTIFACT
    and not by construction -- exactly the shape of claim this repo keeps
    getting wrong. "at or above" mirrors `>=` and survives a retrain.
    """
    return "below" if p_calibrated < threshold else "at or above"


def explanation_fragments(response: ScoreResponse) -> list[str]:
    """
    The fixed strings that MUST appear, in this order, in the rendered output.

    This is the guard's specification and it is derived from the same response
    the renderer reads, so a factor the renderer drops is still listed here and
    the containment check fails. Ordered rather than set-membership: set
    containment misses rank inversion (measured -- 16 of 17 facts caught, rank
    order the one miss), and rank order is the regulated artifact, not a
    presentational choice.
    """
    fragments = [
        _decision_sentence(response.decision),
        _format_probability(response.p_calibrated),
        _format_probability(response.threshold),
    ]
    # `scale` names the axis the CONTRIBUTIONS live on. With no contributions
    # it has nothing to qualify, and requiring it here would force the empty
    # case to print an axis for numbers it does not show -- a fact stated
    # about nothing. It is rendered exactly when it has a referent.
    if response.reason_codes:
        fragments.append(_SCALE_PHRASES[response.scale])
    for code in response.reason_codes:
        fragments.append(_phrase_for(code.feature, code.value))
        fragments.append(f"{code.feature} = {code.value}")
        fragments.append(_format_contribution(code.contribution_log_odds))
    fragments.append(_ONE_SIDED_CLAUSE)
    return fragments


def _format_probability(value: float) -> str:
    return f"{value:.{PROBABILITY_PLACES}f}"


def _format_contribution(value: float) -> str:
    return f"{value:+.{CONTRIBUTION_PLACES}f}"


def render_explanation(response: ScoreResponse) -> str:
    """
    Render one scored applicant as plain-language prose.

    Pure: no I/O, no network, no clock, no randomness. The same ScoreResponse
    renders to the same bytes forever, which is what makes the output
    reviewable in a diff and assertable in a test.

    Self-checks before returning. The renderer is now the thing that must not
    lie, so it verifies its own output against explanation_fragments() and
    raises rather than returning prose that lost a fact. Ordered containment
    catches alteration and omission; it does NOT catch addition, and nothing
    here adds -- which is only true because no model writes any part of this.
    """
    if response.scale not in _SCALE_PHRASES:
        raise RenderError(
            f"No approved phrase for scale {response.scale!r}. explain.py's "
            f"CONTRIBUTION_SCALE is {CONTRIBUTION_SCALE!r}; if it changed, "
            "this module is describing the wrong axis."
        )

    comparison = _comparison_word(response.p_calibrated, response.threshold)
    lines = [
        _decision_sentence(response.decision),
        "",
        f"The calibrated probability of default is "
        f"{_format_probability(response.p_calibrated)}, which is {comparison} "
        f"the decision threshold of {_format_probability(response.threshold)}.",
        "",
        "Data the decision used:",
        f"  your FICO score = {response.credit_report.fico_n:.{FICO_PLACES}f} "
        f"({response.credit_report.fico_version}, from {response.credit_report.bureau})",
        "",
    ]

    if response.reason_codes:
        lines.append(f"{_FACTORS_HEADER} {_SCALE_PHRASES[response.scale]}:")
        for code in response.reason_codes:
            lines.append(
                f"  {code.rank}. {_phrase_for(code.feature, code.value)} "
                f"({code.feature} = {code.value}, "
                f"{_format_contribution(code.contribution_log_odds)})"
            )
        lines += ["", f"{_ALL_LISTED_TOWARD_RISK} {_ONE_SIDED_CLAUSE}"]
    else:
        lines.append(f"{_NO_FACTORS_CLAUSE} {_ONE_SIDED_CLAUSE}")

    text = "\n".join(lines)

    missing = _first_out_of_order(text, explanation_fragments(response))
    if missing is not None:
        raise RenderError(
            f"Rendered output lost or reordered a required fragment: "
            f"{missing!r}. The renderer and explanation_fragments() disagree, "
            "which means one of them is describing a response the other is not."
        )
    return text


def _first_out_of_order(text: str, fragments: list[str]) -> str | None:
    """
    First fragment that is absent, or present but out of order. None if clean.

    Returns the offending fragment rather than a bool so the failure names
    itself -- the same reason _assert_serving_enums_match_artifact prints both
    category sets instead of raising "mismatch".
    """
    position = -1
    for fragment in fragments:
        found = text.find(fragment, position + 1)
        if found == -1:
            return fragment
        position = found
    return None
