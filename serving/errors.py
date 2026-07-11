"""
The error taxonomy, and why each failure lands where it does.

422 -- the request is malformed. Pydantic raises before any artifact is
      touched: a missing field, a string where a float belongs, a value out of
      band, an extra field, an unknown purpose, an unmapped emp_length.
      FastAPI's RequestValidationError handler produces this; no code here.

      `emp_length: null` is NOT in this class. It is a 200. See ScoreRequest.

500 -- the request was fine and the MODEL is not. One case matters:
      _assert_additivity raising ValueError (explain.py:158) because
      base + sum(contributions) does not reconstruct the booster's raw margin
      to within ADDITIVITY_ATOL = 1e-9 (explain.py:155). The applicant did
      nothing wrong; the explanation does not reconstruct the score. It must
      not be downgraded to a warning and it must not return a decision without
      an explanation. Fail closed, log the guard's full message, return a body
      that leaks none of it.

      A CreditBureau.fetch() failure is NOT handled here -- deliberately, not
      by oversight. MockBureau (the only implementation wired in today)
      performs no I/O and cannot fail, so there is no reachable case to catch
      yet; a real bureau client's failure mode is out of scope until it is
      actually wired in. See docs/data-decisions.md's Phase 1 bureau-wiring
      entry for the recorded deferral.

503 -- in production, NOTHING.

      A missing pickle (FileNotFoundError), a feature-contract mismatch
      (load_model_artifact, model_io.py:323), a stale calibrator (load_calibrator,
      calibrate.py:56), or a schema/category divergence
      (_assert_serving_enums_match_artifact) are all DEPLOYMENT errors with
      nothing to retry. The right behavior is for lifespan to raise, uvicorn to
      exit non-zero, and the orchestrator to show a crash loop. A process that
      never serves emits no status code at all.

      The 503 branch below therefore exists for two reachable cases: an app
      object constructed without running startup for the model artifacts, or
      for the bureau client -- which is how tests exercise each unloaded
      state. Neither is dead code and neither is a production path. Saying so
      is cheaper than discovering it later.
"""

from __future__ import annotations

# The 500 body. Deliberately says nothing about the model, the applicant, the
# tolerance, or which feature failed. That belongs in the log, not on the wire.
ADDITIVITY_FAILURE_DETAIL = (
    "Explanation failed an internal consistency check. No decision was returned."
)

# The 503 body for a missing model bundle. See the module docstring on why
# this is not a production path.
ARTIFACTS_UNAVAILABLE_DETAIL = "Model artifacts are not loaded."

# The 503 body for a missing bureau client -- the same "unloaded state,
# reachable only in tests" case as ARTIFACTS_UNAVAILABLE_DETAIL above, kept as
# a separate message because a bureau is not a model artifact and saying so
# would misname the actual gap.
BUREAU_UNAVAILABLE_DETAIL = "Credit bureau is not loaded."
