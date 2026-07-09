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

503 -- in production, NOTHING.

      A missing pickle (FileNotFoundError), a feature-contract mismatch
      (load_model_artifact, train.py:495), a stale calibrator (load_calibrator,
      calibrate.py:56), or a schema/category divergence
      (_assert_serving_enums_match_artifact) are all DEPLOYMENT errors with
      nothing to retry. The right behavior is for lifespan to raise, uvicorn to
      exit non-zero, and the orchestrator to show a crash loop. A process that
      never serves emits no status code at all.

      The 503 branch below therefore exists for one reachable case: an app
      object constructed without running startup, which is how tests exercise
      the unloaded state. It is not dead code and it is not a production path.
      Saying so is cheaper than discovering it later.
"""

from __future__ import annotations

# The 500 body. Deliberately says nothing about the model, the applicant, the
# tolerance, or which feature failed. That belongs in the log, not on the wire.
ADDITIVITY_FAILURE_DETAIL = (
    "Explanation failed an internal consistency check. No decision was returned."
)

# The 503 body. See the module docstring on why this is not a production path.
ARTIFACTS_UNAVAILABLE_DETAIL = "Model artifacts are not loaded."
