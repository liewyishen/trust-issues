"""
Serving configuration: the threshold, and where the artifacts live.

The threshold is here rather than in an artifact because it is not in an
artifact. That is a gap, not a design; see SELECTED_THRESHOLD's comment.
"""

from __future__ import annotations

from pathlib import Path

from src.calibrate import DEFAULT_MODEL_PATH

# ---------------------------------------------------------------------------
# SELECTED_THRESHOLD -- the cutoff the pipeline actually chose.
#
# select_threshold() sweeps evaluate.DEFAULT_THRESHOLDS = np.arange(0.05, 0.95,
# 0.01) (evaluate.py) and maximizes the regret objective on Validation. The
# winner is element [20] of that array, which is NOT 0.25:
#
#     >>> float(np.arange(0.05, 0.95, 0.01)[20])
#     0.25000000000000006
#     >>> float(np.arange(0.05, 0.95, 0.01)[20]) == 0.25
#     False
#
# It was logged as best_threshold by MLflow run cca4c361615c460b999ce1a73bd46439.
# src.explain.DEFAULT_EXPLAIN_THRESHOLD is the literal 0.25 (explain.py) and
# the two disagree at p_cal exactly 0.25: this value rejects, that one approves.
# Under the SHIPPED isotonic calibrator that disagreement is currently
# unreachable -- 0.25 is not an attainable output of a 52-level step function
# (docs/explainability.md Section 4) -- but it becomes reachable the moment the
# calibrator is replaced with a continuous one. This module scores with the
# value the pipeline selected, not with the constant that reads more nicely.
#
# It is NOT packaged with the model. Neither pickle carries a threshold; it
# lives only in MLflow run history, so a service must take it as config or read
# it back from MLflow. A known gap, recorded in docs/design.md Section 5.
# Packaging it into the artifact would reopen the feature contract
# load_model_artifact() enforces (model_io.py) and every test that reads it.
# Deliberately not done here.
# ---------------------------------------------------------------------------
SELECTED_THRESHOLD: float = 0.25000000000000006

# Paths, derived from calibrate.py's DEFAULT_MODEL_PATH rather than spelled
# again. explain_applicants() defaults its calibrator to
# model_path.parent / "isotonic_calibrator.pkl" (explain.py); this
# names the same file explicitly so a misconfiguration fails at startup instead
# of resolving silently on the first request.
MODEL_PATH: Path = DEFAULT_MODEL_PATH
CALIBRATOR_PATH: Path = DEFAULT_MODEL_PATH.parent / "isotonic_calibrator.pkl"

# The frozen fairness audit (scripts/audit_fairness.py). It sits beside the
# model on purpose: serving/fairness.py binds it to that model by trained_at
# and refuses to serve it if they disagree, so the two belong in one place.
#
# Unlike its neighbours in models/, this one is COMMITTED -- models/ and
# data/*.csv are both gitignored, so a fresh clone can neither serve the model
# nor regenerate the audit (it needs the 167 MB CSV). If this file were a build
# output, the fairness evidence would exist only on the machine that ran it.
# See .gitignore's negation, and scripts/audit_fairness.py's docstring.
FAIRNESS_AUDIT_PATH: Path = DEFAULT_MODEL_PATH.parent / "fairness_audit.json"
