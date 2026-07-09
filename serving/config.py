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
# 0.01) (evaluate.py:103) and maximizes the regret objective on Validation. The
# winner is element [20] of that array, which is NOT 0.25:
#
#     >>> float(np.arange(0.05, 0.95, 0.01)[20])
#     0.25000000000000006
#     >>> float(np.arange(0.05, 0.95, 0.01)[20]) == 0.25
#     False
#
# It was logged as best_threshold by MLflow run cca4c361615c460b999ce1a73bd46439.
# src.explain.DEFAULT_EXPLAIN_THRESHOLD is the literal 0.25 (explain.py:119) and
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
# load_model_artifact() enforces (train.py:495) and every test that reads it.
# Deliberately not done here.
# ---------------------------------------------------------------------------
SELECTED_THRESHOLD: float = 0.25000000000000006

# Paths, derived from calibrate.py's DEFAULT_MODEL_PATH (calibrate.py:40) rather
# than spelled again. explain_applicants() defaults its calibrator to
# model_path.parent / "isotonic_calibrator.pkl" (explain.py:472-476); this
# names the same file explicitly so a misconfiguration fails at startup instead
# of resolving silently on the first request.
MODEL_PATH: Path = DEFAULT_MODEL_PATH
CALIBRATOR_PATH: Path = DEFAULT_MODEL_PATH.parent / "isotonic_calibrator.pkl"
