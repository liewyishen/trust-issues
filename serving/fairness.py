"""
Read the frozen fairness audit, and refuse to serve it if it is not about the
model we are actually shipping.

The audit itself lives in src/fairness.py and is run OFFLINE by
scripts/audit_fairness.py -- it needs the 167 MB assessment CSV (the first line
of .dockerignore, because the brief forbids redistributing it) and it takes
~40s. Neither is available or acceptable inside a request. What ships is its
OUTPUT: ~50 Equal-Opportunity ratios with bootstrap CIs, a threshold sweep, and
an ablation. Derived aggregates, not the dataset.

This module computes NO fairness metric. It opens a JSON file and checks one
thing.

------------------------------------------------------------------------------
The one thing: does this audit describe the shipped model?

GET /calibrator cannot go stale -- it reads the live bundle, so whatever it
returns IS what /score decides with. A frozen JSON has no such protection.
Retrain the model and this file still cheerfully reports Mississippi at 0.7448,
about a booster that no longer exists. That is precisely the "say != do" drift
this repo exists to prevent, and here it would be self-inflicted.

The repo already owns the fix. src/calibrate.py's load_calibrator() refuses a
calibrator that was fit against a different model instance -- it binds on
trained_at. This binds the same way, against the same field:

    audit["model"]["trained_at"]   vs   bundle.model_trained_at

On mismatch, /fairness returns 409 and NO RATIOS. Not the numbers with a
warning label attached -- a client that is handed numbers will draw them. The
only reliable way to stop a stale ratio being rendered as a current one is to
not send it.

Fail-closed on the numbers, fail-OPEN on the service. A bad fairness artifact
must not take /score down with it: the audit is a reporting signal (blue in
docs/architecture.html, alongside @explain), not a gate. So a missing or
unreadable file yields an unavailable route, never a boot failure -- the same
policy training_flow.py's explain step already applies, and for the same
reason. tests/test_serving.py is what guarantees the SHIPPED artifact is
present, parseable, and fresh; a runtime crash is not.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from serving.artifacts import ArtifactBundle
from serving.config import FAIRNESS_AUDIT_PATH

# The shapes this module knows how to read. An artifact written by a future
# scripts/audit_fairness.py with a different layout is REFUSED, not read
# half-way -- absent keys would otherwise surface as nulls in the response and
# render as blank chart axes rather than as an error.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})


@dataclass(frozen=True)
class FairnessAudit:
    """A loaded audit, plus why it could not be loaded if it could not."""

    audit: dict | None
    unavailable_reason: str | None

    @property
    def available(self) -> bool:
        return self.audit is not None


def load_fairness_audit(path: str | Path | None = None) -> FairnessAudit:
    """
    Load the frozen audit. Never raises -- reports.

    Returns a FairnessAudit whose `unavailable_reason` is None on success and a
    human-readable sentence otherwise. The caller (serving/app.py) turns that
    into a 404 with the reason in the body, so "there is no audit here" and
    "the audit is corrupt" are distinguishable by a human reading the response,
    without either of them being able to halt the service.
    """
    path = Path(path) if path is not None else FAIRNESS_AUDIT_PATH

    if not path.exists():
        return FairnessAudit(
            None,
            f"No fairness audit artifact at {path}. It is produced offline by "
            "scripts/audit_fairness.py, which needs the assessment CSV -- a file "
            "that deliberately never enters the serving image.",
        )

    try:
        audit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return FairnessAudit(None, f"Fairness audit at {path} is unreadable: {exc}")

    version = audit.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        return FairnessAudit(
            None,
            f"Fairness audit at {path} has schema_version {version!r}; this "
            f"service reads {sorted(SUPPORTED_SCHEMA_VERSIONS)}. Refusing to read "
            "it partially.",
        )

    return FairnessAudit(audit, None)


def audit_model_trained_at(audit: dict) -> str | None:
    """The trained_at of the model the audit actually ran against."""
    return audit.get("model", {}).get("trained_at")


def is_stale(audit: dict, bundle: ArtifactBundle) -> bool:
    """
    Does this audit describe some OTHER model than the one we are serving?

    The same binding src/calibrate.py's load_calibrator() applies between the
    calibrator and the booster, applied between the audit and the booster.

    A None on either side counts as stale. An artifact that cannot say which
    model it audited has not earned the benefit of the doubt -- absence of a
    provenance field is not evidence of a match.
    """
    audited = audit_model_trained_at(audit)
    shipped = bundle.model_trained_at
    if audited is None or shipped is None:
        return True
    return audited != shipped
