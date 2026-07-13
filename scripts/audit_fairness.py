"""
Run the real three-layer fairness audit and freeze its output as a small JSON
artifact the serving layer can hand to a client.

Why an artifact instead of an endpoint that computes
---------------------------------------------------
pipelines/drift_check.py could be wrapped live (serving/app.py's POST /drift):
the mock bureau generates its own applicants, so a request costs ~0.4s and a
slider works. The fairness audit has neither property.

  1. It needs the data. run_fairness_audit() -> load_raw() -> the 167 MB
     assessment CSV. That file is the FIRST line of .dockerignore, because the
     brief is explicit that it must not be redistributed. So the audit cannot
     run inside the deployed image -- not as a size trade-off we could reverse,
     the way /drift's pipelines/ exclusion was. It is a constraint on what may
     be shipped at all.
  2. It costs ~40s. Measured on the real CSV: 2.7s to load, ~28s for
     run_fairness_audit() (Layer 3 retrains two full LightGBMs on 454k rows;
     Layer 1 bootstraps 2,000 resamples per state), ~10.5s more to put CIs on
     both sides of the ablation. No amount of caching makes that a live route.

What IS shippable is the OUTPUT: ~50 Equal-Opportunity ratios with confidence
intervals, a threshold sweep, and an ablation summary. Derived aggregates, not
the dataset. That is what this script writes, and what GET /fairness serves --
the same discipline GET /calibrator already follows (read the real artifact;
never recompute a copy of it in the serving layer).

And it has to be committed, not just built. models/ and data/*.csv are BOTH
gitignored, so a fresh clone can neither serve the model nor regenerate this
audit. If the output were a build artifact, the fairness evidence would exist
only on the machine that ran it. Committing it is what lets the claim travel
with the repo and be reviewed in a diff.

The staleness problem, and the gate
-----------------------------------
A frozen artifact can go stale against the model in a way the calibrator bundle
cannot: retrain, and this JSON still cheerfully reports MS at 0.7448. That is
exactly the "say != do" drift this repo exists to prevent, and it would be
self-inflicted.

The repo already owns the mechanism. src/calibrate.py's load_calibrator()
refuses a calibrator fit against a different model instance -- it binds on
trained_at. This artifact binds the same way: it records the trained_at of the
model it audited, and serving/fairness.py compares that against the SHIPPED
bundle's before serving a single number. On mismatch, /fairness refuses (409)
and returns no ratios at all. A client cannot draw what it was not given.

Fail-closed on the numbers, fail-open on the service: a stale audit must not
take /score down with it. The fairness audit is a REPORTING signal (blue in
docs/architecture.html), not a gate.

Run:
    uv run python scripts/audit_fairness.py            # -> models/fairness_audit.json
    uv run python scripts/audit_fairness.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from serving.artifacts import load_bundle
from src.data_loader import load_raw, temporal_split
from src.fairness import (
    ABLATION_THRESHOLD,
    EO_THRESHOLD,
    MIN_N,
    N_BOOT,
    SWEEP_THRESHOLDS,
    WATCH_STATES,
    audit_layer1,
    run_fairness_audit,
)

ARTIFACT_PATH = Path("models/fairness_audit.json")

# Bumped when the artifact's SHAPE changes. serving/fairness.py refuses a
# version it does not know, rather than silently reading absent keys as None.
SCHEMA_VERSION = 1


def _records(frame: pd.DataFrame) -> list[dict]:
    """
    DataFrame -> JSON rows, with numpy scalars cast to Python types.

    Deliberately NOT frame.to_json(). Pandas defaults to double_precision=10 and
    silently rounds every float to ten decimal places on the way out. That is
    harmless for an EO ratio, and fatal for exactly one number in this artifact:
    SELECTED_THRESHOLD is 0.25000000000000006, and to_json emits it as 0.25 --
    a genuinely different float (serving/config.py spends a comment on the
    difference). The sweep row for the operating point then stops == matching the
    operating point, and a client looking it up finds nothing and quietly falls
    back to the nearest row, reporting the approval rate at 0.26 as if it were
    the one we decide at.

    Caught by asserting the lookup rather than eyeballing the JSON. .to_dict()
    does no rounding; .item() unwraps numpy scalars so json.dumps will take them.
    """
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def build_audit(bundle=None) -> dict:
    """
    Run the REAL audit and shape its output for the wire.

    Every number here comes out of src/fairness.py. This function selects and
    re-keys; it computes no fairness metric of its own. The one thing it does
    that run_fairness_audit() does not is feed Layer 3's two Test frames back
    through the REAL audit_layer1() -- which is why src/fairness.py returns
    them (commit 18daa36). That is a reuse of the audit, not a second
    bootstrap implementation.
    """
    bundle = bundle if bundle is not None else load_bundle()
    shipped_features = list(bundle.booster.feature_name())

    splits = temporal_split(load_raw())

    # Layer 1 audits the shipped model at the threshold /score ACTUALLY decides
    # at -- bundle.threshold, i.e. serving/config.py's SELECTED_THRESHOLD (the
    # literal 0.25000000000000006, not 0.25). Auditing at any other cutoff
    # would answer a question nobody asked: "would the model be fair at an
    # operating point we do not use?"
    # The sweep must CONTAIN the operating point. src/fairness.py's notebook-era
    # SWEEP_THRESHOLDS is [0.12 ... 0.22, 0.26, 0.30] and does not include
    # 0.25000000000000006, so a client asking "how many good applicants does the
    # shipped model approve at the cutoff it actually uses?" would have to read
    # the NEAREST row (0.26) and call it the operating one. That is a small
    # say != do -- the answer would be off by a real amount (82.4% at 0.26) and
    # labelled as if it were exact.
    #
    # audit_layer2 takes its thresholds as a parameter, so the fix is to pass the
    # operating point in rather than to round to it. src/fairness.py is untouched;
    # its constant remains the notebook's.
    sweep = sorted(set(SWEEP_THRESHOLDS) | {float(bundle.threshold)})

    result = run_fairness_audit(
        splits=splits,
        audit_threshold=bundle.threshold,
        sweep_thresholds=sweep,
    )
    layer1, layer2, layer3 = result["layer1"], result["layer2"], result["layer3"]

    # The ablation, with an interval on BOTH sides. audit_layer1 is the real
    # Layer-1 bootstrap; it is simply pointed at each variant's Test frame in
    # turn, at Layer 3's own threshold (0.22, deliberately not the operating
    # threshold -- see ABLATION_THRESHOLD in src/fairness.py).
    ci_with = audit_layer1(layer3["fair_df_with_state"], threshold=ABLATION_THRESHOLD)
    ci_no = audit_layer1(layer3["fair_df_no_state"], threshold=ABLATION_THRESHOLD)
    ablation_ci = ci_with.merge(ci_no, on="state", suffixes=("_with_state", "_no_state"))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # --- provenance: what serving/fairness.py binds against -------------
        "model": {
            # THE gate. serving/fairness.py compares this against the shipped
            # bundle's model_trained_at and refuses on mismatch.
            "trained_at": bundle.model_trained_at,
            "calibrator_trained_at": bundle.calibrator_trained_at,
            "best_iteration": bundle.best_iteration,
            # The fairness conclusion, executed and checkable: the model this
            # audit ran against does not have addr_state as a feature. A client
            # does not have to take that on trust from prose.
            "features": shipped_features,
            "includes_addr_state": "addr_state" in shipped_features,
        },
        # --- the constants the audit actually used, not re-typed here -------
        "constants": {
            "eo_threshold": EO_THRESHOLD,
            "min_n": MIN_N,
            "n_boot": N_BOOT,
            # What the sweep ACTUALLY ran at -- SWEEP_THRESHOLDS plus the
            # operating point -- not the module constant it was derived from.
            # Shipping the constant here while having swept something else would
            # be the same defect in miniature.
            "sweep_thresholds": sweep,
            "ablation_threshold": ABLATION_THRESHOLD,
            "watch_states": list(WATCH_STATES),
        },
        # --- Layer 1: the SHIPPED model, at the SHIPPED operating point -----
        "layer1": {
            "threshold": float(bundle.threshold),
            "states": _records(layer1),
            "n_confirmed": int(
                layer1["verdict"].str.startswith("confirmed", na=False).sum()
            ),
        },
        # --- Layer 2: does the all-clear survive tightening? ----------------
        "layer2": {
            "rows": _records(layer2),
        },
        # --- Layer 3: the counterfactual, now with intervals ----------------
        "layer3": {
            "threshold": float(layer3["threshold"]),
            "auc_with_state": float(layer3["auc_with_state"]),
            "auc_no_state": float(layer3["auc_no_state"]),
            "auc_cost": float(layer3["auc_cost"]),
            "base_approval_with_state": float(layer3["base_approval_with_state"]),
            "base_approval_no_state": float(layer3["base_approval_no_state"]),
            # The 6 watch states with fairness.py's own verdict column.
            "watch": _records(layer3["states"]),
            # Every state with >= MIN_N good applicants, both variants, with
            # CIs. This is the before/after the point estimates cannot support.
            "states": _records(ablation_ci),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-run the audit and compare against the committed artifact. "
             "Writes nothing; exits non-zero if the numbers have moved.",
    )
    parser.add_argument("--out", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args(argv)

    audit = build_audit()

    if args.check:
        if not args.out.exists():
            print(f"MISSING: {args.out} does not exist.", file=sys.stderr)
            return 1
        committed = json.loads(args.out.read_text())
        # generated_at is wall-clock and must not be compared.
        fresh = {k: v for k, v in audit.items() if k != "generated_at"}
        stored = {k: v for k, v in committed.items() if k != "generated_at"}
        if fresh == stored:
            print(f"OK: {args.out} matches a fresh run of the real audit.")
            return 0
        print(f"DRIFTED: {args.out} does not match a fresh run.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    size_kb = args.out.stat().st_size / 1024

    l1, l3 = audit["layer1"], audit["layer3"]
    print(f"\nwrote {args.out}  ({size_kb:.1f} KB)")
    print(f"  audited model trained_at : {audit['model']['trained_at']}")
    print(f"  Layer 1 @ {l1['threshold']:.4f} : {len(l1['states'])} states, "
          f"{l1['n_confirmed']} confirmed")
    print(f"  Layer 3 @ {l3['threshold']:.2f}   : AUC {l3['auc_with_state']:.4f} "
          f"-> {l3['auc_no_state']:.4f}  (cost {l3['auc_cost']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
