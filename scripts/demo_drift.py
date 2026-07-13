"""
Drift demo: turn MockBureau's FICO knob and watch the existing monitor catch it.

This is a standalone demonstration, not part of the pipeline. It wires two
components that already exist -- serving/bureau.py's MockBureau (a deterministic
credit-bureau mock with a mean_fico / std_fico knob) and
pipelines/drift_check.py's distribution monitor (PSI/KS/tripwire/sentinel) --
and shows the monitor going quiet on a no-drift market and firing on a shifted
one. It changes NO existing code: every drift number printed below is computed
by drift_check.drift_metrics / evaluate_alarms, not by this script.

What it shows
-------------
One reference population (mean_fico=700) is compared against two current
populations by the SAME monitor call:

  CONTROL   current mean_fico=700  -- same market, monitor must stay quiet.
  DOWNTURN  current mean_fico=650  -- credit quality slips, monitor must fire.

The point is discrimination, not just "the input moved": the monitor is silent
on FICO when nothing drifted and loud on FICO when it did. The control run is
what makes the downturn alarm meaningful -- a monitor that fired on everything
would prove nothing.

Why the knob isolates FICO
--------------------------
MockBureau.fetch(applicant_id) is deterministic: it hashes applicant_id into a
seed, so the same applicant_id always draws the same report. fico_n is a
Normal(mean_fico, std_fico) draw off that seed; dti_n is a separate uniform
draw off the SAME seed, independent of mean_fico. So reusing one applicant_id
set across the 700 and 650 current batches shifts every applicant's fico_n by
exactly -50 while leaving dti_n byte-identical. dti_n is therefore a negative
control: its PSI/KS must stay ~0 across both runs, proving the monitor reports
drift only where the population actually moved.

The dti tripwire is a mock artifact (disclosed)
-----------------------------------------------
drift_check's tripwire_share and sentinel_rate signals are dti-specific: they
probe the real 2016+ DTI reporting regime (values in (100, 1000], the 999
sentinel). MockBureau's dti_n is a crude uniform[0, DTI_MAX_REAL=1000)
placeholder, nothing like the real 0-60 dti band, so ~90% of it lands in the
tripwire band and the tripwire reads ~0.90 in BOTH runs. That firing is an
artifact of the mock's dti, not a drift signal, and -- being identical in both
runs -- it is not what the FICO knob moves. This demo's claim rests on the
general-purpose distribution signals (PSI/KS on fico_n), which DO transition
from quiet to firing when the knob turns. The raw tripwire value is printed so
nothing is hidden, and the alarm list is split into FICO alarms vs. the rest.

Determinism
-----------
No randomness lives in this script: applicant_ids are fixed strings and
MockBureau seeds its draws from them, so every run prints byte-identical
numbers. Reproduce with:

    uv run python scripts/demo_drift.py

Also served over HTTP
---------------------
serving/app.py's POST /drift is a wrapper over drift_report() below -- the same
sampling path, the same monitor() call, the same drift_check.py functions, with
`mean_fico` as the knob instead of the two settings main() hardcodes. It exists
so a browser can turn the knob and watch PSI move. It computes no drift metric
of its own; if it did, the number on the screen would be a number the monitor
never produced, which is the exact failure this repo is about.

That route is NOT part of the production image, and cannot be. This module
imports pipelines/drift_check.py, which imports mlflow and (via
pipelines/training_flow.py) metaflow -- and the serving image installs neither
(Dockerfile: `uv sync --no-group training`) and copies neither scripts/ nor
pipelines/ (.dockerignore). So serving/app.py imports this lazily and mounts
/drift only where it is importable: present in local dev, absent from the
container. See DRIFT_DEMO_AVAILABLE (serving/app.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ lives beside src/ and pipelines/, not inside them -- the same
# sys.path shim pipelines/drift_check.py and training_flow.py use, so this file
# runs as `python scripts/demo_drift.py` and still imports from src/ and
# pipelines/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from pipelines.drift_check import (  # noqa: E402
    DEFAULT_ALARM_THRESHOLDS,
    drift_metrics,
    evaluate_alarms,
)
from serving.bureau import MockBureau  # noqa: E402

# --- Demo parameters -------------------------------------------------------------
N = 2000                    # applicants per batch
MEAN_NORMAL = 700.0         # "normal market" FICO center
MEAN_DOWNTURN = 650.0       # "downturn market" FICO center -- the knob's other setting
YEAR_REF = 2015             # window label for the reference batch (a monitor slice key,
YEAR_CUR = 2016             # NOT a claim about real LendingClub 2015/2016 data)

# The two credit fields MockBureau reports. dti_n is mandatory for
# drift_metrics (it is the column the monitor was built for) and doubles as the
# negative control here; fico_n is what this demo actually drifts.
MONITORED = ("fico_n", "dti_n")


def reference_ids() -> list[str]:
    """The reference population's applicant ids. Fixed strings, never drawn."""
    return [f"ref-{i:04d}" for i in range(N)]


def current_ids() -> list[str]:
    """
    The current population's applicant ids -- the SAME set at every mean_fico.

    Sharing one id set across the knob's settings is what makes dti_n a negative
    control: MockBureau seeds both draws off the same hash of the id, and only
    fico_n reads mean_fico, so holding the ids fixed shifts every fico_n by the
    knob while leaving every dti_n byte-identical.
    """
    return [f"cur-{i:04d}" for i in range(N)]


def build_batch(bureau: MockBureau, ids: list[str], issue_year: int) -> pd.DataFrame:
    """
    Fetch every id through `bureau` and shape the reports into the flat frame
    drift_metrics expects: one row per applicant, columns fico_n / dti_n /
    issue_year. This is the ONLY bridging code -- turning CreditReport objects
    into the (column, issue_year) frame the monitor slices. No metric is
    computed here.
    """
    reports = [bureau.fetch(i) for i in ids]
    return pd.DataFrame(
        {
            "fico_n": [r.fico_n for r in reports],
            "dti_n": [r.dti_n for r in reports],
            "issue_year": issue_year,
        }
    )


def monitor(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    """
    The monitor call itself: drift_check.py's drift_metrics + evaluate_alarms,
    invoked exactly as the pipeline invokes them, on default thresholds.

    Extracted out of run_monitor() so that the HTTP endpoint (serving/app.py's
    POST /drift) reaches the monitor THROUGH THIS FUNCTION rather than
    assembling its own frame and issuing its own drift_metrics call. Those would
    be four lines each -- which columns, which reference window, which current
    window, what concat order -- and four lines are more than enough room for the
    endpoint's monitor and the demo's monitor to answer differently. There is one
    of them.
    """
    frame = pd.concat([reference, current], ignore_index=True)
    metrics = drift_metrics(
        frame,
        columns=MONITORED,
        reference_years=(YEAR_REF, YEAR_REF),
        current_years=(YEAR_CUR,),
    )
    return metrics, evaluate_alarms(metrics)  # default thresholds -- nothing relaxed


def feature_drift(metrics: dict[str, float], column: str) -> dict:
    """
    One monitored column's distribution signals, and whether the monitor alarmed
    on THEM.

    `alarmed` is not recomputed from a threshold here. It is drift_check.py's own
    evaluate_alarms(), run a second time over exactly this column's two keys --
    so the answer comes from the alarm rule itself, at the alarm rule's own
    thresholds. Writing `psi > 0.25` in this function would be a second copy of
    that rule, and a second copy is the one that goes stale.

    Selecting this column's alarms BY KEY, rather than by searching the full
    alarm list for the column's NAME, is the part that took a bug to get right.
    The dti tripwire's alarm string spells out "...training years hold exactly
    zero dti_n in (100, 1000]..." in its EXPLANATORY PROSE, so `"dti_n" in alarm`
    is True for an alarm that is not a dti_n distribution alarm at all -- and
    dti_n is the NEGATIVE CONTROL, so it would have reported itself as firing in
    the one place its silence is the whole point. (run_monitor's fico/other split
    below is safe from this only because "fico_n" happens not to appear in any
    other alarm's text. That is luck, not design, and not a thing to build a
    response field on.)
    """
    psi_key, ks_key = f"psi_{column}_{YEAR_CUR}", f"ks_{column}_{YEAR_CUR}"
    own = {psi_key: metrics[psi_key], ks_key: metrics[ks_key]}
    return {
        "psi": metrics[psi_key],
        "ks": metrics[ks_key],
        "alarmed": bool(evaluate_alarms(own)),
    }


def drift_report(mean_fico: float) -> dict:
    """
    Turn the FICO knob to `mean_fico`, run the real monitor, return everything it
    said. The demo's whole argument, as data instead of as an ASCII table.

    This is the ONE sampling path -- reference population at MEAN_NORMAL, current
    population at `mean_fico`, over the same two fixed id sets main() uses -- and
    it reaches the metrics through monitor(), the one monitor call. serving/'s
    POST /drift is a wrapper over this function and nothing else; it computes no
    PSI, no KS and no alarm of its own.

    Deterministic: no RNG lives on this path (fixed ids, hash-seeded MockBureau),
    so the same mean_fico returns byte-identical numbers on every call, in every
    process, forever. That is what lets a client drag a slider and get a curve
    rather than a shimmer.

    The raw `metrics` and `alarms` are returned ALONGSIDE the per-feature view,
    not replaced by it: `features` is a re-keying of `metrics`, and shipping the
    thing it was re-keyed from is what makes the re-keying checkable. It is also
    where the disclosed mock artifact lives -- the dti tripwire fires at ~0.90 in
    every run because MockBureau's dti_n is a crude uniform draw (see the module
    docstring), and that alarm is in `alarms` rather than filtered out of it.
    """
    reference = build_batch(MockBureau(mean_fico=MEAN_NORMAL), reference_ids(), YEAR_REF)
    current = build_batch(MockBureau(mean_fico=mean_fico), current_ids(), YEAR_CUR)
    metrics, alarms = monitor(reference, current)

    return {
        "mean_fico": float(mean_fico),
        "reference_mean_fico": MEAN_NORMAL,
        "reference_year": YEAR_REF,
        "current_year": YEAR_CUR,
        "n_reference": int(metrics["n_reference"]),
        "n_current": int(metrics[f"n_{YEAR_CUR}"]),
        # What the batches actually DREW, not what the knob asked for. They differ:
        # MockBureau clips each draw into [FICO_MIN, FICO_MAX], so pushing the knob
        # toward either bound pulls the observed mean off the requested one. A demo
        # that showed only the requested mean would hide its own clipping.
        "observed_mean_fico_reference": float(reference["fico_n"].mean()),
        "observed_mean_fico_current": float(current["fico_n"].mean()),
        "features": {column: feature_drift(metrics, column) for column in MONITORED},
        "thresholds": dict(DEFAULT_ALARM_THRESHOLDS),
        "alarms": alarms,
        "metrics": metrics,
    }


def run_monitor(reference: pd.DataFrame, current: pd.DataFrame, title: str) -> dict:
    """
    Feed one reference/current pair to the REAL monitor and print its verdict.

    The metrics come from monitor() above -- drift_check.py's own functions,
    called exactly as the pipeline calls them. This function prints them.
    """
    metrics, alarms = monitor(reference, current)

    ref_fico = reference["fico_n"].mean()
    cur_fico = current["fico_n"].mean()
    psi_fico = metrics[f"psi_fico_n_{YEAR_CUR}"]
    ks_fico = metrics[f"ks_fico_n_{YEAR_CUR}"]
    psi_dti = metrics[f"psi_dti_n_{YEAR_CUR}"]
    ks_dti = metrics[f"ks_dti_n_{YEAR_CUR}"]
    tripwire = metrics[f"tripwire_share_{YEAR_CUR}"]
    sentinel = metrics[f"sentinel_rate_{YEAR_CUR}"]

    # Split the monitor's own alarms into the FICO distribution signals (the
    # demo's subject) vs. everything else (here: the dti tripwire artifact).
    fico_alarms = [a for a in alarms if "fico_n" in a]
    other_alarms = [a for a in alarms if "fico_n" not in a]

    line = "=" * 74
    print("\n" + line)
    print(f"{title}")
    print(line)
    print(f"  mean fico_n   reference {ref_fico:7.2f}  ->  current {cur_fico:7.2f}  "
          f"(shift {cur_fico - ref_fico:+.2f})")
    print(f"  fico_n drift  PSI={psi_fico:6.4f} (>{DEFAULT_ALARM_THRESHOLDS['psi']} alarms)   "
          f"KS={ks_fico:6.4f} (>{DEFAULT_ALARM_THRESHOLDS['ks']} alarms)")
    print(f"  dti_n  drift  PSI={psi_dti:6.4f}          "
          f"KS={ks_dti:6.4f}          <- negative control (knob left dti_n untouched)")
    print(f"  dti_n  tripwire_share={tripwire:.4f}  sentinel_rate={sentinel:.4f}  "
          f"<- mock artifact (uniform dti_n), constant across runs")
    if fico_alarms:
        print(f"\n  FICO ALARMS ({len(fico_alarms)}):")
        for a in fico_alarms:
            print(f"    ALERT: {a}")
    else:
        print("\n  FICO ALARMS: none -- monitor stays quiet, no FICO drift.")
    if other_alarms:
        print(f"  other alarms ({len(other_alarms)}, dti mock artifact, same in both runs):")
        for a in other_alarms:
            print(f"    (artifact) {a}")
    print(line)

    return {
        "title": title,
        "ref_fico": ref_fico,
        "cur_fico": cur_fico,
        "psi_fico": psi_fico,
        "ks_fico": ks_fico,
        "fico_alarms": fico_alarms,
        "metrics": metrics,
        "alarms": alarms,
    }


def main() -> None:
    cur_ids = current_ids()

    # One reference population at mean_fico=700, labeled YEAR_REF.
    reference = build_batch(MockBureau(mean_fico=MEAN_NORMAL), reference_ids(), YEAR_REF)

    # Two current populations sharing the SAME applicant_ids, labeled YEAR_CUR:
    # only the FICO knob differs (700 vs 650), so between them fico_n shifts by
    # exactly -50 per applicant and dti_n is byte-identical.
    current_normal = build_batch(MockBureau(mean_fico=MEAN_NORMAL), cur_ids, YEAR_CUR)
    current_downturn = build_batch(MockBureau(mean_fico=MEAN_DOWNTURN), cur_ids, YEAR_CUR)

    print("\nDRIFT DEMO -- MockBureau FICO knob vs. pipelines/drift_check.py")
    print(f"reference: {N} applicants @ mean_fico={MEAN_NORMAL:.0f}   "
          f"current: {N} applicants @ mean_fico=700 (control) / 650 (downturn)")
    print("all PSI/KS/tripwire numbers below are drift_check.py's, not this script's")

    control = run_monitor(reference, current_normal, "CONTROL   current mean_fico=700 (same market)")
    downturn = run_monitor(reference, current_downturn, "DOWNTURN  current mean_fico=650 (credit slips)")

    # --- Side-by-side verdict ----------------------------------------------------
    line = "=" * 74
    print("\n" + line)
    print("VERDICT -- turning ONLY the FICO knob (700 -> 650)")
    print(line)
    print(f"  {'':22}{'CONTROL (700)':>16}{'DOWNTURN (650)':>18}")
    print(f"  {'PSI fico_n':22}{control['psi_fico']:>16.4f}{downturn['psi_fico']:>18.4f}")
    print(f"  {'KS  fico_n':22}{control['ks_fico']:>16.4f}{downturn['ks_fico']:>18.4f}")
    print(f"  {'FICO alarms fired':22}{len(control['fico_alarms']):>16}{len(downturn['fico_alarms']):>18}")
    print(line)
    verdict = (
        "PASS: monitor quiet on the unchanged market, fired on the shifted one."
        if not control["fico_alarms"] and downturn["fico_alarms"]
        else "UNEXPECTED: see per-run output above."
    )
    print(f"  {verdict}")
    print(line + "\n")


if __name__ == "__main__":
    main()
