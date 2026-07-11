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


def run_monitor(reference: pd.DataFrame, current: pd.DataFrame, title: str) -> dict:
    """
    Feed one reference/current pair to the REAL monitor and print its verdict.

    drift_metrics + evaluate_alarms are drift_check.py's own functions, called
    exactly as the pipeline calls them; this function only arranges the two
    year-labeled batches into one frame and prints the result.
    """
    frame = pd.concat([reference, current], ignore_index=True)
    metrics = drift_metrics(
        frame,
        columns=MONITORED,
        reference_years=(YEAR_REF, YEAR_REF),
        current_years=(YEAR_CUR,),
    )
    alarms = evaluate_alarms(metrics)  # default thresholds -- nothing relaxed

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
    ref_ids = [f"ref-{i:04d}" for i in range(N)]
    cur_ids = [f"cur-{i:04d}" for i in range(N)]

    # One reference population at mean_fico=700, labeled YEAR_REF.
    reference = build_batch(MockBureau(mean_fico=MEAN_NORMAL), ref_ids, YEAR_REF)

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
