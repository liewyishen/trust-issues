# trust-issues

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Tests](https://img.shields.io/badge/tests-209%20passing-brightgreen.svg)
![Model](https://img.shields.io/badge/model-LightGBM-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<sub>Code is licensed under **MIT** (see [`LICENSE`](LICENSE)). The *dataset* is CC BY 4.0 — see [`data/README.md`](data/README.md).</sub>

**A loan risk model that did the boring parts right.**

> **Don't trust a number until you've tried to break it.**

Most loan-risk models fail quietly — in ways that never show up in a headline AUC. A target
hand-built from twenty loan-status categories. Post-decision fields like `recoveries` leaking the
future into features. A random split that lets the model peek ahead. An uncalibrated score no one
could price a loan with. Some of those mistakes make the score *look better*. This project has
trust issues with every one of them.

So it optimizes for the opposite of a leaderboard: a low, **honest** AUC over a high, contaminated
one. It takes the target from the data's own label, audits for leakage, splits by time, calibrates
its probabilities, and tries to break each fairness finding before believing it.

Run the whole chain — load, validate, train, calibrate, evaluate, fairness-audit — with a single
command:

```bash
uv run python pipelines/training_flow.py run
```

The headline number is a test AUC around **0.67** — and that's the point, not the apology. With
eight application-time features and no credit-bureau history, that is the ceiling. **A 0.67 you can
trust beats a 0.99 you can't.**

![Architecture](docs/architecture.png)

*[Interactive version](docs/architecture.html) — every label states the file:line it was verified
at; the page's Export-as-PNG button regenerates the image above.*

Colour carries meaning and nothing else: red is a gate that raises and halts the run, blue is
reporting that can halt nothing, grey dashed is code not wired into the flow. Two modules sit
outside the pipeline and are drawn that way rather than drawn as connected:
`pipelines/drift_check.py` is a separate manual entry point, and `src/explain.py` has no caller
except its tests. No calibrator edge runs to the fairness step — `run_fairness_audit()` never loads
the shipped calibrator, it refits its own isotonic (`src/fairness.py:586-587`) — so drawing that
edge would be a lie.

---

## What this project does right

| Stage | What |
|-------|------|
| **1. Problem framing** | Binary default target taken from the dataset's own label (fully paid vs. defaulted) — no invented horizon; the cost-sensitive operating threshold is set later, in evaluation (LGD 0.65 / margin 0.12), not assumed upfront |
| **2. Leakage-aware data prep** | A pandera schema gate plus four leakage sentinels wired into the pipeline: two blocklist checks that raise (forbidden post-decision fields in the feature list / reintroduced into the frame), a standalone-AUC gate (any single feature with AUC > 0.9 fails the run), and a temporal-consistency sentinel that self-arms if a date column ever appears — on today's cleaned data it logs an explicit SKIP with its reason, rather than pretending to guard |
| **3. Point-in-time-safe features** | All features constructed from data available at application date only |
| **4. Baseline-first modeling** | Logistic-regression baseline before LightGBM; no premature complexity |
| **5. Calibration + cost-based thresholds** | Isotonic calibration on a disjoint calibration slice; operating threshold chosen by expected profit, not 0.5 |
| **6. Explainability + fairness** | SHAP reason codes in `src/explain.py` — rank-ordered risk-increasing factors on the model's raw **log-odds** axis, never converted to probability contributions (that quantity is undefined here — see "What this isn't"); a hand-rolled three-layer fairness audit — bootstrap CIs on the Equal-Opportunity ratio, a threshold sweep, and an ablation that retrains without `addr_state`. The audit caught `addr_state` acting as a digital-redlining shortcut, so the production model drops it: Mississippi's EO ratio recovers 0.74 → 0.99 for an AUC cost of just 0.0036 |
| **7. Drift monitoring** | A runnable yearly drift check (`pipelines/drift_check.py`): hand-rolled PSI + KS on `dti_n` per issue year against the training-years distribution, a separate 999-sentinel rate, a (100, 1000] tripwire share, and a per-year calibration gap scored with the shipped model. Validated against the dataset's own 2016+ DTI regime shift — the tripwire fires across all of 2016–2018, while the calibration-gap alarm fires on 2016–2017 (2018's gap flips slightly positive, +0.0154, staying under the alarm line); both are quiet on the 2015 baseline. A demonstrated capability on this dataset, not a live production monitoring service |

---

## Why most loan-risk projects fail

1. **Target leakage** — using fields like `int_rate`, `grade`, or `loan_status` that are
   set *after* the credit decision, not before.
2. **Random splits** — shuffling time-series data inflates AUC by letting the model peek
   at future patterns; always split by origination date.
3. **Inflated AUC as a red flag** — a single feature AUC > 0.90 almost always means
   leakage, not signal. Investigate before celebrating.

---

## What the paranoia actually caught

The discipline above isn't decoration — the audits fire on real data. Two cases where being
suspicious paid off:

**`addr_state` was a digital-redlining shortcut, so production dropped it.** The three-layer
fairness audit ends in an ablation that retrains the model *with* and *without* the state feature.
With `addr_state` in, Mississippi's Equal-Opportunity ratio for good borrowers sat at **~0.74** —
well under the 0.80 benchmark. Retrain without it and that ratio recovers to **~0.99**, at a
test-AUC cost of just **0.0036** (0.6690 → 0.6654). The state label wasn't encoding Mississippi's
economics; it was a geographic proxy the model leaned on as a shortcut. The shipped model omits it.
(This is a redlining-*risk* analysis through geography, not a legal disparate-impact finding — see
[What this isn't](#what-this-isnt).)

**495 "impossible" DTI rows weren't dirty data.** The pandera schema gate flagged 495 rows with a
debt-to-income ratio between 100.04 and 991.57 — outside the valid 0–100 band and not the `999`
missing-value sentinel. The lazy fix is to delete them. Instead the anomaly got investigated, and
the evidence pointed the other way: those rows default at **27%** vs. 20% overall (so they aren't a
decimal-shift artifact — the "corrected" values would be the *safest* borrowers, not the riskiest);
they show up as a temporal cliff (0 such rows before 2015, 488 in 2016–2018); and they profile as
real high-income, high-loan-amount borrowers. Verdict: a genuine subpopulation surfaced by a
**2016+ change in how DTI was reported**, not garbage — so the contract was widened deliberately
(100 → 1000), not silently, and the episode flagged a live distribution shift the training years
never see. Full write-up in [`docs/data-decisions.md`](docs/data-decisions.md).

---

## What this isn't

Listing your own limits is part of the same discipline: a model you can trust is one whose author
already knows where it breaks. This is a **case study in modeling judgment**, not a production
credit system.

- **Originated-loan model, not full underwriting.** The data contains only loans LendingClub
  approved; rejected applicants have no observed outcome. Deploying this as a granting model would
  require reject inference.
- **Feature-limited ceiling.** AUC ~0.67 reflects the absence of credit-bureau data (utilization,
  delinquency history, inquiries) — a data limit, not a modeling failure.
- **Stylized economics.** The profit model uses fixed LGD (0.65) and margin (0.12) assumptions.
  Real P&L needs term structure, pricing, servicing costs, and capital charges.
- **Fairness audit uses geography as a proxy.** Without protected-class labels, it audits redlining
  risk through `addr_state` — not a legal disparate-impact determination.
- **SHAP contributions live on the raw log-odds axis, and cannot be moved off it.**
  `src/explain.py` ships rank-ordered reason codes, not percentage-point attributions.
  That is not a mapping left undone — the mapping does not exist. The shipped isotonic
  calibrator is a 52-level step function whose slope is exactly zero across 99.31% of the
  reject region, so "this feature added N points of default probability" has no value to
  compute ([`docs/explainability.md`](docs/explainability.md), §4–§5). What survives both
  the sigmoid and the isotonic step is the **sign and the rank** of each contribution, and
  that is all this module reports. Whether a notice built on rank order alone satisfies
  ECOA / Regulation B is a question for counsel — we are not lawyers.

---

## Tech stack

| Tool | Role in this project |
|------|----------------------|
| **Python 3.11 + uv** | Runtime and reproducible dependency / virtualenv management |
| **LightGBM** | The production model — gradient-boosted trees on application-time features |
| **scikit-learn** | Logistic-regression baseline, preprocessing, and the isotonic calibrator |
| **SciPy** | Two-sample KS statistic in the drift monitor — declared as a direct dependency the moment code imported it directly. Evidently was deliberately not adopted: its web-server + telemetry dependency footprint isn't worth four scalar distribution signals, and hand-rolled PSI matches the repo's hand-rolled audits |
| **Pandera** | Schema gate that *fails* the pipeline on contract violations (it caught the 495 DTI rows) |
| **MLflow** (SQLite backend) | Experiment tracking; the pipeline logs every stage's metrics into one run |
| **Metaflow** | Orchestrates the end-to-end flow (load → … → fairness) as a linear `FlowSpec` |
| **SHAP** | `TreeExplainer` on the shipped booster, wrapped by `src/explain.py`: rank-ordered adverse-action reason codes whose contributions are the raw **log-odds margin**, declared as such in a `scale` field and in every key name. No probability-scale attribution is produced — see [`docs/explainability.md`](docs/explainability.md) §5. Called by `src/` and its tests; not yet wired into the Metaflow pipeline |
| **pytest** | 209 tests across the modeling layer |

---

## Setup

```bash
uv sync                                              # install dependencies
uv run pytest                                        # run the test suite (209 passing)
uv run python pipelines/training_flow.py run         # end-to-end training pipeline
uv run python pipelines/drift_check.py               # yearly input-drift check on dti_n
mlflow ui --backend-store-uri sqlite:///mlflow.db    # browse experiment tracking
uv run jupyter lab                                   # open the analysis notebook
```

---

## Data

See [`data/README.md`](data/README.md) for download instructions.

Real data files are **not committed** to this repo. The dataset is the
**Lending Club loan dataset for granting models**
([Zenodo](https://zenodo.org/records/11295916)) — peer-to-peer loans from
**2007–2018** (published on Zenodo in 2024), released under **CC BY 4.0**,
attribution required if you reproduce results.

---

## Project structure

```
data/          Download instructions + gitignored real files
docs/          Detailed write-ups for each stage
notebooks/     Exploratory analysis notebook (+ HTML export)
pipelines/     Metaflow end-to-end training pipeline + the yearly dti_n drift check
src/           Modeling layer: data loading, validation, features, leakage checks,
               training, calibration, evaluation, fairness, explanation
models/        Trained model + calibrator artifacts (gitignored)
figures/       Generated plots (gitignored)
tests/         pytest suite (209 passing)
```