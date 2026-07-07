# trust-issues

**A loan risk model that did the boring parts right.**

Most loan-risk projects make the same fatal mistakes — wrong target definition, data leakage,
random splits, and trusting inflated AUCs. This project has trust issues with all of that.
It defines the target carefully, audits for leakage, splits by time, calibrates its
probabilities, and checks fairness before trusting any score.

---

## What this project does right

| Stage | What |
|-------|------|
| **1. Problem framing** | Binary default target defined at 12-month horizon; cost matrix specified upfront |
| **2. Leakage-aware data prep** | Forbidden post-decision fields blocked; temporal consistency enforced |
| **3. Point-in-time-safe features** | All features constructed from data available at application date only |
| **4. Baseline-first modeling** | Logistic regression baseline before LightGBM; no premature complexity |
| **5. Calibration + cost-based thresholds** | Isotonic calibration; threshold chosen by expected-value, not 0.5 |
| **6. Explainability + fairness** | SHAP values; fairlearn demographic parity and equalized odds checks |

---

## Setup

```bash
uv sync
uv run jupyter lab
```

---

## Data

See [`data/README.md`](data/README.md) for download instructions.

Real data files are **not committed** to this repo. The dataset is the
LendingClub 2024 granting-model dataset released under **CC BY 4.0** —
attribution required if you reproduce results.

---

## Why most loan-risk projects fail

1. **Target leakage** — using fields like `int_rate`, `grade`, or `loan_status` that are
   set *after* the credit decision, not before.
2. **Random splits** — shuffling time-series data inflates AUC by letting the model peek
   at future patterns; always split by origination date.
3. **Inflated AUC as a red flag** — a single feature AUC > 0.90 almost always means
   leakage, not signal. Investigate before celebrating.

---

## Project structure

```
data/          Download instructions + gitignored real files
docs/          Detailed write-ups for each stage
notebooks/     Exploratory analysis notebook
src/           Python modules (leakage checks, features, calibration, evaluation)
figures/       Saved plots (PNGs committed; .tmp ignored)
tests/         pytest test suite
```
