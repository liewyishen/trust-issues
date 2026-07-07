# Data Decisions

Record of data-quality investigations that changed a contract in
`src/data_validation.py` (or would otherwise be non-obvious from reading the
code alone). Append new entries below; don't overwrite prior ones.

---

## The 495 rows with dti_n > 100: real extreme customers, not dirty data

**Date:** investigation run against the full real CSV, the first time
`validate_loan_data()` was wired into `load_raw()` (see `src/data_loader.py`).

**Finding:** the original schema contract `(0 ≤ dti_n ≤ 100) OR (dti_n == 999)`
rejected 495 real rows with `dti_n` strictly between 100.04 and 991.57 --
neither in the real band nor the known missing-value sentinel.

**Investigation — four pieces of evidence:**
1. **Default-rate reversal:** the 495 rows have a 27.07% default rate vs.
   19.98% overall. If these were a decimal-shift artifact (i.e. the true value
   is `dti_n / 100`), the corrected values would be the *lowest* leverage in
   the dataset (median ~1.5% DTI), predicting a *below*-average default rate --
   the opposite of what's observed. This rules out the decimal-shift
   hypothesis.
2. **Temporal cliff:** 0 rows in Train (2007-2014), 7 in 2015, 488 in
   2016-2018. The near-total absence before 2016 and concentration after
   points to a change in LendingClub's DTI computation/reporting methodology
   around 2015-2016, not to random data corruption.
3. **Not a sentinel / encoding residue:** 483 of 495 values are distinct,
   continuously distributed from 100.04 to 991.57 at the same two-decimal
   precision as ordinary DTI. Sentinels cluster on a handful of reused, exact
   values (like -1 or 9999) -- this population doesn't.
4. **Feature profile:** median revenue $103k vs. $65k overall, median
   loan_amnt $19.2k vs. $12k overall, FICO not depressed. Consistent with a
   real subpopulation of high-income, high-loan-amount borrowers whose
   reported DTI genuinely exceeds 100%, not with a noisy/garbage profile.

**Verdict:** real extreme customers, most likely reflecting a genuine 2016+
change in how DTI was computed or reported upstream -- not sentinel residue,
not a decimal-point error.

**Disposition:** `DTI_MAX_REAL` widened from 100 to 1000 in
`src/data_validation.py` (covers the observed real ceiling of 991.57 with a
small margin). `DTI_SENTINEL = 999` is kept as a separate, explicit OR branch
in the schema check even though it's now numerically redundant (999 ≤ 1000)
-- it remains a distinct missing-value sentinel semantically, not a real DTI
value, and merging it away would erase that meaning from the code. Values with
no evidentiary basis beyond 1000 (e.g. a stray 9999) continue to be rejected.

**TODO:**
- `pipelines/drift_check.py` (not yet built) must monitor `dti_n`'s
  distribution by `issue_year`, not just its marginal range -- the 2016+
  regime is invisible to a check that only looks at the overall histogram,
  since Train never sees it and Test/2018-holdout do. This is a live
  train/serve distribution-shift signal, not just a one-time data-hygiene
  fix.

---

## MLflow tracking backend: file store → SQLite

**Finding:** `src/train.py` originally pinned MLflow's tracking URI to a
plain filesystem store (`file://.../mlruns`) and had to set
`MLFLOW_ALLOW_FILE_STORE=true` to opt back into it, because the installed
`mlflow>=3.14` puts that backend into "maintenance mode" and refuses to
create a new file-based run store otherwise. That flag was a workaround, not
the recommended path, and it meant `mlflow ui` still failed unless the same
env var was set by hand on the command line.

**Disposition:** migrated the tracking URI to a local SQLite database
(`sqlite:///{PROJECT_ROOT}/mlflow.db`, absolute path so it resolves to the
same file regardless of invocation directory). `MLFLOW_ALLOW_FILE_STORE` is
no longer set anywhere in this codebase. `mlflow ui` now starts with:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

no extra flags or environment variables required.

**Note on `./mlruns`:** it is not fully obsolete. MLflow's artifact store
(where `mlflow.log_artifact()` actually copies the joblib model file) is a
separate concept from the tracking backend and still defaults to a local
`./mlruns` directory regardless of whether tracking points at a file store
or a database -- only run *metadata* (params/metrics/tags) moved into
`mlflow.db`. Both `mlflow.db` and `mlruns/` are gitignored.

**TODO:** none currently; revisit if/when `pipelines/training_flow.py`
needs a shared (non-local) MLflow backend for multi-machine runs.

---

## MLflow: calibrate.py / evaluate.py / fairness.py metrics — now unified in the pipeline

**Original finding (pre-pipeline):** calibrate.py's Brier/mean_pred and the
key metrics from evaluate/fairness were only printed to the terminal, not
logged to MLflow. Only train.py logged runs, so a model's calibration and
evaluation story lived in scrollback, not in a queryable archive.

**Disposition (resolved in `pipelines/training_flow.py`):** the Metaflow
pipeline now unifies all four stages' metrics into a **single**
`lgbm_production` MLflow run (experiment `lc_default_risk`). The train step
creates the run; the calibrate/evaluate/fairness steps reopen it by `run_id`
and append their metrics, producing one complete experiment archive per model
execution (25 metrics total: train/val/test AUC + PR-AUC, calibration Brier
raw/cal, evaluation profit/approval/bad-rate, fairness AUC-with/no-state and
cost, plus `fairness_audit_threshold` — the operating point Layer 1 was
audited at, added by the audit-hardening pass recorded below). Zero `src/`
changes — the logging lives entirely in the pipeline layer,
which reads the metrics dicts the `src` functions already return.

**TODO:** none currently.

---

## Executing the fairness conclusion: remove addr_state from the production model

**Audit evidence summary** (see the fairness-related records above and the
`src/fairness.py` module docstring for detail): the three-layer audit confirms
`addr_state` acts as a digital-redlining shortcut, not as a proxy for genuine
economic differences. Layer 3 ablation (threshold 0.22): Mississippi's
good-applicant Equal Opportunity ratio is ~0.734–0.745 with state included and
~0.988–0.990 with state removed; the test-AUC cost of dropping state is only
-0.0035 to -0.0036 (0.6689/0.6690 → 0.6654). Conclusion: removing `addr_state`
from the production model yields a fairness benefit that far outweighs its
predictive contribution.

This is framed as **geographic-proxy / digital-redlining risk, not a legal
discrimination finding** — `addr_state` is not an ECOA-protected class; the
audit surfaces a risk and its cost, it does not issue a legal determination.

**Implementation: a config switch, not a hard deletion.** `src/features.py`
adds a module-level switch `INCLUDE_ADDR_STATE = False` (off by default,
executing the audit conclusion); `CATEGORICAL` is built dynamically via
`build_categorical(INCLUDE_ADDR_STATE)`, while `addr_state`'s column
definition, `emp_order`, and the rest of the feature-engineering logic are
kept intact. A switch was chosen over deleting `addr_state` from the code so
that reproducibility is preserved: `build_categorical(True)` can still rebuild
the "with-state" feature set on demand, letting `fairness.py`'s Layer 3
ablation re-compare and re-verify at any time, rather than leaving only a
hard-coded historical conclusion in a document.

`src/fairness.py`'s Layer 3 (`audit_layer3_ablation`) was therefore refactored
to be fully self-contained: it no longer reads `features.py`'s
(switch-dependent) `FEATURES`/`CATEGORICAL`, nor does it rely on any loaded
production model as the "with-state" comparison baseline. Instead it uses its
own `FEATURES_WITH_STATE`/`FEATURES_NOSTATE` constants and trains both variants
from scratch every time — regardless of which one is currently deployed to
production.

**New numbers after retraining** (real data, `INCLUDE_ADDR_STATE=False`):

```
calibrate_model():
  Raw (uncalibrated):     Brier=0.1717  mean_pred=0.1705  AUC=0.6660
  Calibrated (isotonic):  Brier=0.1692  mean_pred=0.1915  AUC=0.6654

run_evaluation():
  Threshold chosen on VAL: 0.25
  Test profit @ 0.25:        -$288,367,478
  Naive 0.50 profit:         -$473,187,818
  Improvement over naive:    +$184,820,340
  Approval rate on test:      78.0%
  Bad rate among approved:    18.9%
  Bad rate among rejected:    38.5%

run_fairness_audit() Layer 1 (threshold=0.26): all states verdict=clear,
no state's CI falls entirely below 0.80 (the earlier with-state version had
MS revealed by Layer 2 as persistently below 0.80 under tighter thresholds;
after removing state, MS's EO ratio at threshold 0.26 has risen to ~0.994).
```

**Comparison with the with-state version** (pre-removal historical numbers,
for reference): test AUC 0.6689, threshold 0.26, improvement ≈ +$190.6M,
approval rate 80.3%, approved bad rate 19.2%, rejected bad rate 39.6%. The
slight AUC drop (~-0.0035), the smaller improvement (~$190.6M → ~$184.8M), and
the lower approval rate (80.3% → 78.0%) are all expected and acceptable costs
— this is the price of eliminating digital-redlining risk, not the model
getting worse.

**TODO:** none. If the decision to restore `addr_state` is ever reconsidered,
re-run `fairness.run_fairness_audit()` rather than assuming the historical
numbers still hold.

---

## Audit hardening: three promise/implementation gaps closed

**Date:** verified end-to-end against a full pipeline smoke run on the real
CSV (Metaflow run `1783436074070472`, unified MLflow run `a8f8e800…`,
experiment `lc_default_risk`). All numbers below come from that run's output,
not from memory or earlier documents.

**Finding:** a deep code audit found three places where the README's promises
and the pipeline's implementation had drifted apart. For a project whose
selling point is trust, closing promise/implementation gaps outranks adding
features.

1. **Leakage sentinels only half-wired.** README claimed "leakage *fails* the
   pipeline", but `pipelines/training_flow.py`'s start step called only two of
   `src/leakage_check.py`'s four sentinels (`check_forbidden_features`,
   `prove_forbidden_absent`). `flag_suspicious_auc` (single-feature AUC > 0.90
   red flag) and `check_temporal_consistency` (feature timestamp post-dating
   the decision date) lived only in tests and notebook narrative — decoration,
   by the leakage module's own standard.
2. **Temporal check silently absent.** The cleaned dataset has no feature
   timestamp column, so the temporal check could not run at all — and nothing
   recorded that fact anywhere.
3. **Fairness audited at a stale operating point.** `evaluate.py` selects the
   operating threshold on Val (0.25 this run), but `fairness.py`'s Layer 1
   audited at a hard-coded 0.26, and a stale comment still claimed
   "evaluate.py does not exist as of this turn".

**Disposition:**

1. Sentinel 4 is now a fail-closed pipeline gate. `src/leakage_check.py` gains
   `single_feature_aucs()` (numeric features ranked raw; categoricals
   target-encoded; orientation normalized to `max(auc, 1 - auc)` so
   negatively-oriented proxies cannot hide below 0.5; NaN rows dropped
   pairwise; degenerate features skipped) and `check_single_feature_auc()`
   (compute + flag + raise, all violations listed at once). Wired into the
   start step BEFORE any training compute is spent — the check is model-free.
   This run: "OK: no single feature exceeds standalone AUC 0.9 (8 features
   checked)."
2. Sentinel 3 now skips explicitly, never silently. `feature_date_columns()`
   (pipeline layer, pure function) scans for datetime-typed or `*_d`-named
   columns other than `issue_d`; if any exist, each is checked against
   `issue_d` and violations fail the flow; if none exist (today's cleaned
   data), the skip is printed with its reason. The sentinel self-arms the
   moment a future join reintroduces a date column.
3. The pipeline now passes evaluate's `best_threshold` into
   `run_fairness_audit(audit_threshold=...)`, so Layer 1 audits at the model's
   real operating point (0.25 this run; MS EO ratio 0.9945, all states clear),
   and logs `fairness_audit_threshold` to MLflow next to the ratios it
   qualifies. Layer 3's 0.22 is deliberately NOT aligned — the threshold sweep
   found the disparity most visible there, and the constant's comment now says
   so. The stale docstring is gone.

All three fixes rewire the connection between promise and implementation —
no model logic, feature engineering, split, or threshold-selection code was
touched. Test suite: 75 → 85, every new guard tested on both sides (passes
clean input, fires on dirty input).

**TODO:** none here. The calibrated-mean-vs-actual gap this run reprints
(mean_pred 0.1915 vs. actual test default rate 0.2323) is deliberately NOT
"fixed" by this pass — it is a real 2016+ distribution-shift signal, not a
bug, and belongs to the planned `pipelines/drift_check.py` (see the dti_n
entry's still-live TODO above).