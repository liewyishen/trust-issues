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
- ~~`pipelines/drift_check.py` (not yet built) must monitor `dti_n`'s
  distribution by `issue_year`, not just its marginal range~~ **Done.**
  `pipelines/drift_check.py` now exists and monitors `dti_n` per
  `issue_year`, not just its marginal range: per-year PSI/KS against the
  training-years distribution, a separate 999-sentinel-rate signal, and a
  dedicated (100, 1000] tripwire share -- plus the per-year calibration
  gap. The 2016+ regime that is invisible to an overall histogram is
  exactly what its first real run caught; see the "Drift monitoring" entry
  below for the design and numbers.

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
bug, and ~~belongs to the planned `pipelines/drift_check.py` (see the dti_n
entry's still-live TODO above)~~ **is now monitored**:
`pipelines/drift_check.py` reports it per year as `calib_gap_<year>` (first
real run: 2016 = −0.0372 and 2017 = −0.0468, both alarmed; 2015 baseline =
+0.0003). See the "Drift monitoring" entry below.

---

## Drift monitoring: the 2016+ dti_n regime shift gets a watcher

**Date:** verified against the first real-data drift run (MLflow run
`ae7fcc58…`, run name `drift_check`, experiment `lc_default_risk`). All
numbers below come from that run's logged metrics, not from memory or
earlier documents.

**Finding:** two earlier entries left the same live signal unmonitored. The
495-row investigation (above) established a 2016+ dti_n reporting regime the
training years never see -- 0 rows in Train (2007-2014), 7 in 2015, 488 in
2016-2018 -- and the audit-hardening entry deliberately declined to "fix"
the calibrated-mean-vs-actual gap (0.1915 vs. 0.2323) because it is that
shift's trace, not a bug. Both entries pointed at a then-unbuilt
`pipelines/drift_check.py`.

**Disposition:** `pipelines/drift_check.py` now exists: a plain module with
a callable entry (`uv run python pipelines/drift_check.py`). Three scoping
decisions made before implementation:

- **No Metaflow FlowSpec.** The check is one linear step; a FlowSpec would
  pickle the full DataFrame across a subprocess boundary for zero
  orchestration benefit.
- **Own MLflow run** (`drift_check`, same `lc_default_risk` experiment).
  Monitoring and training have different lifecycles, so drift metrics never
  append onto an `lgbm_production` run.
- **Hand-rolled PSI + scipy KS, not Evidently.** Four scalar signals do not
  justify Evidently's base dependency footprint (litestar/uvicorn/watchdog/
  nltk plus usage telemetry) in a batch job, and hand-rolled matches this
  repo's hand-rolled fairness audit. `scipy` is now declared as a direct
  dependency -- it was already in the tree transitively; declaring what the
  code imports directly is the point.

Five per-issue_year signals, reference = training years (2007-2014),
current = each of 2015-2018 as FULL years (not the val/calib subsamples):

1. `psi_dti_n_<year>` -- PSI over quantile bins fixed ONCE from the
   reference. Outer bin edges are pinned to the schema contract's
   [0, 1000], not the reference min/max: the training years top out far
   below 100, so an edge at their max would silently drop the 2016+ tail
   out of every bin. Empty bins are epsilon-clipped (1e-4, no
   renormalisation) so a fully emptied bin screams finitely instead of
   overflowing.
2. `ks_dti_n_<year>` -- two-sample KS statistic, never the p-value: at
   n ~ 4.5e5 every visible shift is "significant", so only the effect size
   informs.
3. `sentinel_rate_<year>` -- share of `dti_n == 999`. The sentinel is
   excluded from PSI/KS and from the tripwire (fed raw into PSI it fakes a
   mass spike at 999, and it falls inside (100, 1000] numerically); a
   change in missingness is its own signal, alarmed on the delta vs. the
   reference rate.
4. `tripwire_share_<year>` -- share of real `dti_n` in (100, 1000], the
   pre-widening ceiling repurposed as a tripwire. This signal exists
   because quantile-binned PSI is nearly blind to a ~0.1% tail: without it
   the monitor would sleep through the very shift that justified building
   it.
5. `calib_gap_<year>` -- mean calibrated PD minus actual default rate,
   scored with the SHIPPED model + calibrator (never refit; the scoring
   path replicates `calibrate_model()`'s exactly). The calibrator was fit
   on the 2015 calib slice, so 2015 is the natural not-yet-drifted
   baseline.

**What the first real run shows** (run `ae7fcc58…`, 5 drift ALERTs):

| year | tripwire_share      | calib_gap           | psi_dti_n |
|------|---------------------|---------------------|-----------|
| 2015 | 0.000019 (quiet)    | +0.0003 (baseline)  | 0.0498    |
| 2016 | 0.000246 (ALERT)    | −0.0372 (ALERT)     | 0.0392    |
| 2017 | 0.001543 (ALERT)    | −0.0468 (ALERT)     | 0.0274    |
| 2018 | 0.002760 (ALERT)    | +0.0154 (quiet)     | 0.0502    |

The tripwire climbs monotonically through 2016-2018 while PSI stays below
0.06 everywhere (KS below 0.082, under its 0.10 threshold) -- the monitor's
own first run confirms the design premise: PSI alone would have slept
through the regime shift. The calibration gap goes negative exactly where
the regime lives (2016-2017: the model under-predicts risk) and is ~0 on
the 2015 baseline. 2018's positive gap (+0.0154, under the 0.02 threshold)
points the other way -- consistent with the suspected right-censoring that
already keeps 2018 out of Test as a separate holdout.

**Alarm semantics, deliberately different from the leakage sentinels:** a
leakage sentinel firing means the run's output is invalid, so it raises. A
drift alarm is the observation this monitor exists to record -- on this
dataset 2016+ is EXPECTED to fire -- so the default is report-don't-raise:
alarms print as `ALERT:` lines, are counted into `n_drift_alarms`, and are
returned to the caller; automation can opt into `fail_on_alarm=True`.

Test suite: 85 → 104. The should-fire test injects a (100, 1000] tail into
a synthetic current year and asserts the tripwire fires while PSI stays
quiet -- the PSI-blindness finding as an executable assertion, not prose.

---

## Cost objective is `regret` (break-even 0.270), not `pure_profit` (0.156): an explicit, switchable identity

**Date:** verified against the test suite (104 → 116) and a direct argmax
check on 150k well-calibrated synthetic rows. All break-even numbers below
come from `uv run pytest` / that check, not from memory.

**Finding:** a code review found that `src/evaluate.py`'s `total_profit`
optimizes an objective whose break-even is NOT the one the module's own
test comment claimed. `total_profit` charges an opportunity cost on every
GOOD applicant that was rejected (`- profit_good[rejected_good]`). Because
the total good margin G over the population is fixed, rejected_good_margin
= G − approved_good_margin, so the objective is algebraically

    2*approved_good_margin − approved_bad_LGD − G,

i.e. it DOUBLE-WEIGHTS approving goods relative to a pure originated-book
P&L. For a well-calibrated p the decision boundary is therefore
`p < 2m/(2m+LGD) = 0.24/0.89 = 0.270`, not the pure-profit
`m/(m+LGD) = 0.12/0.77 = 0.156` the old `test_evaluate.py` comment
asserted. The evidence points at 0.270: the real-data operating threshold
is 0.25 (shipped no-state model) / 0.26 (historical with-state), and a
150k-row argmax lands on 0.26-0.28 for this objective and 0.14-0.17 for
pure profit. The opportunity-cost term was always live; the doc just named
the wrong break-even.

**The two objectives -- business semantics:**

- **`pure_profit`** -- originated-book P&L only. A rejected loan never
  originates, so it carries no term: +margin on an approved good, −LGD on
  an approved bad, zero on anything rejected. This is the "only count the
  loans we actually booked" view. Break-even `m/(m+LGD) = 0.156`.
- **`regret`** -- `pure_profit` minus the foregone interest margin on each
  rejected GOOD applicant. This treats a false rejection as a real cost:
  in a market where the lender competes for creditworthy borrowers, a good
  applicant turned away is margin the book will not earn. It is the fuller
  decision-analytic (regret / opportunity-cost) framing. Break-even
  `2m/(2m+LGD) = 0.270`.

The two are not a cosmetic relabeling: on identical data they select
materially different operating points (~0.26 vs ~0.15), which cascade into
different approval rates, bad rates, and loss figures.

**Decision: keep `regret` as the default.** Two reasons. (1) Every number
this repo reports -- best_t ~0.25-0.26, approval ~78-80%, improvement
~$185M over naive 0.50 (see the "Execute the fairness conclusion" and
evaluation entries / README) -- was computed under `regret`. Switching the
default to `pure_profit` would silently move all of them at once, with no
change to the model. (2) On the merits, for an originated-loan lender the
opportunity cost of a wrong rejection is a genuine P&L line, not a fiction;
`regret` is the more complete framing and `pure_profit` the narrower one.
This is a decision about which business question the threshold answers, and
`regret` answers the one the whole project has been answering all along.
Naming it changes nothing about the shipped operating point -- it only
makes the identity auditable instead of buried in one summation.

**Implementation: a config switch, not a hard choice.** `src/evaluate.py`
adds a module-level `PROFIT_OBJECTIVE = "regret"`; `pure_profit()` and
`regret_profit()` both exist as named functions; `total_profit(...,
objective=...)` dispatches to the chosen one (default = the switch);
`select_threshold` / `evaluate_at_threshold` / `run_evaluation` all forward
an `objective` argument defaulting to it. Same reproducibility rationale as
`features.py`'s `INCLUDE_ADDR_STATE` switch (see the addr_state entry
above): the alternative objective stays runnable and directly comparable at
any time, rather than being deleted into a paragraph. The prior test
asserted only `best_t < 0.5` -- true for both objectives, so it could never
catch the conflation. It is replaced by tests that PIN each break-even
(`regret` ~0.270, `pure_profit` ~0.156) and assert the two pick different
thresholds, so any future silent flip fails CI. No model logic, feature
engineering, split, or the threshold-selection algorithm itself was
touched -- only the objective's identity was made explicit.

**Attached behavior change (W1/W2): inference now consumes the packaged
contract, it does not just carry it.** The same review found two places
where an artifact recorded a safety property that no code enforced.
`src/train.py`'s `train_and_save()` bundles `features` / `categorical` /
`params` into the model `.pkl` with a note that this makes the artifact
"self-describing", and `src/calibrate.py` stored `model_path` alongside the
calibrator to make its binding "explicit and inspectable". But a grep of
the whole repo showed no consumer ever read `artifact["features"]` /
`["categorical"]` / `["params"]` (only `category_maps` was used), and
`load_calibrator()` discarded the binding on load. Inference re-read
`features.py`'s live module globals instead -- so a `FEATURES` /
`INCLUDE_ADDR_STATE` change between train and score would have silently
encoded against a different feature set than the model was trained on,
exactly the train/serve skew the packaging comments warned about, with no
guard. Disposition: `src/train.py` gains `load_model_artifact()`, which
fail-closes when the packaged `features`/`categorical` no longer match
`features.py`'s live globals; calibrate / evaluate / fairness / drift all
load through it instead of a bare `joblib.load`. `load_calibrator(path,
model_artifact=...)` now raises on a stale calibrator (the model's
`trained_at` recorded at fit time no longer matches the model in use),
closing the "retrain the model, forget to recalibrate" failure. The
self-describing artifact is now self-ENFORCING -- both guards tested on
both sides (matching contract passes, mismatch fails closed). The three
now-unused `joblib` imports this left behind were removed, and the dead
`fairlearn` dependency (the audit is hand-rolled) was dropped from
`pyproject.toml`, applying the repo's own "declare only what you import"
rule to itself.

**TODO:** none. If `PROFIT_OBJECTIVE` is ever flipped to `pure_profit`,
re-run `run_evaluation()` and update EVERY threshold-dependent number in
this doc and the README (best_t, approval rate, bad rates, improvement) --
they are all objective-specific, and the historical `regret` figures do
not carry over.

---

## The addr_state AUC cost now has a single authoritative value: 0.6690 → 0.6654, cost -0.0036

**Date:** 2026-07-09. Read from a full real-data pipeline run -- Metaflow
run-id `1783580432440889`, unified MLflow run
`cca4c361615c460b999ce1a73bd46439`, experiment `lc_default_risk`. Every
number below comes from that run's logged MLflow metrics at full double
precision (`repr`, not the terminal's four-decimal print, not memory).

**Finding:** the Layer-3 ablation AUC cost of dropping `addr_state` was
written two ways across the repo -- the README said 0.0036 (with-state AUC
0.6690), while `src/fairness.py`'s two docstrings said 0.0035 (with-state
AUC 0.6689). Line 127 of this file honestly recorded BOTH roundings
(`0.6689/0.6690 -> -0.0035 to -0.0036`) because no single run had ever been
pinned as authoritative: the terminal only prints four decimals, which
cannot distinguish 0.6689 from 0.6690. The disagreement was pure
fourth-decimal rounding of one number, never a disagreement about the
model.

**Resolution -- the authoritative full-precision values from run
`cca4c361`:**

```
auc_with_state = 0.6690306566920463      (rounds to 0.6690)
auc_no_state   = 0.6654356816065161      (rounds to 0.6654)
auc_cost       = -0.0035949750855301943  (rounds to -0.0036; equals
                                          auc_no_state - auc_with_state
                                          to full precision)
```

The correct four-decimal figures are **with-state 0.6690, cost -0.0036**.
The with-state AUC is 0.66903 -- about 8e-5 above the 0.66895 boundary
between 0.6689 and 0.6690, an unambiguous round-up, NOT a 0.66895 midpoint
tie. This vindicates the README (already 0.6690 / 0.0036, left untouched)
and identifies `src/fairness.py`'s `0.6689` / `~-0.0035` as the stale side;
fairness.py was corrected to 0.6690 / -0.0036 to match. The notebook is a
with-state historical artifact and was deliberately not touched.

**Line 127 of this file is kept as-is, NOT edited.** Its double-valued
range truthfully records the repo's state BEFORE this run existed -- two
roundings coexisting with no authoritative source to choose between them.
Overwriting it in place would falsify that history and break this file's
append-only rule. This entry is the authoritative pointer from here on;
line 127 stands as the historical record of the ambiguity before it was
resolved.

**Reproducibility, worth recording in its own right:** every
threshold-dependent number this run produced is identical to the prior
real-data run already documented above -- threshold 0.25, test profit
-$288,367,478, improvement over naive $184,820,340, approval 78.0%, bad
rate approved 18.9% / rejected 38.5%, Brier 0.1717 -> 0.1692, mean_pred
0.1915 vs. actual 0.2323, Layer 1 @ 0.25 all states clear. MS's Layer-3
EO recovery is 0.744823 -> 0.987926, inside the 0.734-0.745 / 0.988-0.990
spread this doc recorded earlier. The pipeline reproduces exactly; the only
thing that ever "moved" between documents was which fourth-decimal rounding
of the with-state AUC a given file happened to quote.

**TODO:** none. If the model is ever retrained, re-read
`auc_with_state`/`auc_no_state` from that run's MLflow metrics (not the
terminal, not this doc) and re-standardize the README and `src/fairness.py`
to the new authoritative values.

---

## Phase 1 credit-bureau foundation: CreditReport and MockBureau (data contract only, not wired to `/score`)

**Date:** 2026-07-11.

**Context:** Phase 1's goal is to move credit-bureau-shaped fields (starting
with `fico_n`) from applicant self-report to a system-fetched bureau pull.
This entry records the foundation built before any wiring happened: a data
contract (`CreditReport`), a swappable interface (`CreditBureau`), and a
deterministic mock implementation (`MockBureau`), all added in a new
`serving/bureau.py`. `serving/schema.py`'s `ScoreRequest`, `serving/app.py`'s
`/score` handler, and `src/explain.py`'s `explain_applicants()` are
untouched -- none of them import from `serving/bureau.py`. Four decisions
were made building it.

**Decision 1: `CreditReport` carries only the credit fields the shipped
model actually consumes (`fico_n`, `dti_n`) -- no field is added for report
"completeness."**

**Rationale:** the shipped model has exactly 8 features
(`src/features.py:64-69`); of those, only `fico_n` and `dti_n` are
credit-bureau-shaped. Fields a real bureau report also carries --
`delinq_2yrs`, `open_acc`, `revol_util`, `total_acc`, and similar -- have no
consumer anywhere in this codebase. `revol_util` specifically was confirmed
via a repo-wide grep to have zero references in `src/`, `serving/`,
`pipelines/`, `docs/`, or the README before this work; the only two mentions
that exist now are in `serving/bureau.py:36` and `tests/test_bureau.py`,
both naming it as an example of a field deliberately not added.

**Trade-off:** `CreditReport` (`serving/bureau.py:83-146`) reads less like a
complete, realistic bureau report than it could. That is accepted rather
than fixed here -- the unconsumed fields stay out until the model has an
actual use for them, deferred rather than added as dead weight.

**Decision 2: `dti_n` is carried on `CreditReport` as a bureau-supplied,
pre-computed field -- this layer computes no DTI.**

**Rationale:** this repo's `dti_n` is read directly from the LendingClub
source CSV (`src/data_loader.py:80-119`'s `load_raw()` parses dates and
validates; it performs no DTI arithmetic) and is declared in `LOAN_SCHEMA`
as a `Column`, not a derived value (`src/data_validation.py:123-144`).
`src/features.py`'s `add_features()` passes `dti_n` through unmodified along
with the other `NUMERIC` fields (`src/features.py:64`); there is no
`monthly_debt` or other finer-grained raw field anywhere in this codebase a
DTI formula could be built from. A real credit bureau supplies `dti_n` the
same way -- pre-computed from the applicant's tradeline history, not derived
by the lender's own serving code. `CreditReport.dti_n`
(`serving/bureau.py:127-146`) therefore mirrors the exact band-or-sentinel
constraint `ScoreRequest._dti_in_band_or_sentinel` already enforces
(`serving/schema.py:126-143`), built from the same imported
`DTI_MAX_REAL`/`DTI_SENTINEL` constants, and passes the value through
unchanged.

**Trade-off:** writing a `dti = debt / income` formula here was explicitly
rejected, even though it would have made `CreditReport` self-sufficient --
doing so would introduce a computation the training data never went
through, the exact train/serve skew this project's core discipline (train
the way you serve) exists to prevent. `CreditReport` therefore depends on
whichever bureau implementation supplies it to hand over a pre-computed
`dti_n`; this is a constraint imposed by the current data reality, not a
design preference.

**Decision 3: `MockBureau`'s `fico_n` is drawn from a
`Normal(mean_fico, std_fico)` distribution seeded by `applicant_id`, with
`mean_fico` a configurable constructor argument (default 700.0, `std_fico`
default 50.0).**

**Rationale:** two requirements had to hold simultaneously. Determinism --
the same `applicant_id` must return a byte-identical report on every call,
forever, verified by `tests/test_bureau.py::test_same_applicant_produces_same_report`
and `::test_determinism_holds_across_separate_bureau_instances` -- is met by
seeding a `numpy.random.default_rng` from a SHA-256 digest of
`applicant_id` (`serving/bureau.py:191-221`), a pure function with no
instance state, the same reproducibility discipline as
`src/data_loader.py`'s `RANDOM_SEED = 42`. Controllability --
`MockBureau(mean_fico=650)` must shift the whole output distribution, not
just individual reports -- is why `mean_fico`/`std_fico` are constructor
arguments (`serving/bureau.py:191-193`) rather than module constants: this
is the knob a future drift-injection demo needs. An earlier hash-uniform
design (mapping the hash directly to a uniform float across
`[FICO_MIN, FICO_MAX]`) was considered and rejected: it had no adjustable
mean and could not simulate a population-level shift, and a uniform
distribution does not resemble a real population's FICO distribution
(bell-shaped, clustered around a center, not flat).

Measured evidence (this session, one-off verification script, run and
discarded, not committed to the repo): fetching 100 distinct `applicant_id`s
through `MockBureau(mean_fico=700.0)` produced a sample mean `fico_n` of
706.13; through `MockBureau(mean_fico=650.0)`, 656.13 -- a ~50-point
separation matching the ~50-point gap between the two `mean_fico` settings,
confirming the knob moves the distribution as intended. `fico_n` is clipped
into `[FICO_MIN, FICO_MAX]` after the normal draw (`serving/bureau.py:221`,
both constants imported from `src/data_validation.py:90`), since
`CreditReport.fico_n`'s own `Field(ge=FICO_MIN, le=FICO_MAX)` constraint
would otherwise reject a tail draw outright.

**Trade-off:** `mean_fico=700.0` / `std_fico=50.0` are demo assumptions
about a "normal" population, not measured from this repo's real FICO
distribution (`src/data_validation.py:88-90`'s comment puts the real
observed band at approximately 612-847) or from any dataset here --
documented as such in `MockBureau`'s docstring per this project's rule
against silently armchaired numbers. A few extra lines of seeded-RNG code
were accepted in exchange for a distribution that can actually demonstrate
drift.

**Decision 4: this round stayed strictly additive -- only
`serving/bureau.py` and `tests/test_bureau.py` were created; nothing was
wired into any existing module.**

**Rationale:** wiring (removing `fico_n` from `ScoreRequest`, inserting a
bureau call into `/score`) would touch `serving/schema.py` and
`serving/app.py` and cascade into `tests/test_serving.py`'s 35 tests, 67 of
which hardcode the current 7-field payload shape (confirmed by grep on
`tests/test_serving.py`) -- a change with real blast radius, of a different
character than laying a foundation. Splitting the two into separate steps
keeps each one reviewable and revertible on its own.

**Evidence:** after adding the two new files, the full suite collected 242
tests (217 pre-existing + 25 new) and ran 242 passed / 0 failed. The one
transient exception was
`tests/test_readme.py::test_readme_test_count_matches_the_live_collection`,
itself a self-checking invariant test (`tests/test_readme.py:75-93`) -- it
correctly turned red the moment the live count diverged from the four
`217`s still written in `README.md`, and was brought back to green by
updating exactly those four occurrences (the badge and three prose
mentions) to `242`, with no other number in `README.md` touched.

**TODO:** wiring `CreditReport`/`MockBureau` into `ScoreRequest` and
`/score` is a deliberately separate, not-yet-scheduled step (see Decision
4). When it happens, re-run the full suite and expect
`tests/test_serving.py` to need real edits, not just a re-collection.

---

## Phase 1 bureau wiring: `fico_n` is now sourced from `CreditBureau`, not the applicant

**Date:** 2026-07-11.

**Context:** the previous entry laid the foundation (`CreditReport`,
`CreditBureau`, `MockBureau`) without wiring it to anything. This entry
records that wiring: `fico_n` moves from applicant self-report to a
bureau-fetched value on the live `/score` path. `dti_n` stays
applicant-reported this round -- a confirmed decision, not an oversight:
`MockBureau.fetch()` always produces a real, in-band `dti_n` (never the 999
sentinel), so moving `dti_n` to the bureau this round would have made the
sentinel-handling contract untestable at the HTTP boundary without also
extending `MockBureau` to simulate sentinel-producing applicants -- out of
scope for a wiring step. Moving `fico_n` alone also matches the "starting
with `fico_n`" scope the prior entry already recorded.

**What changed:**

- `ScoreRequest` (`serving/schema.py`) drops `fico_n` and gains
  `applicant_id: str = Field(min_length=1)`. The six remaining
  applicant-reported fields are `revenue`, `dti_n`, `loan_amnt`,
  `emp_length`, `purpose`, `home_ownership_n`. Submitting `fico_n` on a
  request is now a 422 under the existing `extra="forbid"` gate, the same as
  submitting any other unrecognized field.
- `ScoreResponse` gains three bureau-provenance fields -- `bureau: str`,
  `fico_version: str`, `credit_report_pulled_at: datetime` -- filled in by
  `/score` from the `CreditReport` it fetched, not by `explain_applicants()`,
  which knows nothing about the bureau layer and never returns these keys.
  `tests/test_serving.py::test_response_mirrors_explain_applicants_key_for_key`
  now asserts key-equality after subtracting these three, so the mirror
  invariant still holds on everything `explain_applicants()` is actually
  responsible for.
- `serving/app.py` gains a `CreditBureau` dependency, injected exactly the
  way `ArtifactBundle` already is: `lifespan()` constructs `MockBureau()`
  once at startup, `get_bureau(request)` mirrors `get_bundle(request)`
  (503 only in the reachable-only-in-tests unloaded-state case,
  `serving/errors.py`'s new `BUREAU_UNAVAILABLE_DETAIL`), and
  `create_app(bundle=None, bureau=None)` accepts a pre-built bureau for
  tests. `/score` calls `bureau.fetch(applicant.applicant_id)` unguarded (see
  the TODO below), then `_to_raw_frame(request, report)` merges the
  six applicant fields with `report.fico_n` into the exact seven-field raw
  dict `explain_applicants()`/`add_features()`/`_x()` already expected --
  byte-identical in shape to what this function produced before the bureau
  existed, just assembled from two sources instead of one.
  `report.dti_n` and `report`'s provenance fields (`pulled_at`, `bureau`,
  `fico_version`, `inquiry_window_days`) are deliberately NOT merged into
  that frame: its contract is exactly the seven raw fields the model
  expects, nothing more.
- Nothing in `src/` or `pipelines/` changed. `explain_applicants()`,
  `_x()`, and `add_features()` are untouched -- the merge produces a frame
  identical in shape to what they already consumed, so there was nothing for
  them to adapt to.

**Test suite:** `tests/test_serving.py` went from 53 to 59 collected tests
(6 new: bureau called with the request's `applicant_id`; the three
provenance fields surface in the response and match the fetched
`CreditReport`; the same `applicant_id` scores byte-identically across two
independent `/score` calls; ten distinct `applicant_id`s don't collapse to
one margin; a blank `applicant_id` is a 422; a missing bureau client is a
503, mirroring the existing missing-bundle case). Nine existing tests were
rewritten rather than left to pass by fixture-shape coincidence: the
request/raw-frame shape split required a second fixture (`GOOD_RAW`, the
seven-field model-input shape, alongside `GOOD`, the six-field-plus-
`applicant_id` request shape) for the tests that call `_x()`/
`explain_applicants()` directly; the two strict-float contract tests were
retargeted from `fico_n` (no longer a request field) to `revenue`; the
out-of-band-numerics test dropped its `fico_n` sub-case and the extra-field
test gained one (submitting `fico_n` is now the relevant assertion); the
missing-bundle 503 test now also injects a bureau, so that 503 is
unambiguously attributable to the bundle alone. No test was deleted,
skipped, or had an assertion weakened to reach green -- every change updates
what the test checks to the new contract, verified by the same self-checking
list this entry describes and re-run in full afterward: **248 tests
collected, 247 passed** (the sole exception is `tests/test_readme.py`'s own
self-checking test-count invariant, which correctly detected that
`README.md`'s written counts no longer match the live collection -- README
sync is outside this round's authorized file list and was left for a
separate pass, the same as it was after the prior bureau-foundation entry).

**TODO (deferred, a decision, not an oversight):** `CreditBureau.fetch()`'s
docstring states it "raises on a failed pull" (`serving/bureau.py:159-160`),
but `/score` calls it unguarded -- no `try`/`except`, no corresponding HTTP
error class. This is not an omission: `MockBureau`, the only implementation
wired in today, performs no I/O and cannot fail, so there is no reachable
failure path to handle yet, and `serving/errors.py`'s taxonomy only
documents branches that can actually be reached (see its own module
docstring on why the existing 503 branches are "not dead code"). Building a
catch block around a call that cannot fail would be exactly the
speculative, "just in case" code this project's discipline argues against
elsewhere (e.g. `ScoreResponse`'s deliberately absent
`contribution_to_probability` field, `docs/explainability.md` Section 5).
When a real bureau client is wired in, this is the first thing to add:
`serving/errors.py` gains a fourth documented class (a 502, matching "an
upstream dependency failed" semantics -- distinct from the existing 503,
which means "this service never finished loading," not "a call to another
service failed"), and `/score` wraps `bureau.fetch()` accordingly.

---

## Phase 2: `src/train.py` split into `train.py` (MLflow) + `model_io.py` (everything else); serving image 2.64GB -> 937MB

**Date:** 2026-07-11.

**Context:** `serving/` (and `calibrate.py` / `evaluate.py` / `fairness.py` /
`explain.py` / `pipelines/drift_check.py`) never called `train_and_save()`, the
only function in `src/train.py` that actually touches MLflow -- but the module
still imported `mlflow` and called `mlflow.set_tracking_uri(...)` at import
time, so every one of those consumers pulled MLflow (and its transitive
dependencies: pyarrow, sqlalchemy, botocore, cryptography, ...) into the
serving image regardless.

**Disposition:** `load_model_artifact`, `_x`, `_xy`, `_to_lgb_frame`,
`_train_categories`, `_report_metrics`, `LGB_PARAMS`, `train_baseline`,
`train_lgb`, `run_spw_ablation`, `PROJECT_ROOT`, `DEFAULT_MODEL_DIR` moved
verbatim into a new `src/model_io.py`, which imports no `mlflow`. `src/train.py`
keeps only the tracking-URI setup and `train_and_save()`. Six callers and four
test files updated their import paths; `pipelines/training_flow.py`'s import of
`train_and_save` (the one name that didn't move) is untouched. Verified at
runtime, not just by grep: importing `serving.app` no longer puts `mlflow` or
`src.train` into `sys.modules`.

`pyproject.toml` gained a `training` dependency group (`mlflow`, `metaflow`,
`matplotlib`, `seaborn` -- verified against the code, not assumed, to have no
consumer serving's import graph reaches) with `[tool.uv] default-groups =
["dev", "training"]`, so a bare `uv sync` still installs everything locally;
the Docker build opts out explicitly (`--no-dev --no-group training`). The
Dockerfile's blanket `RUN chown -R appuser /app` -- measured to duplicate
~885MB of the preceding layer via overlay2 copy-up -- was replaced by
`COPY --chown=appuser:appuser`, with `useradd` moved earlier so the user
exists first. Measured, not estimated, via a fresh `docker build` (a
`git stash` round-trip reproduced the "before" number bit-for-bit against the
earlier recon measurement): serving image DISK USAGE **2.64GB -> 937MB**
(CONTENT SIZE 612MB -> 223MB). Local pytest was unaffected throughout --
the `.venv` was never re-synced -- and stayed at 248 passed, 0 failed.

**Note on this file's own W1/W2 entry above ("Attached behavior change (W1/W2):
inference now consumes the packaged contract..."):** that entry's line "Disposition:
`src/train.py` gains `load_model_artifact()`" is left exactly as written, per
this file's append-only rule -- it correctly described where the function was
added at the time. `load_model_artifact()` (and the rest of the functions
named above) now live in `src/model_io.py`, not `src/train.py`. This entry is
the pointer from here on; the W1/W2 entry stands as the historical record of
where the guard was first added, same discipline as the addr_state-AUC entry's
note on its own line 127.

**Documentation convention adopted in this round:** README.md, `docs/design.md`,
and `docs/explainability.md` had accumulated many `file.py:123` / `file.py:45-67`
citations. A line number goes stale on the next unrelated refactor near that
line (this round's own `src/train.py` split moved several); a function, class,
or constant name does not, so those three documents now cite symbols
(`` `calibrate_model()` ``, `` `LOAN_SCHEMA` ``, `` `TreeExplainer.shap_values` ``,
...) instead of line numbers wherever a stable name exists. This file
(`data-decisions.md`) is exempt -- append-only, so its existing citations
(including the now-stale ones the note above calls out) are left as written,
the same historical-record reasoning as everywhere else in this file.

**TODO:** none. If a future round wants `data-decisions.md`'s own historical
`file.py:line` citations converted to symbols too, that would need a new
append-only entry per citation, not an in-place edit -- not attempted here.

---

## The "move `dti_n` to the bureau" deferral has an unrecorded cost: `MockBureau.dti_n` is uniform over [0, 1000), and wiring it in silently feeds the model off-manifold DTI

**Date:** 2026-07-12.

**Context:** the deferral list (this file's "Phase 1 bureau wiring" entry, echoed
in `docs/design.md` §5 and `docs/PROJECT_STATUS.md`) carries the one-liner "Move
`dti_n` to the bureau -- Phase 1 moved only `fico_n`." Read cold, that scans as a
mechanical next step: change `_to_raw_frame()` to take `dti_n` from the report
instead of the request, done. It is not. This entry records the cost that
one-liner does not carry, so the next person to pick it up does not discover it
by shipping it.

**Finding:** `MockBureau.fetch()` (`serving/bureau.py`) does not draw `dti_n` from
anything resembling a real DTI distribution. It carves eight bytes out of the
applicant_id's SHA-256 digest, divides by `2**64` to get a uniform draw in [0, 1),
and multiplies by `DTI_MAX_REAL` -- so `dti_n` is uniform over [0, 1000).
Measured over 20,000 synthetic applicant_ids:

    min        0.01
    median   499.48
    mean     499.46
    max      999.95

    94.0% of draws exceed 60    (the top of the real LendingClub DTI band)
    89.9% of draws exceed 100   (the ORIGINAL DTI_MAX_REAL, before it was widened)
     0.0% of draws are the 999 sentinel

Real `dti_n` in this dataset sits roughly 0-60. `DTI_MAX_REAL` is 1000 only
because it was widened to admit a narrow anomalous tail -- see this file's "The
495 rows with dti_n > 100" entry for that investigation. The mock's median
applicant is therefore not a high-leverage borrower; it is a borrower who does
not exist.

(Note for anyone cross-checking: the 495 in that entry's title is a ROW COUNT,
not a DTI value, and is unrelated to the ~499 median measured here. The
near-collision is a coincidence, and it has already misled one reading of these
two facts.)

**Why this is harmless today, precisely:** `/score` never reads the mock's
`dti_n`. `_to_raw_frame()` (`serving/app.py`) builds the model-input frame from
`request.model_dump(exclude={"applicant_id"})` plus exactly one key taken from
the pull, `"fico_n": report.fico_n`. `report.dti_n` is fetched, validated, and
then never referenced -- `dti_n` reaches the model off the *request* object, from
the applicant, where it has always come from. The Phase 1 wiring entry already
records that `report.dti_n` is deliberately not merged. What was never recorded
is that this omission is not merely tidy scoping: it is the only thing standing
between the model and the numbers above.

**The cost:** wire `report.dti_n` into `_to_raw_frame()` as the deferral reads,
and the model immediately consumes DTI centered near 500. Nothing raises.

- **Pydantic passes.** `CreditReport.dti_n` and `ScoreRequest.dti_n` share the
  same band-or-sentinel rule, built from the same imported `DTI_MAX_REAL` /
  `DTI_SENTINEL`. A draw of 499.48 is inside [0, 1000]. It is a *valid* value; it
  is not a *plausible* one, and the schema was never asked to know the difference.
- **The additivity guard passes.** `_assert_additivity` (`src/explain.py`) checks
  that the SHAP contributions reconstruct the model's margin to
  `ADDITIVITY_ATOL`. That is a statement about the arithmetic, not about the
  input: the explanation of a nonsense score is still a correct explanation of
  that score. The math stays exact all the way down.
- **The drift monitor is not in the path.** `pipelines/drift_check.py` is a manual
  entry point, not an inline gate on `/score`, and it reports rather than raises.

The failure is silent by construction: every guard this repo owns is a guard on
*correctness*, and every one of them would be satisfied. The service would return
well-formed, additive, fully-explained decisions computed from an input
distribution the model has never seen. That -- a confident wrong answer with no
alarm attached -- is the exact failure mode the rest of this project exists to
make impossible, and the deferral list currently invites it in one line.

**What the deferral actually requires: two preconditions, not one.**

1. **A realistic `dti_n` distribution in `MockBureau`.** Uniform[0, 1000) cannot be
   wired in. Whatever replaces it will be an armchair number and must be disclosed
   as one, exactly as `MockBureau`'s own docstring already discloses that
   `mean_fico=700.0` / `std_fico=50.0` are demo assumptions not measured from any
   dataset in this repo.
2. **A path that can emit the 999 sentinel.** This is not a new requirement -- it is
   the *original* reason `dti_n` was deferred. The Phase 1 wiring entry states it:
   `MockBureau.fetch()` never produces the sentinel, so moving `dti_n` to the bureau
   would make the sentinel-handling contract untestable at the HTTP boundary without
   also extending `MockBureau` to simulate sentinel-producing applicants. The 0.0%
   sentinel rate measured above confirms that still holds.

These are the same deferral's two halves, and they were being tracked as one
sentence and zero sentences respectively. Both are the same kind of gap:
`MockBureau` can produce neither a realistic DTI nor a missing one.

**Disposition:** no code changed. `dti_n` stays applicant-reported; the deferral
stays deferred, and stays correct to have deferred. What changes is that it is no
longer a one-liner -- it is a deferral with a recorded cost and two named
preconditions, so shipping it without both is now a documented mistake rather
than an available one.

**TODO:** when `dti_n` does move to the bureau, both preconditions land in the same
round as the `_to_raw_frame()` change -- never before it. A `MockBureau` that
returns uniform[0, 1000) `dti_n` must not be reachable from `/score` in any
intermediate commit.

---

## Why "serving never reaches `matplotlib`" was false, and why grep could not catch it

**Date:** 2026-07-12.

**Context:** commit `3e612cb` corrected a false claim that had been written into
three places at once (`pyproject.toml`'s dependency comments, `docs/design.md` §4,
and `README.md`'s serving-layer section): that serving's import graph never reaches
any of the four training-only packages (`mlflow`, `metaflow`, `matplotlib`,
`seaborn`). Three of the four are genuinely unreached. `matplotlib` is reached. The
corrected prose now says so. This entry records the *method* -- why the claim was
wrong, and why the obvious check certifies it as right -- because the conclusion is
a fact about one package on one day, while the method is the part that stops the
next person from re-deriving the same false claim the same way.

**The trap:** the claim was not lazy. It survives the check almost anyone would run:

    $ grep -rn "import matplotlib" serving/
    $ grep -rni "matplotlib" serving/
    $

Both are empty. `serving/` genuinely never writes that import, and the string
"matplotlib" does not occur anywhere in the package. A grep-based audit therefore
"confirms" that serving is matplotlib-free -- and is wrong.

**The truth:** the import is transitive and exception-guarded. `serving/artifacts.py`
imports `lightgbm`; lightgbm's `compat` module (third-party, in site-packages) runs
this at import time, setting its `MATPLOTLIB_INSTALLED` flag:

    try:
        import matplotlib
        MATPLOTLIB_INSTALLED = True
    except ImportError:
        MATPLOTLIB_INSTALLED = False

So `import serving.app` really does execute `import matplotlib`. Nothing in this
repo asked for it and no string in this repo names it -- and because the import is
wrapped in `try`/`except`, its *absence* is equally invisible: the guard swallows
the `ImportError` and sets a flag. The import is real, silent when present, and
silent when missing.

**The only reliable check is runtime module inspection, not static grep:**

    $ python -c "import serving.app, sys; print({m: (m in sys.modules) for m in ['mlflow','metaflow','seaborn','matplotlib']})"
    {'mlflow': False, 'metaflow': False, 'seaborn': False, 'matplotlib': True}

That one line is what surfaced the error. It is also the check that must be re-run
to defend any future claim of this shape.

**The slimming decision itself was correct, and stands.** This entry is not "the
image was built wrong" -- it was not. Precisely because lightgbm's import is
guarded, `matplotlib`'s absence from the slim image is *tolerated*:
`MATPLOTLIB_INSTALLED` becomes `False`, lightgbm's plotting helpers are disabled,
and nothing on the scoring path touches them. Excluding `matplotlib` from the
runtime image via the `training` dependency group remains right, and the 2.64GB ->
937MB result (this file's Phase 2 entry) is unaffected. What was wrong was the
*stated reason*: the four packages were grouped as "never reached," when the honest
grouping is **three never reached, one reached and tolerated**. The decision was
right and the justification was untrue, and this repo's thesis is that the second
one still matters when the first one holds.

**The rule, generalized:** any claim of the form "serving's import graph does not
reach package X" must be verified by post-import `sys.modules` inspection, never by
grepping for `import X`. A transitive, exception-guarded import is invisible to grep
(no first-party source names it) *and* invisible to a smoke test (the guard swallows
the failure), yet it is real -- and in a slim-image argument it is exactly the case
that decides whether the argument is true. Grep proves what this repo's own code
says. Only the interpreter knows what it actually imports.

**TODO:** none; this is a verification-method record. The three-unreached /
one-tolerated split is stated in `pyproject.toml`, `docs/design.md` §4, and
`README.md` as of `3e612cb`.


---

## The bureau's contribution to `/score` moves from three loose provenance fields to a nested `credit_report`, and the fetched `fico_n` goes in with it -- but the fetched `dti_n` does not

**Date:** 2026-07-12.

**Context:** a browser frontend was being built against `/score`, and its
"credit report, fetched by the system" panel needed the FICO the bureau
returned. It could not have it. `ScoreResponse` carried the three fields that
said WHICH pull happened (`bureau`, `fico_version`, `credit_report_pulled_at`)
and none that said what was IN it: the fetched `fico_n` was merged into the
scored frame by `_to_raw_frame` (`serving/app.py`) and then never surfaced.
Verified against a live request before anything was changed -- applicant
`demo-001` scores on a bureau `fico_n` of 703.9270999057935, and that number
appeared nowhere in the 200 it got back. (`contributions_log_odds["fico_n"]` is
present, but that is a SHAP contribution in log-odds, not the value;
`reason_codes` carries a `value` only when the feature is among the top adverse
factors, which a strong FICO never is.) A client was told which report was
pulled and never what the decision was made on.

**The shape chosen, and the one rejected:** the obvious fix -- add `fico_n` as a
fourth top-level key -- was rejected on naming grounds, which turned out to be
structural grounds. `tests/test_serving.py`'s `BUREAU_PROVENANCE_KEYS` guarded
the response's mirror invariant by naming the keys `/score` adds that
`explain_applicants()` does not. Adding `fico_n` to that set would have made its
name describe something three-quarters true: `fico_n` is DATA -- the value the
booster consumed -- not provenance, and no honest name exists for a set holding
both. The constant would have had to stretch. Instead the response nests:

    ScoreResponse.credit_report: ScoredCreditReport   # serving/schema.py
        fico_n        # the value the model actually scored on
        bureau        # }
        fico_version  # }  which pull supplied it
        pulled_at     # }

and `BUREAU_PROVENANCE_KEYS` was REMOVED rather than renamed, replaced by two
constants that each describe exactly what they hold:
`BUREAU_SOURCED_TOP_LEVEL_KEY = "credit_report"` (the one key `/score` adds) and
`CREDIT_REPORT_KEYS` (the nested block's four -- explicitly not called
"provenance", for the reason above).

**The `dti_n` exclusion -- the decision this entry exists for.** `CreditReport`
HAS a `dti_n`. `ScoredCreditReport` does not, and must not. `_to_raw_frame` takes
`dti_n` from the REQUEST and leaves `report.dti_n` unread (the Phase 1
bureau-wiring entry above records that). So a "credit report" block containing
the pulled `dti_n` would show a client a bureau DTI sitting beside a decision
computed from the applicant's self-reported one. Measured on the same live
request:

    request dti_n  (what was SCORED)              18.0
    bureau  dti_n  (fetched, deliberately unused) 514.32

That panel would have been a decision described one way and made another -- the
exact failure this repo exists to prevent, and the exact cost the entry above
("The 'move dti_n to the bureau' deferral has an unrecorded cost") measures. The
omission is guarded, not merely intended:
`test_scored_credit_report_does_not_carry_dti_n` asserts `dti_n` is absent from
`ScoredCreditReport` AND present on `CreditReport`, so the absence is provably a
choice; if someone drops `dti_n` from `CreditReport` entirely, the test goes red
and forces them to read why. `applicant_id` and `inquiry_window_days` are
excluded too, for smaller reasons recorded in `ScoredCreditReport`'s docstring.

**`pulled_at` is a wire rename, not a move.** The old top-level name was
`credit_report_pulled_at`; the prefix existed only to disambiguate it from
`model_trained_at` / `calibrator_trained_at` at the top level. Nested, it would
stutter (`credit_report.credit_report_pulled_at`) and -- the real reason -- would
stop matching `CreditReport`'s own field name, which is what
`test_scored_credit_report_invents_no_field_names` asserts the block's keys
against (`CREDIT_REPORT_KEYS <= set(CreditReport.model_fields)`). That test is
what makes "a client-facing subset of `CreditReport`" a fact rather than a
docstring claim, and the old spelling could not satisfy it. A client reading
`credit_report_pulled_at` will not find it.

**The mirror invariant got stronger on the way through.** It was subtractive --
`set(ScoreResponse.model_fields) - BUREAU_PROVENANCE_KEYS == set(explain keys)` --
which would have passed just as happily if a fourth bureau key were added to both
the model and the constant excusing it. It is now additive:
`set(ScoreResponse.model_fields) == set(explain keys) | {"credit_report"}`. The
count of keys `/score` adds is now pinned at exactly one.

**Blast radius:** breaking on the wire (`bureau`, `fico_version`,
`credit_report_pulled_at` all move and the last is renamed) and free in practice
-- `frontend/` did not exist yet, so there was no client to break. This was the
cheapest moment the change would ever have; it was taken then rather than after a
consumer existed.

**Disposition:** `src/` untouched -- no scoring logic changed, and no decision
changes value. This is a response-SHAPE change. `README.md`, `docs/design.md`
Section 5 and `docs/PROJECT_STATUS.md` were re-synced in the same commit as the
code, so no window exists in which they describe the old shape.

**TODO:** none. The Phase 1 bureau-wiring entry above still describes
`ScoreResponse` "gaining three bureau-provenance fields"; that sentence was
accurate when written and is left exactly as written, the same append-only
discipline this file applies everywhere else. This entry supersedes it.

---

## The fairness audit ships as a frozen artifact bound to the model, not as a live route -- and refuses to serve its numbers when the binding breaks

**Date:** 2026-07-13.

**Context:** the frontend needed to show the `addr_state` finding. That finding is
this repo's loudest fairness claim -- Mississippi's Equal-Opportunity ratio
recovering from ~0.74 to ~0.99 once the state label is dropped -- and it turned
out to be **persisted nowhere**. MLflow holds exactly five fairness numbers, which
is the complete list across every run ever logged:

    auc_with_state · auc_no_state · auc_cost · n_confirmed_proxy_states
    fairness_audit_threshold

`fairness_scalars()` (`pipelines/training_flow.py`) says so in its own docstring --
it "drops fair_df and the layer DataFrames". So Layer 1's fifty states, fifty
bootstrap CIs and fifty verdicts were reduced to one float, `n_confirmed_proxy_states`,
and on the authoritative run that float is `0.0`. The per-state ratios existed only
as stdout from `run_fairness_audit()`'s `print()`, hand-copied into two docstrings
and two docs. The number the README leads with had no machine-readable source.

**Why it could not be a live route, and why that is a different reason from `/drift`'s.**
`POST /drift` wraps `pipelines/drift_check.py` live, and that works because
`MockBureau` generates its own population: ~0.4s per request. The fairness audit has
two properties the drift monitor does not.

1. **It needs the data, and the data may not ship.** `run_fairness_audit()` calls
   `load_raw()`, which reads the 167 MB assessment CSV. That file is the FIRST line
   of `.dockerignore`, with the reason written there: the brief is explicit that it
   must not be redistributed. This is NOT the same class of exclusion as `/drift`'s.
   `pipelines/` is kept out of the image for size and dependency reasons and could be
   copied back in tomorrow. The dataset cannot. No live fairness route can ever work
   in the image, and no amount of engineering changes that.
2. **It costs ~40s.** Measured on the real 1,347,681-row CSV: 2.7s to load,
   ~28s for `run_fairness_audit()` (Layer 3 retrains two full LightGBM models on
   454k rows; Layer 1 bootstraps 2,000 resamples per state across 50 states), plus
   ~10.5s to put CIs on both sides of the ablation. No caching makes that a request,
   and there is nothing to parameterize anyway -- a fairness audit is a
   population-level fact about a model, not a what-if.

Worth recording because it was checked rather than assumed: `import src.fairness`
pulls in **no** mlflow and **no** metaflow (the Phase 2 `model_io.py` split already
closed that path). The import graph was not the obstacle. The data and the 40 seconds
were.

**The decision: freeze the OUTPUT, and commit it.** `scripts/audit_fairness.py` runs
the real `run_fairness_audit()` offline and writes ~38 KB of **derived aggregate
ratios** -- 50 EO ratios with bootstrap CIs, the threshold sweep, the ablation -- to
`models/fairness_audit.json`. Aggregate ratios are not the dataset, which is exactly
what makes shipping them legitimate where shipping the CSV is not.

It is **committed**, and its two `.pkl` neighbours are not. That asymmetry is the
point: `models/` and `data/*.csv` are BOTH gitignored, so a fresh clone can neither
serve the model nor regenerate this audit. As a build artifact, the evidence behind
the repo's loudest fairness claim would exist only on whichever machine last ran it,
and the README would be asserting it on nobody's authority.

`.gitignore` needed `models/*`, not `models/`. Git does not descend into an excluded
DIRECTORY, so a `!models/fairness_audit.json` negation inside one silently does
nothing -- the file stays ignored and the negation looks like it works. Verified with
`git add --dry-run` / `git ls-files --others --exclude-standard`, not with
`git check-ignore`, which exits 0 when a NEGATION matches and therefore reads like
"ignored" when it means the opposite.

**The staleness gate -- the decision this entry exists for.** A frozen artifact is
the one thing in this service that can disagree with the booster. `GET /calibrator`
cannot: it reads the live `ArtifactBundle`, so whatever it returns IS what `/score`
decides with. A JSON file has no such protection. Retrain, and it still cheerfully
reports Mississippi at 0.7448 about a model that no longer exists -- the exact
"say != do" this repo audits everywhere else, self-inflicted.

The mechanism was already in the repo and was reused rather than reinvented:
`load_calibrator()` (`src/calibrate.py`) refuses a calibrator fit against a different
model instance, binding on `trained_at`. `serving/fairness.py`'s `is_stale()` binds
the audit the same way, on the same field, against `ArtifactBundle.model_trained_at`.
On mismatch, `GET /fairness` returns **409 with both timestamps and NOT ONE RATIO**.

Sending the numbers with a warning attached was the other option and was rejected. It
is not a middle ground: a client that is handed ratios will draw them, and a reader
remembers the chart, not the caveat. The only reliable way to stop a stale number
being rendered as a current one is to never send it. An audit that cannot NAME the
model it ran against is treated as stale too -- absence of a provenance field is not
evidence of a match.

**Fail-closed on the numbers; fail-open on the service.** The audit is a REPORTING
signal (blue in `architecture.html`, beside `@explain`), not a gate. A missing,
corrupt or stale artifact yields a 404 or 409 on `/fairness` and never stops `/score`
-- the same policy `training_flow.py`'s `explain` step already applies, and for the
same reason: a broken observability signal must not throw away a decision the model
is perfectly capable of making. `load_fairness_audit()` therefore never raises; it
reports. `tests/test_serving.py` is what guarantees the SHIPPED artifact is present,
parseable and fresh -- a runtime crash is not. Verified: `/score` and `/healthz`
return 200 with the artifact fresh, stale, and absent.

**Two defects the artifact surfaced, both fixed, neither visible by reading the JSON.**

- `SWEEP_THRESHOLDS` (`src/fairness.py`) is the notebook's `[0.12 ... 0.22, 0.26,
  0.30]` and does **not** contain the operating threshold. A client asking "what does
  the shipped model approve at the cutoff it actually uses?" had to read the NEAREST
  row (0.26) and call it the operating one -- reporting a real, different approval
  rate as if it were exact. `scripts/audit_fairness.py` now passes the operating
  point into `audit_layer2` explicitly; `src/fairness.py`'s constant is untouched.
- `pandas.to_json` defaults to `double_precision=10` and **silently rounds every
  float to ten decimals**. Harmless for an EO ratio, and fatal for exactly one number
  here: `SELECTED_THRESHOLD` is `0.25000000000000006` and it was being written as
  `0.25` -- a genuinely different float (`serving/config.py` spends a comment on the
  difference). The sweep row was present and the `==` lookup for it still failed.
  `_records()` no longer goes through `to_json`. The 0.26 row happens to round to the
  same 82.4%, so the DISPLAYED number would have coincided -- the mechanism was wrong,
  not (this time) the output, which is precisely the class of defect that ships.

**Disposition:** `src/` scoring logic untouched; no decision changes value. The one
`src/` change is additive and recorded in the entry below.

**TODO:** none. If the model is retrained, re-run `uv run python
scripts/audit_fairness.py` -- the freshness test goes red until you do, which is the
intended behaviour, not a nuisance.

---

## `audit_layer3_ablation()` returns the Test frames for both variants: the ablation gets a confidence interval on each side

**Date:** 2026-07-13.

**Context:** Layer 3 trains a with-state model and a no-state model, scores both on
Test, reduces each to one EO ratio per state, and dropped the predictions on the
floor. That left the repo's headline fairness claim as **two bare point estimates**
-- "MS is 0.745 with the label and 0.988 without" -- which is precisely the credulity
`audit_layer1()` exists to refuse. Layer 1 bootstraps a CI *because* a point estimate
below 0.80 on a finite sample is noise rather than evidence. The ablation was making
exactly the kind of claim the rest of the module refuses to make.

**The change, and its boundaries.** `audit_layer3_ablation()` now returns
`fair_df_with_state` and `fair_df_no_state` -- Test-level frames in the
`addr_state` / `y_true` / `p` shape that `audit_layer1()` and `audit_layer2()`
ALREADY take as input. A caller therefore puts a real bootstrap CI on both sides of
the ablation by **reusing the real audit**, rather than a second CI implementation
growing up downstream.

Additive only. No existing computation is touched: both frames are slices of arrays
the function already had in hand. The printed audit report is **byte-for-byte
identical** before and after (sha256 `4147b1b0b9b53d9d`, 4,282 bytes, verified against
the real CSV). Nothing already in-tree pays the extra bootstrap: `run_fairness_audit()`
still audits only the shipped model, and `fairness_scalars()` reduces the dict to
scalars and drops the frames. Feeding them back through `audit_layer1()` costs ~10.5s
(2,000 resamples x 50 states x 2 variants) and is the caller's decision to make, not
the function's.

**What the intervals show** (threshold 0.22, the ablation's own):

    state   EO with state     95% CI              EO no state    95% CI
    MS      0.7448            [0.7155, 0.7741]    0.9879         [0.9621, 1.0124]

MS is `confirmed (CI fully < 0.80)` with the state label and `clear` without it, and
the two intervals **do not overlap**. Exactly one state is confirmed with `addr_state`
in; **zero** are confirmed with it out. The claim is now an interval claim: the shift
survives a bootstrap, which two point estimates could never have established.

**The cost of looking: Layer 3's own verdict column is weaker than it sounds.** That
column is decided on POINT ESTIMATES (`eo_with_state < EO_THRESHOLD`), and it
disagrees with the intervals. `NV` is labelled "was already clear" on
`eo_with_state = 0.800147` -- a verdict turning on the fourth decimal -- while its CI
`[0.7823, 0.8183]` straddles 0.80 and is honestly "inconclusive". `AL` likewise. The
logic is left exactly as it is (it mirrors notebook Cell 41); this change only makes
the evidence available to say so, and the frontend says so rather than picking
whichever reading flatters the demo. Trusting a point estimate is the specific error
this audit exists to refuse, and the audit was not immune to it.

**Disposition:** `src/fairness.py` only; no scoring logic, no decision, no metric
value changes. Committed separately (`18daa36`) from everything that consumes it, so
the `src/` diff can be reviewed in isolation.

**TODO:** none.

---

## The MS EO spread `0.734-0.745` / `0.988-0.990` is superseded, and the guard that should have caught it now exists

**Date:** 2026-07-17.

**Superseded, and by what.** The Layer-3 spread this file recorded at lines 125-126 --
Mississippi's good-applicant EO ratio "~0.734–0.745 with state included and ~0.988–0.990
with state removed" -- is superseded by run `cca4c361` and by
`models/fairness_audit.json`. The authoritative values are the table at line 1237, under
"`audit_layer3_ablation()` returns the Test frames for both variants":

    state   EO with state     95% CI              EO no state    95% CI
    MS      0.7448            [0.7155, 0.7741]    0.9879         [0.9621, 1.0124]

Lines 125-126 and 508 are **kept exactly as they are**, per this file's append-only rule
and the precedent 6e759b8 set for the AUC. They record what was believed before an
authoritative run existed. This entry is the pointer from here on.

**What the spread actually was.** Not a bootstrap CI, not a threshold spread, not a typo:
a spread across two different runs of the same quantity, both at threshold 0.22.
`0.734`/`0.990` are the notebook's run (Cell 41, taken with `INCLUDE_ADDR_STATE = True`);
`0.745`/`0.988` are `src/fairness.py`'s. Every endpoint is a real output of a real run.
Neither pair was ever an interval.

**Why it survived 8 days and three doc passes: 6e759b8 checked containment.** That commit
read the authoritative AUC out of `cca4c361` at full precision and corrected
`src/fairness.py`'s docstrings from 0.6689 to 0.6690. About the EO, in the same breath,
it wrote: "MS's Layer-3 EO recovery is 0.744823 -> 0.987926, **inside** the 0.734-0.745 /
0.988-0.990 spread this doc recorded earlier."

It verified that the new point estimate fell inside the old spread, concluded the spread
was therefore still true, and left it standing. One commit, two epochs: the AUC got a
point estimate, the EO kept a spread -- in adjacent lines of the same docstring.
`docs/design.md` was written 7h34m later and inherited exactly that split, citing
`cca4c361` for the AUC and the pre-`cca4c361` spread for the EO, in one sentence.

**Why containment was the wrong test.** A spread that happens to contain the point
estimate is not a confidence interval. Both containments here are byproducts of the
bootstrap being wide enough to swallow the older run:

    0.734 in [0.7155, 0.7741]        0.990 in [0.9621, 1.0124]

and the CIs those intervals come from did not exist until 2026-07-13, six days AFTER the
spread was written. The containment is arithmetic, not derivation. Reporting the spread as
though it were an interval launders a stale run into a statistical claim -- worse than a
plainly wrong number, because it reads as evidence. It is the same credulity
`audit_layer1()` bootstraps to refuse, committed by the documents describing the audit.

**The structural fix, and why it went in first.** `tests/test_docs_fairness.py`
(`e9b1286`) pins every current-system fairness number to `models/fairness_audit.json`: ten
named sites, each bracketed by a prose-anchored region and declaring which artifact key
paths it owes, at the precision it writes. A registry, not a free-grep over floats -- that
would hit `uv.lock` timestamps, the ~0.4s bureau latency, Brier scores and the AUC pair.

It was committed **RED**, failing on all six real sites, before one was edited.
Hand-editing first and writing the guard after is precisely what 6e759b8 did. `2efb632`
turned them green.

It rejects the range FORM, not merely stale values. A site quoting `~0.734–0.745`
satisfies a bare "does 0.745 appear?" check and goes green while saying something false --
that is the loophole containment walked through. The next retrain reddens CI instead of
shipping a stale claim.

It answers only "does the doc say the number the artifact holds?", never "is that number
good evidence?". `tests/test_serving.py`'s
`test_the_ablation_is_the_evidence_the_readme_claims_it_is` answers the second, and
correctly refuses to pin point estimates. Two questions, two files; merging them would
re-commit the error the CIs were added to fix.

**The exemption rule -- which files are ALLOWED to go stale.** Stated in a document here
for the first time. Until now it lived only inside 6e759b8's message, as a remark about
line 127. A rule a guard enforces but no document states is itself a say-equals-do hole.

    Files that RECORD HISTORY are exempt.
    Files that DESCRIBE THE CURRENT SYSTEM are not.

- **This file is exempt.** Append-only. Superseded numbers stay standing as the record of
  what was believed when; corrections are appended, never edited in. A guard that reddens
  it is arguing with its design.
- **`notebooks/analysis.ipynb` is exempt.** A with-state historical artifact. Its `0.734`
  is that run's true output at `INCLUDE_ADDR_STATE = True` and is correct in context;
  "fixing" it would make the notebook lie about its own run.
- **Everything else quoting these numbers is not**, and is in the registry:
  `docs/design.md`, `README.md` (x2), `frontend/README.md`, `src/fairness.py` (x2),
  `src/features.py`, `tests/test_fairness.py` (x2), `frontend/src/lib/api.ts`.

**Provenance, checked rather than assumed.** `design.md` now sources both the AUC and the
EO from `cca4c361`, but the artifact is NOT that run's output file: it was generated
2026-07-13 by `scripts/audit_fairness.py`, `cca4c361` ran 2026-07-09, and
`model.trained_at` moved (07-09 -> 07-11) between them. They agree anyway, and
deterministically -- `audit_layer3_ablation()` trains both variants itself at fixed
`num_boost_round` with no early stopping and never reads the shipped model, so Layer 3
does not depend on `model.trained_at` at all. `cca4c361` recorded `0.744823 -> 0.987926`
and `auc_with_state = 0.6690306566920463`; the artifact holds `0.7448231090766632 ->
0.9879261537678061` and that AUC at identical full double precision.

**Disposition:** docs, docstrings and comments only. No scoring logic, no threshold, no
decision, no metric value changes -- verified by parsing `HEAD`'s and the working tree's
`src/fairness.py` and `src/features.py`, stripping every docstring node, and comparing
`ast.dump()`: identical for both files.

**TODO:** none. If the model is retrained, re-run `uv run python
scripts/audit_fairness.py`; `tests/test_docs_fairness.py` then goes red for every doc
still quoting the old ratios, which is the intended behaviour, not a nuisance.

---

## A blast radius grepped for stale values is a damage inventory, not an exposure inventory -- and the safer site is the one you find first

**Date:** 2026-07-17.

**Context.** The `0.734-0.745` drift was found by grepping `0.734|0.745|0.988|0.990`
across the repo. That sweep reported a blast radius and the radius was acted on: nine
sites, six of them stale, all fixed by `2efb632` and pinned by `e9b1286`. Two more sites
quoting the same finding were found only afterwards, one per subsequent commit
(`3f0c12a`, `4b0bc27`). This entry is not a correction of the roster -- `_SITES` is the
roster's source of truth and `ea88407` stands as written. It records why the sweep could
not have found them, because that reason generalises and currently exists only inside a
commit message. `6e759b8` is this repo's proof of what happens to a rule that lives only
there.

**1. A search for stale values can only find sites that are already wrong.** The pattern
`0.734|0.745|0.988|0.990` is a list of the values known to be bad. It matches a site if
and only if that site is already broken. Both misses were **correct at the time**:

    .gitignore:46      "recovering 0.74 -> 0.99"                    (2 dp, true)
    README.md:341      "(94 of them in `tests/test_serving.py`)"    (exact, true)

Neither contains any of the four patterns, so neither could be seen, no matter how
carefully the output was read. The sweep was **complete as a damage inventory and
incomplete as an exposure inventory**, and those are different documents answering
different questions -- "what is broken?" versus "what is unbound?". A search finds only
what it already knows to look for; the sweep knew the stale values, so it found staleness,
and staleness was never the property that mattered.

The corollary was stated in `3f0c12a`'s message and is quoted here rather than restated,
because that commit is its source:

> A registry's job is not to catch what has drifted. It is to bind everything that could.

**2. Risk is inversely correlated with how conspicuous an unroped site is.** Of the two
that hid, the one found first was the **safer** one:

    .gitignore:46   0.74 / 0.99, 2 dp     rounding slack -- survives drift in the 3rd decimal
    README.md:341   94, exact integer     zero buffer -- one added test and it is wrong

`94` had no rounding to absorb anything, and `tests/test_serving.py` is the file the next
phase of this project modifies. The site that looked most harmless -- a parenthetical
aside about a test count, not a fairness ratio at all -- was the closest to breaking. It
was found last precisely because nothing about it drew attention. Conspicuousness is not
evidence of risk, and the two are anti-correlated often enough to plan around.

**3. A guard written against an already-correct site must be proven to fail before it is
believed.** `e9b1286` was committed RED against six real sites; its bite was never in
question. `4b0bc27`'s per-file pin
(`test_readme_per_file_counts_match_the_live_collection`) had no bug to redden it -- `94`
was correct -- so it passed on arrival, which is exactly the condition under which a guard
can be decoration and look identical to a guard that works.

So one was manufactured. `README.md:341`'s `94` was changed to `95`, nothing else touched,
and the full suite run: **1 failed, 301 passed** -- that test and only that test.
Reverted; green at 302. **A guard that passes and a guard that bites are not the same
thing**, and for a guard born green the difference is not observable without doing this.
The general rule: if a pin is written against a site that is already right, mutate the
site and watch it fail, or the pin is untested code asserting nothing.

**4. The pin binds the shape, not the instance.** `_PER_FILE_PATTERN` in
`tests/test_readme.py` matches `(N of them in `tests/<file>`)` generically, not the
`test_serving.py` row that prompted it, and resolves each claim against
`item.nodeid.split("::")[0]` on the live collection. A future
`36 of them in tests/test_explain.py` is roped the moment it is written, by a test nobody
remembers to update. Binding only the number that happens to exist today is the same error
this arc exists to fix -- one level up.

**Where the boundary between the two guards sits, and why.** They are split by SOURCE,
not by file: `tests/test_docs_fairness.py` pins to `models/fairness_audit.json`;
`tests/test_readme.py` pins to the live pytest collection. `94`'s true source is
`len(collect from tests/test_serving.py)` -- a collection fact -- so it went to
`test_readme.py` even though it lives in the same README line as a number the fairness
registry watches. The artifact has no test-count key; expressing `94` in the fairness
registry would have required inventing one.

**The exemption is load-bearing here for the first time.** This entry quotes `0.734` and
`0.74 / 0.99` in prose, and it can only do that because this file is exempt from
`_SITES` -- `_SPREAD` would fire on `0.734-0.745` above and redden the guard permanently
otherwise. That is not a loophole being exploited. It is the trade the exemption rule
names: *files that record history are exempt; files that describe the current system are
not.* An append-only log's right to hold superseded numbers is precisely what lets the
guard be strict everywhere else, because the history has somewhere to live that is not a
docstring. A repo with no exempt file either cannot write this entry or must weaken the
rule for every file to accommodate it.

**Disposition:** documentation only. No code, no `src/`, no test, no threshold, no metric
value changes. This entry adds no collected item, so the count stays 302.

**TODO:** none. The next blast-radius sweep should search for the SITES that quote a
governed quantity, not for the values currently believed wrong -- start from `_SITES` and
`_PER_FILE_PATTERN` and ask what is missing from them, not from a grep of what is broken.

---

## `serving/` reaches 59 third-party packages and tolerates 26 that the image does not install -- `matplotlib` was never the edge case, it was the only one priced

**Date:** 2026-07-17.

**Context.** This file's entry *"Why 'serving never reaches `matplotlib`' was false, and
why grep could not catch it"* is **true, and this entry does not correct it**. It asks
whether serving's import graph reaches four packages -- `mlflow`, `metaflow`, `seaborn`,
`matplotlib` -- and answers correctly: three never reached, one reached and tolerated.
Every word of that holds. What it did not do, and did not claim to do, is ask the general
question. Nobody did, for five months. This entry is that question, measured.

**The graph, from the interpreter and not from an estimate.** A fresh process, `import
serving.app`, `sys.modules` minus stdlib and first-party:

    3012 modules
    59 third-party top-level packages
    10 of pyproject.toml's 11 declared dependencies reached
    49 undeclared transitives

`uvicorn` is the eleventh, and it is absent for a good reason: it is the CMD, not an
import -- `serving.app` is what it loads, so it cannot be in `serving.app`'s own graph.

**26 of the 59 are in the graph and not in the image.** They are reached locally, where
`uv sync` installs the `dev` and `training` groups by default, and are absent from
`uv export --frozen --no-dev --no-group training`. Blocking all 26 at once with a
`MetaPathFinder` -- the image's import conditions, simulated -- and importing:

    import serving.app: OK  (51 third-party tops remaining)

**All 26 are reached-and-tolerated. The image is correct and has been all along.**

**Where they come from**, traced by hooking `builtins.__import__`:

    IPython      <- serving/app.py -> serving/artifacts.py -> src/explain.py -> shap
    numba        <- the same path
    yaml         <- shap -> numba
    matplotlib   <- serving/app.py -> serving/artifacts.py -> lightgbm
    PIL          <- lightgbm -> matplotlib
    defusedxml   <- matplotlib -> PIL
    psutil       <- lightgbm -> sklearn -> joblib
    pyarrow      <- serving/app.py -> pandas

`shap` drags in the whole of IPython: `jedi`, `parso`, `prompt_toolkit`, `pygments`,
`traitlets`, `asttokens`, `executing`, `stack_data`, `pure_eval`, `wcwidth`. A REPL's
dependency tree is in the import graph of a loan-scoring service, and no document in this
repo mentions it.

**The finding.** `matplotlib` is not the decisive edge case it reads as. **It is one of
26, and it is the only one that was priced.** `tests/test_serving.py` spends seventeen
lines explaining why `scripts` is in the graph legitimately. Not one line anywhere
explains why `IPython` is. The image has been right for months and the reason was never
written down -- which is this repo's own thesis pointed at itself: *the decision was right
and the justification was untrue* is the failure mode, and *the decision was right and the
justification was absent* is the same failure with less to argue with.

**The fifth blind spot, as the general form of the four before it.** The rule this file
set -- *don't grep, inspect `sys.modules`* -- was obeyed. It was then used to check
**four names**. The right instrument, pointed at an enumeration:

    grep                  -> sys.modules      matplotlib was reached; grep said clean
    stale-value grep      -> registry         0.734: only already-wrong sites are findable
    six-name blocklist    -> the property     openai is invisible; it does not exist yet
    sys.modules           -> the socket       MLflow's telemetry, where sys.modules ALSO
                                              passes it: test_training_flow.py imports
                                              mlflow and sends nothing. Telemetry fires on
                                              USE, not import (e8bc315).
    sys.modules, correctly, on a list of four -> this entry

Each time the instrument was upgraded and the *scope* was not. An enumeration is a damage
inventory of what has already gone wrong -- the same shape as the entry above this one,
one layer down the stack.

**The conflation, recorded here because `6764375` fixes its text but this is where the
reasoning lives.** `tests/test_serving.py`'s comment justified the import-graph roster
with the image being 937MB instead of 2.64GB. Those are two properties:

- **What loads at boot** -- `import serving.app`'s graph. Decides whether the container
  dies on startup. This is what the roster measures.
- **What the image installs** -- `uv sync --frozen --no-dev --no-group training`, plus
  the `COPY` lines and `apt-get install`. Decides the 937MB. The roster does not bound
  it.

Ways `serving/` can grow heavier that **no import-graph assertion can structurally see**:

1. **A handler-scope import.** `/drift`'s own pattern (`serving/app.py`'s handler-level
   `from scripts.demo_drift import drift_report`), built on purpose.
2. **A declared-but-unimported dependency.** `uv sync` installs what `pyproject.toml`
   declares; the graph only sees what is imported.
3. **A package gaining weight under the same import name.** Same graph, larger image.
4. **`apt-get install`.** `libgomp1` today.
5. **A larger `models/` COPY.**

**(2) is the sharpest, and it is the one that matters next.** `openai` in
`pyproject.toml`, imported inside a route handler: the image grows 5-10 MB and the import
graph is byte-identical. An allowlist over the graph -- the obvious inversion of the
roster -- would be **green on exactly the design it would be built to police**. That,
plus two other findings, is why no allowlist was built: a top-level allowlist can only
ban `scripts` (breaking the `find_spec` probe those seventeen lines defend) or admit it
(letting `scripts.demo_drift` back in), and its exception set would have to contain
`81d243bd2c585b0f4821__mypyc` -- a compiled artifact of `charset-normalizer` whose name
carries a build hash, so the list would expire when a dependency is **rebuilt**, not
upgraded.

**Disposition:** documentation only. No code, no `src/`, no `serving/`, no
`pyproject.toml`, no test, no threshold, no metric value. **The image is unchanged and was
already correct.** This entry buys the reason, not the outcome -- the same trade as the
`matplotlib` entry it generalizes, and made for the same reason that entry gives: the
conclusion is a fact about one day, the method is what stops the next person re-deriving
the false claim the same way. It adds no collected item, so the count stays 302.

**TODO:** one, and it is real rather than tidy. The tolerance of all 26 is a property of
**third-party** code -- `lightgbm` wraps its `matplotlib` import in `try`/`except
ImportError`, and something equivalent must hold for `shap`/`IPython` or the image could
not boot. That tolerance was verified here at today's pinned versions and **nothing
watches it**. Any upgrade can turn a guarded import into an unguarded one, and the failure
would appear as a container that dies at boot -- in a repo with no scheduler and no
automated `docker build` (`docs/design.md` §4), and with a CI that builds no image
(`.github/workflows/ci.yml`), that means it appears in front of whoever
runs the image next. The check is cheap and, unlike the rejected allowlist, **derivable
rather than hand-maintained**: the set is `graph - (uv export --no-dev --no-group
training)`, both sides computable, and the assertion is `import serving.app` under a
`MetaPathFinder` that blocks them -- a boot simulation, not a leanness claim. Whether that
is worth building is a decision nobody has made, and this entry does not make it. It is
recorded so that the decision is available rather than forgotten.

---

## A test name is a quantifier: `pulls_in_no_training_dependency` was false for as long as it was green

**Date:** 2026-07-17.

**Context.** `tests/test_serving.py` has carried a subprocess guard on serving's import
graph since `341418f`. It was named
`test_importing_serving_app_pulls_in_no_training_dependency`. It passed. It had always
passed. It was false the entire time.

    pyproject.toml   [dependency-groups] training = ["matplotlib>=3.11.0",
                                                     "metaflow>=2.19.35",
                                                     "mlflow>=3.14.0",
                                                     "seaborn>=0.13.2"]

    measured         $ python -c "import sys, serving.app; \
                                  print('matplotlib' in sys.modules)"
                     True

    the test         1 passed

`matplotlib` is a training dependency. It is in serving's import graph. The test that
says it is not was green — because it never asked. The watched list was a hand-copied
tuple holding three of the group's four members, and the member it omitted was the one
that leaks.

**What the name promised, and what the body checked.** The name quantifies over "training
dependency" — a universal, with a definition living in `pyproject.toml`. The body checked
membership in `('mlflow', 'metaflow', 'seaborn', 'pipelines', 'src.fairness',
'scripts.demo_drift')`, a literal typed by hand on some particular day. Those are not the
same set and nothing made them agree. A transcription is only as true as the moment
someone typed it, and nothing announces the moment it stops being true. This one had
stopped. The assertion could not see it, because the assertion was not looking at the
group — it was looking at a memory of the group.

**Both of us defended this name one round earlier, while hunting for exactly this.** The
review that produced `6764375` asked whether the name over-claimed — whether "no training
dependency" was being read as "serving stays lean" — concluded it did not, and wrote so.
That conclusion was correct and it was not the question. Nobody asked whether the
narrower claim was TRUE. The name looked safe *because* it was narrower than the comment
above it, and a narrower claim reads as a checked one. `6764375`'s commit message
contains a section titled **"WHY THE TEST WAS NEVER WRONG."** That sentence is false, it
is unamendable, and this entry is where it gets corrected. The assertion was never wrong.
The name was.

**Why code cannot hold this.** The test *should* be green. `matplotlib` is genuinely
tolerated: `serving/artifacts.py` imports `lightgbm`, `lightgbm.compat` does `try: import
matplotlib / except ImportError: MATPLOTLIB_INSTALLED = False`, and the slim image runs
without it. There is no defect to fix and no assertion to strengthen. The only thing
wrong was a word. That is precisely the class of thing this file exists for: a decision
code cannot carry.

**The fix, and what it moved.** `de52802` made the test derive `[dependency-groups]
training` from `pyproject.toml` instead of restating it, and landed **red on
`matplotlib`, on purpose** — the record that the claim was false before anyone fixed it,
the same reason `e8bc315` committed the network guard red, and the mistake `6e759b8`
made by squashing a finding into its fix. `3840807` then wrote the tolerance where it
executes:

    _TOLERATED = frozenset({"matplotlib"})

**That is the first time the exception has existed anywhere that runs.** It was already
correct, and already written five times, across four files: `pyproject.toml` twice (the
`[project]` comment and the training-group comment), `docs/design.md`'s serving-layer
section, this file's *"Why 'serving never reaches `matplotlib`' was false"* entry, and
`README.md`'s serving-layer section. Five correct copies, and a green test contradicting
all five, for as long as prose was the only place the exception lived. Prose is not what
a subprocess disagrees with.

**This entry sits beside `:916`, it does not correct it.** That entry says grep could not
catch `matplotlib` because `matplotlib` is never spelled in `serving/`, and only
`sys.modules` could. True, and true within its scope: it is about the *method* that finds
a leak. This is about a *name* that misdescribes a leak the method already found. `:916`
found the fact in 2026-07-12 and wrote it down five times. The test kept saying otherwise
for five more days. Neither entry is the other's correction.

**The tolerance is hand-written and cannot be derived — so it was given an expiry
instead.** "Which imports does `lightgbm` wrap in `try`/`except`" is a fact about
`lightgbm`'s source, not a row in a table this repo owns. A decision, not a definition.
Derivation is unavailable, so `_TOLERATED - present` must be empty: if `matplotlib` ever
stops being reached, the tolerance becomes a claim about a package nothing pulls and the
test says so. Mutation-tested — `_TOLERATED = {"matplotlib", "seaborn"}` fails. An
exception that cannot expire is how the transcription it replaced stayed green while
false.

**The name changed too, and the reason is worth keeping.** "No X" is a universal;
`matplotlib` is a counterexample; declaring the exception makes the *body* honest, not
the name. It is now `..._no_untolerated_training_dependency` — the quantifier gets its
qualifier, as in `88301a1`'s `_in_one_process`. The charitable reading ("training
dependency" means "something serving needs the training group for," which `matplotlib` is
not) is available and is what carried the name through review. It does not survive the
derivation: once the test reads the group instead of remembering it, that group's members
are what the name is quantified over.

### What is deliberately not written here

There is a wider shape visible behind this — a general law about instruments aimed at
enumerations, spanning several rounds of this audit. **It is not written, and the omission
is the point.** It has no enforcement: the only mechanism that would check it is a
name-versus-body checker, which this repo has twice declined as a meta-loop, and rightly
— a name is a claim in English and no rule reads it. Worse, the author of that law was
shown wrong on this very entry one round ago, and elevating a fresh finding into a
universal is the same move as the blocklist that caused it: generalising from what you
happened to notice. Two instances of a shape are a shape, not a law. It is recorded as
visible and unwritten so that the next person to see it knows it was seen and declined,
rather than missed.

### The method that found this had the same defect

The scan that produced this entry flagged test names by matching a **16-word list** of
quantifiers and scope words (`across_`, `every_`, `always_`, `never_`, `all_`, `any_`,
process, instance, route, …). It flagged 37 of 273 test functions. Those 37 were read by
hand. **236 were not read at all — a word list decided they were uninteresting.**

That word list is a damage inventory, not an exposure inventory, in exactly the sense
this file's `:1368` entry means it: it can only find names that used a word someone
thought of in advance. The proof is in this round's own results. The second finding
(`test_bureau_is_restricted_to_the_closed_enum`, a transcribed enum roster) was **not in
the 37**. It surfaced from a different, looser lens run as a self-check — one that was
nearly not run at all, and that flagged 85 further names nobody has read.

**So: four findings is what the list found. It is not what exists.** The number is a
lower bound and must not be read as a count. The same failure this entry describes, in
the method that discovered it, in the same round.
