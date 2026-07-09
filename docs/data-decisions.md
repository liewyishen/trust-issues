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