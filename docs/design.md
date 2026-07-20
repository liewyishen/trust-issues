# Design

## 1. What the system does

This repository trains a credit-default model on the LendingClub granting-model
dataset (2007–2018), scoring an applicant's default probability from **seven raw
application-time fields**, which feature engineering turns into **eight model
features** — `emp_length` yields both an ordinal and an explicit not-disclosed flag
(`src/features.py`). It takes the target from the data's own label, splits by
origination year, audits for leakage and fairness, and calibrates so the score can be
priced. Test ROC-AUC is about 0.67 — the ceiling without credit-bureau history — and
that is the result, not a problem to optimize away.

## 2. The diagram

![Architecture](architecture.png)

Colour carries meaning and nothing else. **Red is a gate that raises and halts the
run**: the schema check and three of the four leakage sentinels. **Blue reports and
halts nothing**: MLflow metrics, the temporal sentinel's SKIP, the drift check. **Grey
dashed is not wired into the flow** — `pipelines/drift_check.py` (a manual entry point) is
drawn outside the pipeline, not as connected. (`src/explain.py` was drawn there too; the
`explain` step now wires it in, and the diagram — re-traced 2026-07-11 after the Phase 2
`train.py` → `model_io.py` split — draws it in the flow.) No calibrator edge reaches
`@fairness`, which refits its own isotonic inside `run_fairness_audit()`. Every label's
symbol citation is in [`architecture.html`](architecture.html).

## 3. Design decisions and their costs

**Split by time; hold out 2018.** A random split lets the model peek at future
patterns. Train is 2007–2014 (n≈453,804), Validation and Calibration are disjoint
40,000-row slices of 2015, Test is 2016–2017 (n=462,174) (`src/data_loader.py`'s
`temporal_split()` docstring); 2018 is withheld for suspected right-censoring. The price: a
lower headline number, and a training set that never sees the 2016+ DTI regime — the
train/serve shift the drift monitor watches.

**Drop `addr_state`.** The ablation (threshold 0.22) put Mississippi's good-applicant
Equal-Opportunity ratio at **0.745, 95% CI [0.716, 0.774]** with state included — the
whole interval under the 0.80 benchmark — and **0.988, CI [0.962, 1.012]** without, the
two intervals disjoint. The price is a test-AUC cost of **0.0036** (0.6690 → 0.6654).
Both sides are the authoritative run `cca4c361`'s, reproduced to full precision by the
frozen `models/fairness_audit.json` that `GET /fairness` serves. We paid it: the state
label was a geographic proxy, not Mississippi's economics.

**Optimize regret, not pure profit.** The objectives have different break-even
thresholds for a well-calibrated probability — `m/(m+LGD) = 0.156` for pure profit,
`2m/(2m+LGD) = 0.270` once the margin foregone on a wrongly-rejected good borrower is
counted (`src/evaluate.py`'s `pure_profit()`/`regret_profit()` docstrings). The data
matches 0.270: the term was always there implicitly, and the decision was to name it and
make it switchable.

**Isotonic calibration — chosen without pricing it.** Isotonic was selected on Brier
alone, and nobody noticed the choice foreclosed probability-scale attribution: a
52-level step function has zero slope across 99.31% of the reject region, so "this
feature added N points of default probability" has no value to compute. It has since
been priced. Platt scaling and beta calibration are **better** calibrated than isotonic
— 10 calibration-slice refits across two splits, **0 of 40 sign flips** — though the
margin is tiny, 0.051% to 0.104% of Brier. Brier is the wrong place to look. At the
*same* threshold 0.25, Platt rejects **11,376 more** Test rows than isotonic (24.44%
vs 21.98%) and beta **14,106 more** (25.04%): the three calibrators put `p_cal = 0.25`
at different raw scores (`p_raw*` = 0.228948 / 0.220823 / 0.218886). Holding the
threshold fixed does not hold the policy fixed. A swap is not a drop-in —
`best_threshold` must be re-selected on Validation under the cost objective, and
`test_profit`, `approval_rate` and the fairness audit's operating threshold move with
it. Free in Brier; not free in policy. **Replacement is deferred, and the deferral is
a decision, not an oversight.** Evidence: [`explainability.md`](explainability.md) §7a.

**Rank-ordered reason codes, not percentage-point attribution.** `src/explain.py` ships
the sign and rank of each contribution on the raw **log-odds** axis, scale declared in
a `scale` field and in every key name. Under the shipped calibrator, probability-point
attribution is undefined rather than uncomputed — so there is no
`contribution_to_probability` key, not even set to `None`.

Depth: [`data-decisions.md`](data-decisions.md) (data quality, fairness);
[`explainability.md`](explainability.md) (calibration, attribution).

## 4. Not yet wired

- **`serving/` is not deployed.** `serving/app.py` answers HTTP requests in-process, under Docker,
  and to `frontend/` on `localhost`, but nothing runs it as a live service — no host, no
  orchestration. A frontend now consumes it (§7), which changes who calls the API and changes
  nothing about where it runs. Deployment is the one step of Phase 2 still outstanding, and the
  `CORS_ALLOW_ORIGINS` allowlist (`serving/app.py`) is written for Vite's dev origins — it is a
  thing to revisit at deploy time, not a thing that already works in production.
- **Phase 2 did decouple `serving/` from MLflow, though.** `src/train.py` split into
  `src/train.py` (MLflow orchestration; `train_and_save()` only) and `src/model_io.py` (the
  encoding helpers, the LightGBM training loop, and `load_model_artifact()` — everything
  `serving/` and the pipeline's non-training steps actually import). `serving/`'s import
  graph now bottoms out at `model_io.py` and never reaches `mlflow`.
  `pyproject.toml` groups `mlflow` / `metaflow` / `matplotlib` / `seaborn` into a `training`
  dependency group the Docker build excludes; the serving image measured **2.64GB →
  937MB**. This is a change to what ships in the image, not a change to the bullet above —
  serving still is not deployed anywhere.
- **Three of those four are unreached; `matplotlib` is reached and tolerated.** The
  grouping above is correct, but the reason first written down for it was not, and the
  runtime check is what caught that. `mlflow`, `metaflow` and `seaborn` are genuinely
  absent from `sys.modules` after `import serving.app`. `matplotlib` is not:
  `serving/artifacts.py` imports `lightgbm`, and LightGBM's compat module imports
  `matplotlib` on the way in, so serving's import graph *does* reach it. Leaving it out of
  the image is still safe — LightGBM guards that import (`try` / `except ImportError`,
  setting `MATPLOTLIB_INSTALLED = False`), so it degrades instead of failing. The accurate
  claim is **tolerated, not unreached**, and the two are not the same sentence.
- **`drift_check.py` is a manual entry point**, not a scheduled job — no cron, no
  scheduler exists here, and the CI that does exist does not run it: it needs the 167 MB
  dataset that never ships. It reports; it does not raise (`fail_on_alarm=False`).
- **`SHAP_SAMPLE_N = 4000`** is inherited from the notebook, with no stated rationale
  (`src/explain.py`'s `SHAP_SAMPLE_N` constant). The `explain` step now logs it as the
  `shap_sample_n` param;
  the ranking it produces is rank-stable across seeds and sample sizes (measured 2026-07-11),
  so 4000 is pinned-and-logged rather than defended — but it stays an inherited constant, not
  a measured convergence threshold.

## 5. Known constraints on serving

Constraints a serving design must answer, not defects to apologize for.

- **The selected threshold is in no artifact.** `best_threshold = 0.25000000000000006`
  lives only in MLflow run history; neither pickle carries a threshold. A service must
  take it as config or read it from MLflow. It is not `0.25` — at `p_cal` exactly 0.25
  the two disagree. `serving/config.py` takes it as config (`SELECTED_THRESHOLD`), and
  the gap is unchanged: the artifact still does not carry it.
- **Clients are handed that threshold; they never assume it.** `GET /calibrator` returns
  `bundle.threshold` alongside the calibrator's knots for one reason — a client drawing
  the reject boundary must draw it where the service actually puts it. A client that
  hardcoded `0.25` would draw a line the service does not decide at, and the difference
  is real, not pedantic. The same rule holds for every constant the UI renders: the
  fairness `eo_threshold` and the drift `DEFAULT_ALARM_THRESHOLDS` are shipped on the
  wire rather than retyped in the browser (§7).
- **`LOAN_SCHEMA` cannot validate a live request.** It requires `Default` and
  `addr_state` as non-nullable columns (`src/data_validation.py`'s `LOAN_SCHEMA`); a
  request has neither. The gate is a *training* contract.
- **`TreeExplainer.expected_value` is instance state**, overwritten by every
  `shap_values()` call, so a shared explainer is unsafe under concurrency
  ([`explainability.md`](explainability.md) §10) — *in principle*. **Under the
  shipped execution model that race is unreachable**, and this sentence used to
  imply otherwise. `/score` is an `async def` handler that never awaits, so two
  calls do not interleave: measured, one thread, disjoint in-handler intervals.
  The danger is double-covered — once by rebuilding the explainer per request,
  once by non-overlap — and the second cover is one keyword deep. As a `def`
  handler `/score` runs in Starlette's threadpool and the intervals overlap, for
  no wall-clock gain (the work is CPU-bound). `tests/test_serving.py`'s
  `test_score_cannot_yield_the_event_loop_mid_request` is what makes "under the
  shipped execution model" a checked fact rather than this paragraph.
- **`fico_n` comes from a mock bureau.** `/score` fetches it through the
  credit-bureau layer (`serving/bureau.py`'s `CreditBureau` protocol), not off
  the request — a client that submits its own `fico_n` is rejected
  (`ScoreRequest`, `extra="forbid"`). The only implementation is `MockBureau`: a
  deterministic, `mean_fico`-configurable Normal draw, not a real vendor. The
  layer is real; its data source is simulated.
- **A pull may raise, and `/score` does not catch it.** `CreditBureau.fetch()`'s
  contract permits raising on a failed pull, but the handler calls
  `bureau.fetch(applicant_id)` unguarded because `MockBureau` cannot structurally
  fail — there is no reachable failure to handle yet. Real-vendor failure
  handling is a recorded deferral (`docs/data-decisions.md`'s Phase 1
  bureau-wiring entry), not an omission.
- **Only `fico_n` moved; `dti_n` stays applicant-reported.** Phase 1 relocated
  `fico_n` to the bureau but left `dti_n` on the request; `_to_raw_frame()` takes
  `dti_n` from the request and leaves the pulled `report.dti_n` unused. The
  bureau sources one of the two credit fields today, by design, not both.
- **Every decision is returned with the pull it was made on, and with its own
  explanation.** `/score` returns `ExplainedScoreResponse` (`serving/schema.py`),
  a **subclass** of `ScoreResponse`. From the base it carries one nested
  `credit_report` key (`ScoredCreditReport`): the fetched `fico_n` the booster
  actually scored on, plus the `bureau`, `fico_version` and `pulled_at`
  identifying which pull supplied it. A decision made on a bureau-sourced
  `fico_n` therefore ships both the value and the report's identity — the
  provenance a real bureau integration would audit against, and the datum needed
  to check the decision against it. The subclass adds exactly one field of its
  own, `explanation`, rendered from the base's fields by `serving/render.py` —
  pure code, no model. Putting it on the subclass rather than on `ScoreResponse`
  is *why* `set(ScoreResponse.model_fields) == set(explain keys) |
  {"credit_report"}` still holds untouched (`docs/data-decisions.md`): the
  renderer's input never contains the renderer's own output.
- **The pulled `dti_n` is absent from that block, deliberately.** `CreditReport`
  has a `dti_n`; the decision does not use it (`_to_raw_frame()` reads `dti_n`
  off the request). A "credit report" block showing a DTI the model never
  consumed would describe the decision one way while it was made another — see
  `docs/data-decisions.md`, which measures how far apart the two values are under
  `MockBureau`. The block carries what was scored, and nothing else.

## 6. Human-in-the-loop (not implemented)

A referral band on the calibrated probability — "review everyone between 0.20 and 0.30"
— **is not currently expressible.** The composed scoring function takes only 21 distinct
values anywhere in the reject region ([`explainability.md`](explainability.md) §4): a
band does not select a population, it selects a step.

One rule *is* well-defined today. An applicant can be rejected with every SHAP
contribution ≤ 0, when the base value alone clears the boundary; `reason_codes` is then
`[]`. That applicant cannot be issued an adverse-action notice listing principal
reasons, because there are none, and must route to review regardless of `p_cal`. That
is a structural property of the model, not an error state.

## 7. The interface: two ways to serve a number, and one way to refuse

`serving/` grew from two routes to five, and a `frontend/` that consumes them. The routes are not
variations on one idea — they divide cleanly into **two patterns**, and choosing between them is
the design decision worth recording.

**The `/calibrator` pattern: read the live artifact.** `GET /calibrator` returns
`IsotonicRegression.X_thresholds_` / `.y_thresholds_` straight off the `ArtifactBundle`
(`serving/artifacts.py`) that `/score` composes decisions with — the same object, not a second read
of the same path. It therefore **cannot go stale**: whatever it returns *is* what the service
decides with. The threshold travels with the curve for the same reason — a client drawing the
reject boundary must draw it where the service puts it (`SELECTED_THRESHOLD`, `serving/config.py`),
not at the literal `0.25`, which is a different float.

**The `/drift` pattern: wrap the live computation.** `POST /drift` re-implements no PSI, no KS and
no alarm rule. It turns `MockBureau`'s `mean_fico` knob, builds the two batches through the *same*
sampling path the CLI demo uses (`scripts/demo_drift.py`'s `drift_report()`), and hands them to
`pipelines/drift_check.py`'s real `drift_metrics()` / `evaluate_alarms()`. A second PSI in the
serving layer would be precisely the drift-between-two-sources this repo exists to prevent. This is
affordable because the mock bureau *generates* its own population: ~0.4 s on the first request,
~50 ms warm — a slider works on that.

**Fairness could use neither, and that is the interesting part.** `run_fairness_audit()` has both
properties the other two lack:

- **It needs the data, and the data may not ship.** `load_raw()` reads the 167 MB assessment CSV —
  the first line of `.dockerignore`, because the brief forbids redistributing it. This is *not* the
  same kind of exclusion as `/drift`'s: `pipelines/` is left out of the image for size and
  dependency reasons and could be copied back in. The dataset cannot. No live route can ever work
  in the image, and no engineering changes that.
- **It costs ~40 s.** `audit_layer3_ablation()` retrains two full LightGBM models on 454k rows;
  `audit_layer1()` bootstraps 2,000 resamples per state across 50 states. No caching makes that a
  request, and there is no knob to turn anyway — a fairness audit is a population-level fact about
  a model, not a what-if.

So fairness takes a **third pattern: freeze the output, and bind it to the model.**
`scripts/audit_fairness.py` runs the real audit offline and writes ~38 KB of *derived aggregate
ratios* — 50 EO ratios with bootstrap CIs, the threshold sweep, the ablation — to
`models/fairness_audit.json`. Aggregate ratios are not the dataset; that is what makes shipping
them legitimate where shipping the CSV is not. The artifact is **committed**, unlike its two `.pkl`
neighbours, because `models/` and `data/*.csv` are both gitignored: as a build output, the evidence
behind the repo's loudest fairness claim would exist only on the machine that last ran it.

**The staleness gate.** A frozen artifact is the one thing in this service that *can* disagree with
the booster. Retrain, and the JSON still reports Mississippi at 0.7448 about a model that no longer
exists — the same "say ≠ do" this repo audits everywhere else, self-inflicted. The mechanism was
already in the repo: `load_calibrator()` (`src/calibrate.py`) refuses a calibrator fit against a
different model instance, binding on `trained_at`. `serving/fairness.py`'s `is_stale()` binds the
audit the same way, on the same field, against `ArtifactBundle.model_trained_at`. On mismatch,
`GET /fairness` returns **409 with both timestamps and not one ratio**.

Sending the numbers with a warning attached is not a middle ground. A client handed ratios will
draw them, and a reader remembers the chart, not the caveat. The only reliable way to stop a stale
number being rendered as a current one is to never send it. An audit that cannot *name* the model
it ran against is treated as stale too — absence of a provenance field is not evidence of a match.

**Fail-closed on the numbers; fail-open on the service.** The audit is a *reporting* signal (blue in
[`architecture.html`](architecture.html), like `@explain`), not a gate. A missing, corrupt or stale
artifact yields a 404 or 409 on `/fairness` and **never stops `/score`** — the same policy
`training_flow.py`'s `explain` step already applies, and for the same reason: a broken observability
signal must not throw away a decision the model is perfectly capable of making. `load_fairness_audit()`
therefore never raises; it *reports*, and `tests/test_serving.py` is what guarantees the shipped
artifact is present, parseable and fresh. A runtime crash is not.

**The honest-404 principle, stated once.** Two routes are absent from some deployments — `/drift`
(mounted behind `DRIFT_DEMO_AVAILABLE`'s `find_spec` probe) and `/fairness` (when no artifact
ships). Both say so. Neither degrades to a plausible chart drawn from a default, a cached snapshot,
or a hardcoded curve. **An empty state that admits the endpoint is missing is strictly more useful
than a believable picture that came from nowhere** — and the frontend renders it that way, verified
against real services in both states.

**What the frontend is, and what it is not.** `frontend/` is React + TypeScript + Vite + Tailwind
over the live API: score one applicant, compare two *ceteris paribus*, and three views whose only
job is to make an existing claim checkable — the calibrator's 52-level step function, the drift
monitor firing beside an unmoved negative control, and the fairness ablation with an interval on
both sides. It **computes none of its own numbers**. The 0.80 fairness line is drawn at the
`eo_threshold` the response returns; the drift alarm line at `DEFAULT_ALARM_THRESHOLDS`; the reject
boundary at the threshold `/calibrator` hands back. Every one of those could have been typed as a
literal into the client, and each would have been a second source of truth for a number the service
already owns. The frontend makes the rigor visible. It does not add any.

> Rendered to A4 at 10pt with the diagram at 45% width: 2 pages (headless Chromium,
> `/Type /Page` count). At 11pt with the diagram inline it is 3.
