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

**Drop `addr_state`.** The ablation put Mississippi's good-applicant
Equal-Opportunity ratio at ~0.734–0.745 with state included and ~0.988–0.990 without.
The price, from the authoritative run `cca4c361`, is a test-AUC cost of **0.0036**
(0.6690 → 0.6654). We paid it: the state label was a geographic proxy, not
Mississippi's economics.

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

- **`serving/` is not deployed.** `serving/app.py` answers HTTP requests in-process and under Docker, but nothing runs it as a live service — no host, no orchestration.
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
- **`drift_check.py` is a manual entry point**, not a scheduled job — no CI, no cron,
  no scheduler exists here. It reports; it does not raise (`fail_on_alarm=False`).
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
  the two disagree.
- **`LOAN_SCHEMA` cannot validate a live request.** It requires `Default` and
  `addr_state` as non-nullable columns (`src/data_validation.py`'s `LOAN_SCHEMA`); a
  request has neither. The gate is a *training* contract.
- **`TreeExplainer.expected_value` is instance state**, overwritten by every
  `shap_values()` call, so a shared explainer is unsafe under concurrency
  ([`explainability.md`](explainability.md) §10).
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
- **Every decision is returned with the pull it was made on.** `ScoreResponse`
  carries one nested `credit_report` key (`ScoredCreditReport`): the fetched
  `fico_n` the booster actually scored on, plus the `bureau`, `fico_version` and
  `pulled_at` identifying which pull supplied it. A decision made on a
  bureau-sourced `fico_n` therefore ships both the value and the report's
  identity — the provenance a real bureau integration would audit against, and
  the datum needed to check the decision against it.
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

> Rendered to A4 at 10pt with the diagram at 45% width: 2 pages (headless Chromium,
> `/Type /Page` count). At 11pt with the diagram inline it is 3.
