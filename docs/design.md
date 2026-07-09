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
dashed is not wired into the flow** — `pipelines/drift_check.py` (a manual entry point)
and `src/explain.py` (no caller but its tests) are drawn outside the pipeline, not as
connected. No calibrator edge reaches `@fairness`, which refits its own isotonic
(`src/fairness.py:586-587`). Every label's file:line is in
[`architecture.html`](architecture.html).

## 3. Design decisions and their costs

**Split by time; hold out 2018.** A random split lets the model peek at future
patterns. Train is 2007–2014 (n≈453,804), Validation and Calibration are disjoint
40,000-row slices of 2015, Test is 2016–2017 (n=462,174) (`src/data_loader.py:144-149`);
2018 is withheld for suspected right-censoring. The price: a lower headline number, and
a training set that never sees the 2016+ DTI regime — the train/serve shift the drift
monitor watches.

**Drop `addr_state`.** The ablation put Mississippi's good-applicant
Equal-Opportunity ratio at ~0.734–0.745 with state included and ~0.988–0.990 without.
The price, from the authoritative run `cca4c361`, is a test-AUC cost of **0.0036**
(0.6690 → 0.6654). We paid it: the state label was a geographic proxy, not
Mississippi's economics.

**Optimize regret, not pure profit.** The objectives have different break-even
thresholds for a well-calibrated probability — `m/(m+LGD) = 0.156` for pure profit,
`2m/(2m+LGD) = 0.270` once the margin foregone on a wrongly-rejected good borrower is
counted (`src/evaluate.py:84`, `:93`). The data matches 0.270: the term was always
there implicitly, and the decision was to name it and make it switchable.

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

- **`serving/` does not exist.** Nothing here answers an HTTP request.
- **`drift_check.py` is a manual entry point**, not a scheduled job — no CI, no cron,
  no scheduler exists here. It reports; it does not raise (`fail_on_alarm=False`).
- **Global SHAP importance is not logged to MLflow.** It belongs beside test AUC in the
  same run; today it prints and returns.
- **`SHAP_SAMPLE_N = 4000`** is inherited from the notebook, with no stated rationale.
  It is not a measured convergence threshold (`src/explain.py:130-136`).

## 5. Known constraints on serving

Constraints a serving design must answer, not defects to apologize for.

- **The selected threshold is in no artifact.** `best_threshold = 0.25000000000000006`
  lives only in MLflow run history; neither pickle carries a threshold. A service must
  take it as config or read it from MLflow. It is not `0.25` — at `p_cal` exactly 0.25
  the two disagree.
- **`LOAN_SCHEMA` cannot validate a live request.** It requires `Default` and
  `addr_state` as non-nullable columns (`src/data_validation.py:171-183`); a request
  has neither. The gate is a *training* contract.
- **`TreeExplainer.expected_value` is instance state**, overwritten by every
  `shap_values()` call, so a shared explainer is unsafe under concurrency
  ([`explainability.md`](explainability.md) §10).

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
