# Explainability

Findings from the reconnaissance pass on SHAP and the isotonic calibrator.
This is a record of what was measured, not a design. No solution is proposed
or chosen here; the open questions at the end are left open on purpose.

**Provenance.** Every empirical claim below is tagged with one of:
- `file:line` -- read directly from source.
- *computed from `models/isotonic_calibrator.pkl`* -- obtained by loading the
  artifact and reading `X_thresholds_` / `y_thresholds_`. No model was run and
  no split was loaded.
- *synthetic control run* -- a standalone numerical experiment on generated
  data, touching none of this project's models or data. Seeds are recorded so
  the numbers can be reproduced.

---

## 1. The gap

SHAP, as currently used, explains a different quantity than the one the
decision is made on.

`shap.TreeExplainer(model_ns)` is constructed without a `model_output`
argument (`notebooks/analysis.ipynb:1675`), so for a LightGBM booster trained
with `objective="binary"` (`src/train.py:94`) it attributes the model's **raw
log-odds margin**. The notebook says so in its own prose
(`notebooks/analysis.ipynb:1633`) and prints the same caveat at runtime
(`notebooks/analysis.ipynb:1760-1763`):

> SHAP explains the model's raw margin, not the calibrated probability.
> Production adverse-action notices require calibration-consistent,
> legally-reviewed reason mappings.

The decision, meanwhile, is made on the isotonic-calibrated probability. In
the same notebook cell the probability is computed on a **separate path** that
the SHAP values never touch: `p_shap_cal = iso_ns.transform(model_ns.predict(X_shap))`
(`notebooks/analysis.ipynb:1773`). The production path is the same shape --
`_predict_calibrated()` at `src/evaluate.py:219-258`.

Two transforms sit between the explained quantity and the decided quantity:

1. **sigmoid**, applied inside LightGBM when `Booster.predict()` converts the
   summed leaf margins into a probability.
2. **isotonic**, applied afterwards by the calibrator.

SHAP's additivity guarantee holds on the input side of transform 1. The
decision is read off the output side of transform 2. Nothing in the current
code bridges them, and the notebook is explicit that it does not
(`notebooks/analysis.ipynb:1633`).

---

## 2. What the calibrator actually is

*All numbers in this section: computed from `models/isotonic_calibrator.pkl`.*

The shipped calibrator is not a smooth curve. It is a **step function with 52
levels**.

| Property | Value |
| --- | --- |
| Knots (`len(X_thresholds_)`) | 104 |
| Distinct output values (`len(np.unique(y_thresholds_))`) | 52 |
| Flat blocks | 52 |
| Ramps connecting them | 51 |
| Domain | `X_min_ = 0.010907`, `X_max_ = 0.597072` |
| Range | `y_thresholds_[0] = 0.0`, `y_thresholds_[-1] = 0.558824` |
| Flat fraction, in-domain (X-measure) | **99.53%** |
| Flat fraction over `[0, 1]`, including both clip tails | **99.73%** |

The 104 knots are the start/end endpoints of the 52 blocks. Within a block the
curve is exactly flat; between blocks it rises across a ramp whose median
width is `3.7e-06` in `p_raw`. The 51 ramps have a combined width of `0.0027`
across a domain of span `0.586166`.

**Construction.** The calibrator is built with exactly one argument:

```python
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_calib_raw, y_calib)
```

(`src/calibrate.py:179-180`). `y_min` and `y_max` are left at their `None`
defaults; `increasing` is left at `True`. sklearn passes `y_min`/`y_max`
straight through to `isotonic_regression()`, and `None` means no clipping. All
five construction sites in the repo are identical: `src/calibrate.py:179`,
`src/fairness.py:368`, `src/fairness.py:586`,
`notebooks/analysis.ipynb:1119`, `notebooks/analysis.ipynb:1567`.

`out_of_bounds="clip"` (overriding sklearn's `'nan'` default) makes both tails
constant: `p_raw <= 0.010907` maps to `0.0`, and `p_raw > 0.597072` maps to
`0.558824`.

**Sample count.** Fit on **40,000** rows. `N_VAL = 40_000`
(`src/data_loader.py:49`); Val and Calib are disjoint positional slices of
shuffled 2015 (`src/data_loader.py:156-157`); `calibrate_model()` fits on
`splits["calib"]` (`src/calibrate.py:171`, `src/calibrate.py:180`).

**The domain is `p_raw`, not log-odds.** This is the single most important
structural fact in this document, and it is easy to get backwards.
`calibrate_model()` fits on `model.predict(X_calib_lgb, num_iteration=...)`
(`src/calibrate.py:176`, `src/calibrate.py:180`). With `objective="binary"`
(`src/train.py:94`), `Booster.predict()` returns **probabilities**, not raw
scores -- its `raw_score` argument defaults to `False` and is never passed.

The artifact confirms it: `X_min_ = 0.010907`, `X_max_ = 0.597072`. Those are
probabilities. A log-odds domain would straddle zero and extend well outside
`[0, 1]`.

So the calibrator is a map `p_raw -> p_cal`, both in probability units. The
SHAP values are in log-odds units. **They do not share an x-axis.**

---

## 3. Why 52 blocks is correct, not a bug

52 output levels from 40,000 training rows looks like aggressive compression.
It is not a defect in the fit, a bad hyperparameter, or a truncation. It is
the known convergence rate of isotonic regression.

**Mechanism.** PAVA (Pool Adjacent Violators) merges adjacent points whenever
the running means violate monotonicity. Sort Calib by `p_raw` and read off
`y in {0, 1}`: every `1` followed by a `0` is a violation, so merging cascades
almost everywhere. A block stops absorbing its neighbours only when its
empirical default rate is *separated from the adjacent blocks by more than
binomial noise*. Two forces oppose each other:

- Narrower block (in `p_raw`) => smaller true PD difference from its
  neighbours, scaling with block width.
- Narrower block => fewer samples inside it => larger noise in the empirical
  mean, scaling as `n_block ** -0.5`.

Setting the two equal gives an equilibrium block width proportional to
`n ** (-1/3)`, hence a block count of order `n ** (1/3)`. This is Groeneboom's
cube-root asymptotics: the number of jumps of the isotonic estimator is
`O_p(n ** (1/3))`.

**Check against the observed value.** `40000 ** (1/3) = 34.2`; observed 52;
ratio `1.52`.

**Synthetic control run.** Binary labels generated from a perfectly calibrated
`p ~ Beta(2.2, 8.0) * 0.6 + 0.01` (a support of roughly `[0.01, 0.61]`, chosen
to resemble the shipped calibrator's observed `p_raw` domain), `y ~ Bernoulli(p)`,
`IsotonicRegression(out_of_bounds="clip")`. No project model or split is
involved. Seed `= n`:

| n | blocks | `n ** (1/3)` | ratio |
| --- | --- | --- | --- |
| 1,000 | 9 | 10.0 | 0.90 |
| 10,000 | 38 | 21.5 | 1.76 |
| 40,000 | 53 | 34.2 | 1.55 |
| 160,000 | 74 | 54.3 | 1.36 |
| 640,000 | 117 | 86.2 | 1.36 |

Across a 640-fold increase in `n`, the block count grows 13-fold
(9 -> 117). `640 ** (1/3) = 8.6`. Consistent with a cube-root law;
flatly inconsistent with anything linear in `n`.

Block count is seed-sensitive at fixed `n`, so a single run should not be read
as a point prediction. Repeating the `n = 40,000` control across seeds 0..19
gives **min 38, max 57, mean 47.0, median 47.0**. The shipped calibrator's
**52** sits at the 95th percentile of that control distribution -- inside it,
toward the upper end.

Block count also tracks signal strength, as the mechanism predicts. At
`n = 40,000`, seed `40000`:

| Generating distribution for `p` | blocks |
| --- | --- |
| `Uniform(0.01, 0.99)` (wide spread) | 98 |
| `Beta(2.2, 8.0) * 0.6 + 0.01` (matches our domain) | 53 |
| `Uniform(0.15, 0.25)` (narrow) | 25 |
| `y` drawn independently of `p` (no signal) | 11 |

The shipped model's raw scores are compressed into `[0.010907, 0.597072]` and
its test AUC is `0.6660` (`src/calibrate.py:125`). That is the "matches our
domain" row. 52 blocks is what a weak ranker on 40,000 binary labels produces.

**`Y_max = 0.558824` is a PAVA block mean, not a `y_max` clip.** Four
independent lines of evidence:

1. `y_max` is never passed (`src/calibrate.py:179`); it defaults to `None`,
   which disables clipping.
2. `0.558824` equals `19 / 34` **bitwise**
   (`y_thresholds_[-1] == 19/34` evaluates to `True`; the value is
   `0.5588235294117647`). The top block therefore contains 34 Calib rows, 19 of
   which defaulted. A human-chosen ceiling would be `0.55` or `0.6`, not `19/34`.
3. It is consistent with the block geometry: the top flat block spans
   `p_raw in [0.497082, 0.597072]`, the far right tail, where only 34 of 40,000
   rows land. Sparse tail, wide block.
4. *Synthetic control run* (n = 4,000, seed 7): with `y_max=None` the last
   threshold came out at `1.000000`; with `y_max=0.558824` explicitly passed it
   came out at exactly `0.558824`. A clip reproduces the clip value exactly. The
   default reproduces whatever the data says.

**`Y_min = 0.0`** is the same phenomenon at the other end: the bottom block
(`p_raw in [0.010907, 0.017797]`) contains zero defaults, so its empirical mean
is `0.0`. Not a `y_min` floor -- `y_min` is also `None`.

---

## 4. Consequences for attribution at the decision boundary

*All numbers in this section: computed from `models/isotonic_calibrator.pkl`.*

Taking the reject rule as `p_cal >= 0.25`:

| Quantity | Value |
| --- | --- |
| `p_raw*`, the smallest raw score that is rejected | `0.228948` |
| Flat fraction of the reject region, in-domain (X-measure) | **99.31%** |
| Flat fraction including the clip tail `(0.597072, 1.0]` | 99.67% |
| Distinct `p_cal` values attainable at or above 0.25 | **21** |
| Flat blocks overlapping the reject region | 21 |
| Ramps overlapping the reject region | 21 |

**`p_cal = 0.25` is not an attainable value.** The calibrator's output takes
only 52 distinct values, and `0.25` is not one of them. The decision boundary
lands *inside a ramp*:

```
p_raw 0.22894648  ->  p_cal 0.241453
p_raw 0.22895049  ->  p_cal 0.264844
ramp width 4.013e-06   p_cal jump 0.023391   slope 5829.5
```

The nearest attainable outputs on either side are `0.241453` and `0.264844`.
Between them, a raw-score movement of four parts in a million flips the
decision.

**Half the raw axis maps to a single output.** Every `p_raw` in
`[0.497082, 1.0]` -- the top block plus the `out_of_bounds="clip"` tail --
maps to `p_cal = 0.558824`. That is one output value for `0.503` of the unit
interval in raw-score units.

The 0.25 boundary is not special in this respect. Re-solving at the other
thresholds present in the repo (`ABLATION_THRESHOLD = 0.22` at
`src/fairness.py:115`; `DEFAULT_AUDIT_THRESHOLD = 0.26` at
`src/fairness.py:107`) gives in-domain flat fractions of `99.35%` and
`99.31%`. `0.25` and `0.26` in fact back-solve onto the *same ramp*
(`p_raw* = 0.228948` and `0.228950`).

**Caveat on the measure.** All of the above is Lebesgue measure on the
`p_raw` axis, not applicant mass. No split was loaded, so "99.31% of the
reject region" means 99.31% of its *length*, and emphatically **not** "99.31%
of rejected applicants". A supporting argument that needs no data: the 51
ramps have combined width `0.0027`, so under any raw-score density that is
bounded, the share of applicants landing on a ramp is negligible. That is an
argument, not a measurement.

---

## 5. What this rules out

### (a) Local-slope chain rule

Rescaling raw-margin SHAP into probability units by multiplying through the
calibrator's local derivative.

**Mechanism.** `d p_cal / d p_raw` is *exactly zero* on every flat block, by
construction -- the block is a constant. Multiplying every feature's
contribution by zero annihilates the entire explanation.

**The numbers.** Within the reject region, `99.31%` of the X-measure carries
slope exactly `0.0`. On the remaining `0.69%` -- the 21 ramps -- the slope
ranges from `26.0` to `6914.7` (median `872.3`). Over the full domain the
ramp slopes reach `78991.5`. The derivative is either zero or effectively
divergent; there is no regime in between. At the decision boundary itself the
slope is `5829.5` across a ramp `4.013e-06` wide.

### (b) Black-box SHAP on the composed function

Running a model-agnostic explainer on `f(x) = calibrate(model(x))`.

**Mechanism.** `f` is piecewise constant: a 52-step function overall, and only
21 distinct values anywhere in the reject region. Model-agnostic explainers
(KernelSHAP, permutation, occlusion) estimate contributions from the change in
`f` under coalition masking or perturbation. When `f` is locally constant, most
perturbations produce **no change in the output**, so the attribution is zero
for reasons that have nothing to do with the feature's importance.

Whatever signal does survive comes from the small subset of perturbations that
happen to cross a block boundary -- and the boundaries are the `0.0027`-wide
ramps. The result is dominated by which side of a step the perturbed point
lands on.

### (c) The rank-preservation argument

The argument that isotonic is monotone, therefore ranks are preserved,
therefore raw-margin explanations transfer unchanged.

**Mechanism.** The argument equivocates on "monotone". `IsotonicRegression`
with `increasing=True` guarantees **non-decreasing**, not **strictly
increasing**. Non-decreasing permits ties, and this calibrator is built almost
entirely out of ties: 52 flat blocks covering 99.53% of the domain.

**The number that kills it.** Every applicant with `p_raw` anywhere in
`[0.497082, 1.0]` receives `p_cal = 0.558824`. Their raw-score ranking is
total; their calibrated ranking is a single tie. Ranks are **collapsed**,
not preserved.

This matters beyond attribution. It is also the mechanism behind the AUC
change already documented at `src/calibrate.py:128-132` -- "isotonic's flat
regions occasionally tying two examples that had different raw scores". That
docstring describes the same phenomenon at a scale where it costs `0.0006`
AUC. At the level of an individual explanation it is not a rounding effect;
it is the whole story.

---

## 6. What still works

Nothing here impugns the SHAP values themselves.

- **Raw-margin SHAP is mathematically sound.** TreeSHAP's additivity holds
  exactly on the log-odds axis: `base_value + sum(shap_values) == raw_margin`.
  That identity is not affected by anything downstream.
- **The ranking of contributions on the raw axis is intact.** Which features
  pushed this applicant toward default, in what order, by how much *in log-odds
  units* -- all well-defined and already computed
  (`notebooks/analysis.ipynb:1790-1807`).
- **Only the translation into probability points is undefined.** The sentence
  "this feature added N percentage points to your default probability" has no
  well-defined value under a piecewise-constant calibrator. The sentence "this
  feature was the largest single contributor to the score that led to the
  decision" is unaffected.

**Regulatory context, offered as context and not as a legal conclusion --
nobody in this repository is a lawyer.** ECOA / Regulation B adverse-action
notices call for the *principal reasons* for the decision, stated
specifically and in rank order. They do not, on their face, require numeric
per-feature contributions. If that reading holds, the artifact this project
already produces -- a ranked list of risk-increasing factors
(`notebooks/analysis.ipynb:1800-1807`) -- is closer to the required form than
a percentage-point decomposition would be. This needs review by someone
qualified to give it. The notebook's existing disclaimer
(`notebooks/analysis.ipynb:1633`) already says the reason codes are prototypes
pending exactly that review, and this document does not change that status.

---

## 7. The trade-off we did not know we were making

Isotonic regression was chosen on calibration quality. The real-run numbers
recorded in `calibrate_model()`'s docstring (`src/calibrate.py:124-126`):

```
Actual test default rate:      0.2323
Raw (uncalibrated):     Brier=0.1717  mean_pred=0.1705  AUC=0.6660
Calibrated (isotonic):  Brier=0.1692  mean_pred=0.1915  AUC=0.6654
```

Brier improves by `0.0025`. That was the whole basis for the choice, and the
module is careful to say the improvement is a refinement rather than a rescue
(`src/calibrate.py:17-23`): removing `scale_pos_weight` did roughly 90% of the
calibration work before isotonic ever ran.

What was never written down is the **cost**: isotonic buys that Brier
improvement by making the probability scale piecewise constant. Sections 4 and
5 are that cost, itemized. It was paid silently.

Parametric alternatives -- Platt scaling (a sigmoid on the raw score, two
parameters) and beta calibration (three parameters) -- are differentiable
everywhere and strictly increasing. Under either, `d p_cal / d p_raw > 0` at
every point, ranks are preserved strictly rather than collapsed, and
percentage-point attribution has a well-defined value.

**Their Brier score on this project's calib/test split has never been
measured.** Not in `src/calibrate.py`, not in the notebook, not anywhere in
the repo. The trade -- monotone-but-flat with a known Brier of `0.1692`, versus
smooth-and-invertible with an unknown Brier -- has never been priced. We do not
currently know whether it costs `0.0001` or `0.005`, and therefore we do not
know whether the attribution property is expensive or nearly free.

Stating this plainly: the current calibrator was selected without anyone
noticing that the selection foreclosed probability-scale attribution.

---

## 7a. The trade-off was priced. It does not exist.

Section 7 above is left exactly as written. It records what was believed before
anything was measured, and its central assumption was wrong -- not about the
magnitude, about the **sign**.

Section 7 assumed smooth calibration would buy differentiability and rank
preservation *at some unknown Brier cost*, and asked whether that cost was
`0.0001` or `0.005`. It is neither. On this model and these splits, Platt
scaling and beta calibration are **better** calibrated than isotonic, not
worse. There is no cost to pay.

*Provenance for this entire section: two throwaway scripts run outside the
repository (`price_calibrators.py`, `refit_variance.py`), reading `src/` but
changing nothing. `IsotonicRegression(out_of_bounds="clip")` per
`src/calibrate.py:179`; Platt is `LogisticRegression(penalty=None)` on
`logit(p_raw)`, which recovers the model's raw log-odds margin exactly since
`p_raw = sigmoid(margin)`; beta calibration is `betacal==1.1.0` (Kull et al.'s
reference implementation), installed into a scratch `--target` directory --
neither `.venv` nor `uv.lock` nor `pyproject.toml` was modified, and `betacal`
is not importable from the project environment. Nothing was hand-rolled.*

**Sanity check on the harness.** Refitting isotonic on `splits["calib"]`
reproduces the shipped `models/isotonic_calibrator.pkl` **bitwise**:
`np.array_equal` on both `X_thresholds_` and `y_thresholds_` returns `True`,
104 knots each. The splits and the model reproduce, so the three calibrators
are genuinely fit on the same 40,000 calib rows and scored on the same rows.

### Design bound -- what this study does and does not vary

The LightGBM booster is trained on **2007-2014** and is **fixed** across every
seed. Re-shuffling 2015 and re-cutting Val/Calib (mirroring
`src/data_loader.py:152-157`) changes only *which 40,000 rows the calibrator is
fit on*. `splits["test"]` (2016-2017, n=462,174) and `splits["holdout_2018"]`
(n=56,160) are **identical across all seeds**; raw scores were predicted once
per split and reused.

This therefore measures **calibration-slice variance only**. It does not vary
the model fit, the train/test boundary, the year definitions, or the feature
set. A regime-shift stress test beyond 2018 is a different question, not
answered here.

### Refit variance: 10 seeds x 2 splits

**TEST (2016-2017), n=462,174, actual default rate 0.232252**

| Calibrator | mean Brier | sd | min | max | mean AUC | distinct outputs |
| --- | --- | --- | --- | --- | --- | --- |
| isotonic | 0.169278 | 1.85e-04 | 0.168985 | 0.169534 | 0.665458 | **463** |
| Platt | 0.169192 | 1.80e-04 | 0.168890 | 0.169458 | 0.665982 | 455,099 |
| beta | **0.169174** | 1.86e-04 | 0.168869 | 0.169433 | 0.665982 | 455,099 |

| Paired delta (TEST) | mean | sd | min | max | t (df=9) |
| --- | --- | --- | --- | --- | --- |
| Platt − isotonic | −8.551e-05 | 3.39e-05 | −1.390e-04 | **−3.508e-05** | −7.98 |
| beta − isotonic | −1.035e-04 | 3.35e-05 | −1.660e-04 | **−5.649e-05** | −9.77 |

**HOLDOUT_2018, n=56,160, actual default rate 0.1575** -- a genuinely different
regime; this year is excluded from Test for suspected right-censoring
(`src/data_loader.py:129-132`).

| Calibrator | mean Brier | sd | min | max | mean AUC |
| --- | --- | --- | --- | --- | --- |
| isotonic | 0.126332 | 7.66e-05 | 0.126233 | 0.126472 | 0.673203 |
| Platt | 0.126214 | 8.17e-05 | 0.126102 | 0.126337 | 0.673817 |
| beta | **0.126200** | 7.91e-05 | 0.126084 | 0.126315 | 0.673817 |

| Paired delta (HOLDOUT_2018) | mean | sd | min | max | t (df=9) |
| --- | --- | --- | --- | --- | --- |
| Platt − isotonic | −1.176e-04 | 3.18e-05 | −1.533e-04 | **−7.086e-05** | −11.69 |
| beta − isotonic | −1.315e-04 | 3.12e-05 | −1.713e-04 | **−9.273e-05** | −13.33 |

The `max` column is the finding: **it never reaches zero.** Even the worst seed
for Platt beats isotonic by 3.5e-05 on Test.

### Sign flips: 0 out of 40

Across 2 calibrators x 2 splits x 10 seeds, the sign never flips.

| Split | Platt worse than isotonic | beta worse than isotonic |
| --- | --- | --- |
| TEST | **0/10** | **0/10** |
| HOLDOUT_2018 | **0/10** | **0/10** |

Isotonic is last on Brier in 10/10 seeds on both splits. The only ordering that
moves is *between* the two smooth calibrators: 9/10 seeds give
`beta < platt < isotonic`; seed 2 gives `platt < beta < isotonic`.

### Why the comparison must be paired

This is the statistical crux and it is easy to get wrong. The across-seed sd of
isotonic's own Brier on Test is **1.85e-04** -- *larger* than the mean delta of
**8.55e-05**. Comparing marginal means without pairing would read as noise and
the finding would be dismissed.

It is not noise. Within each seed, all three calibrators are fit on the
identical calib slice and scored on the identical test rows, so the comparison
is paired, and the paired sd is **3.39e-05**. The seed-to-seed wobble is a
*common mode* that moves all three calibrators together; it cancels in the
difference. That is why the levels are loose and the deltas are tight.

### Distinct output values

On Test, isotonic produces **463** distinct probabilities from 462,174 rows:
its 52 block values plus the handful of rows landing on ramps. Platt and beta
each produce **455,099** -- exactly the number of distinct `p_raw` values in
Test. Both are strictly increasing bijections, so they preserve every raw score
as a distinct probability. Nothing is tied that was not already tied.

### Minimum slope over the reject region (`p_cal >= 0.25`)

| Calibrator | `p_raw*` | min `dp_cal/dp_raw` | rejected share of Test |
| --- | --- | --- | --- |
| isotonic | 0.228948 | **0** (exactly) | 21.98% |
| Platt | 0.220823 | 0.960376 | 24.44% |
| beta | 0.218886 | 0.793324 | 25.04% |

Isotonic's is exactly zero, computed from the knot segments rather than a grid
(the ramps are ~4e-06 wide and a grid would alias them). Platt's numeric slope
was cross-checked against the closed form `w * p_cal(1-p_cal) / (p(1-p))` and
agrees to six decimals at `0.960376`. Fitted parameters: Platt
`w=1.054148, c=0.230536`; beta `a=1.163201, b=0.605803, m=0.463044`.

This is what Section 5 said was impossible under isotonic and is simply
available under either smooth calibrator.

### The effect size is small, and is not on its own a reason to swap

Stated plainly, because the direction of the result invites overreach. The mean
paired delta is **0.051% of Brier** (Platt on Test) to **0.104%** (beta on
HOLDOUT_2018):

| Split | Platt | beta |
| --- | --- | --- |
| TEST | 0.0505% | 0.0612% |
| HOLDOUT_2018 | 0.0931% | 0.1041% |

An improvement of five hundredths of one percent in Brier is not a reason to
change a shipped artifact. **The reason to swap, if there is one, is
differentiability and rank preservation** -- the properties Sections 4 and 5
established that isotonic cannot provide at all. Brier establishes only that
acquiring those properties **costs nothing**. It converts the question from a
trade-off into a free option. It does not, by itself, argue for exercising it.

### The 2018 widening

The advantage is *larger* on the shifted-regime holdout than on Test: Platt
improves by −8.551e-05 on Test versus **−1.176e-04** on HOLDOUT_2018, with a
tighter t-statistic (−7.98 vs −11.69). Beta likewise (−1.035e-04 vs
−1.315e-04).

One reading is that isotonic's 52-level step function partly memorizes the 2015
calib slice's local noise, and that a lookup table transfers worse than a two-
or three-parameter monotone curve when the population's default rate moves from
0.2323 to 0.1575. **That is a pattern consistent with the numbers, not an
isolated mechanism.** Nothing in this study controls for the alternative
explanations, and no experiment here was designed to separate them. It is
recorded because the direction is informative, not because it is established.

### Closing the Section 4 open question: ramp occupancy is measured

Section 4 argued from bounded density that the share of applicants landing on
one of the 51 ramps must be negligible, and explicitly labelled that an
argument rather than a measurement. It has now been measured, on
`splits["test"]` with the shipped calibrator:

| Quantity | Count | Share |
| --- | --- | --- |
| Test rows | 462,174 | -- |
| Landing on a flat block | 461,758 | 99.9100% |
| **Landing on a ramp** | **416** | **0.0900%** |
| Rejected rows (`p_cal >= 0.25`) | 101,601 | 21.98% |
| **Rejected and on a ramp** | **198** | **0.195% of rejected** |

The argument was right. It is now a measurement, and Section 4's X-measure
figures (99.53% flat in-domain, 99.31% flat over the reject region) are
confirmed to translate into applicant mass: **99.91% of scored applicants sit
on a flat block**, where `dp_cal/dp_raw` is exactly zero.

---

## 8. Open questions

Recorded, not resolved. Nothing below should be treated as decided.

- ~~**What is Platt scaling's Brier score on our calib/test split?** Beta
  calibration's? Unmeasured.~~ **ANSWERED, see Section 7a.** Both are *lower*
  than isotonic's, on Test and on HOLDOUT_2018, across 10 calibration-slice
  refits, with 0/40 sign flips.
- ~~**Is the Brier difference worth the loss of attribution?**~~ **DISSOLVED,
  see Section 7a.** There is no loss to weigh: the smooth calibrators are
  better on Brier, not worse. The question was posed as a trade-off and the
  trade-off does not exist. What remains is not "is it worth it" but "should a
  shipped artifact be changed at all" -- a different question, recorded below.
- **Does serving actually need numeric contributions, or do rank-ordered
  reason codes suffice?** This is a requirements question, not a modeling
  question, and it may make everything above moot. Still open.
- ~~**What fraction of applicants actually land on a ramp?**~~ **ANSWERED, see
  Section 7a.** 416 of 462,174 Test rows (0.0900%); 198 of 101,601 rejected
  rows (0.195%). Section 4's bounded-density argument was correct, and is now
  a measurement rather than an argument.
- **Should the shipped calibrator be replaced?** Not decided here, and
  deliberately not argued either way in this document. Replacing it would
  re-open `best_threshold`, `test_profit`, `approval_rate`, every number in
  `docs/data-decisions.md` that cites them, and an unknown number of tests.
  Section 7a establishes only that the swap would cost nothing on Brier --
  which is a fact about calibration quality, not a decision about a shipped
  artifact.
- **Are `model_ns` and the shipped booster the same model?**
  `notebooks/analysis.ipynb:1565` retrains `model_ns` with a fixed
  `num_boost_round=best_rounds` and no early stopping; `models/lgbm_model.pkl`
  carries `best_iteration = 240` produced by early stopping in `train_lgb()`.
  Their feature sets match (8 columns, no `addr_state`). Whether the boosters
  are identical has **not** been verified -- no tree dump comparison was made.
- **What was the with-state SHAP run, and where did its output go?**
  `notebooks/analysis.ipynb:1626` confirms that a with-state SHAP version once
  existed -- it displayed `addr_state=TX` as a mitigating factor, which is why
  it was replaced. **None of its output survives**, in any tracked or untracked
  file in this repository.

  Scratch notes preceding this investigation cite three specific values
  attributed to that run: a `loan_amnt` SHAP contribution, a `fico_n` SHAP
  contribution, and a rejected applicant's `P(default)`. **Those digits are
  deliberately not reproduced here.** They have no provenance. Writing them
  into a tracked file would make them discoverable by `grep`, where a future
  reader could mistake presence-in-repo for provenance -- which is exactly how
  an unsourced number becomes a cited one. The finding is that they have no
  source; restating them would undercut the finding.

  The negative search, recorded so it need not be repeated: `grep` across
  `notebooks/analysis.ipynb`, `notebooks/analysis.html`, `README.md`, `docs/`,
  and `.ipynb_checkpoints/` returns nothing; `git log -S` across both commits
  that touch the notebook (`83fbd1f`, `3c7ae29`) returns nothing.

  What the committed notebook *does* contain is sourced, and describes a
  different applicant entirely: `P(default) = 0.236`, `loan_amnt = 20000` at
  `SHAP = +0.1606`, `fico_n = 697.0` at `SHAP = +0.0255`
  (`notebooks/analysis.ipynb:1723-1735`). Any explanation-related number
  quoted from this project should trace to that cell, or to a fresh run --
  never to the scratch notes.

---

## 9. Appended note: a latent `IndexError` in the notebook's SHAP cell

Found while writing `src/explain.py` (Batch C, Step 3). Recorded here rather
than fixed; the notebook is a historical artifact and nothing in `src/` depends
on it.

`shap.TreeExplainer.expected_value` **is mutated by `shap_values()`**. Before
the first call it is a length-1 `ndarray`; after the call it is a scalar
`np.float64`. Verified against the shipped booster under `shap==0.50.0`:

```
at construction:  ndarray  array([-1.69927944])   size 1
after shap_values():       np.float64(-1.6992794393197017)
```

The notebook's global SHAP cell reads it like this:

```python
# notebooks/analysis.ipynb:1675-1680
explainer = shap.TreeExplainer(model_ns)
sv = explainer.shap_values(X_shap)                       # :1676
shap_default = sv[1] if isinstance(sv, list) else sv
base_value = (explainer.expected_value[1]                # :1678
              if isinstance(explainer.expected_value, (list, np.ndarray))
              else explainer.expected_value)
```

The `isinstance(..., np.ndarray)` branch at `:1678` indexes `[1]` on what it
believes is a two-element per-class array. At construction that array has
**length 1**, so `expected_value[1]` raises:

```
IndexError: index 1 is out of bounds for axis 0 with size 1
```

The cell only works because `shap_values()` on line `:1676` runs first and
collapses `expected_value` to a scalar, which sends `:1678` down the `else`
branch instead. **The correctness of that cell is a property of its line
order, not of its logic.** Move the `base_value` assignment above the
`shap_values` call -- a refactor with no obvious hazard -- and it raises.

`src/explain.py` avoids this structurally rather than by ordering luck:
`_shap_matrix()` reads `expected_value` only after the `shap_values()` call it
made itself, normalizes size-1 and size-2 arrays explicitly, and raises a named
`ValueError` on any other size instead of indexing blind.
`tests/test_explain.py` pins the read-after-call ordering with a stub explainer
that reproduces the mutation.

Not fixed in the notebook. Flagged so the next person to touch that cell knows
what they are standing on.

---

## 10. The explainer is a wrapper

Recorded, not decided. Nothing in this repo changes because of this section
except the guard armed in `fix(explain): the additivity guard was inert -- arm it`.

### TreeExplainer's fast path for LightGBM is one `predict` call

For a LightGBM booster constructed with no background data,
`shap.TreeExplainer(booster).shap_values(X, tree_limit=L)` computes nothing of
its own. `feature_perturbation="auto"` resolves to `"tree_path_dependent"`
(`_tree.py:248`), which selects the fast path (`_tree.py:551-555`), whose entire
body for this model type is:

```python
phi = self.model.original_model.predict(X, num_iteration=tree_limit, pred_contrib=True)   # :580
...
self.expected_value = phi[0, -1]    # :615
out = phi[:, :-1]                   # :616
return out                          # :622
```

The contributions are LightGBM's. The `expected_value` attribute is a cached
copy of a column shap already held.

**Bit-identical, measured:** 200 real Test rows x 8 features, `tree_limit =
best_iteration = 240`. Maximum absolute difference `0.000000e+00`; `0 / 1600`
differing cells; `np.array_equal` is `True`. Not "agrees to tolerance" -- the
same floats. `explainer.expected_value == phi[0, -1]` exactly, and `phi[:, -1]`
is constant across rows (`ptp = 0.0`). Pinned by
`test_shap_values_equals_pred_contrib_on_the_shipped_artifact`.

### 65 ms of construction is discarded work on this path

| Operation | min | median |
|---|---|---|
| `shap.TreeExplainer(booster)` | 64.91 ms | 69.19 ms |
| `explainer.shap_values(1 row, tree_limit)` | 1.50 ms | 7.60 ms |
| `booster.predict(1 row, pred_contrib=True)` | 1.46 ms | 2.07 ms |

`TreeExplainer.__init__` parses the booster into a `TreeEnsemble`
(`_tree.py:279`) and precomputes an `expected_value` from `model.values`
(`_tree.py:321-325`). The fast path reads neither: it overwrites
`expected_value` from `phi` and never touches the parsed arrays. The
construction cost is real and its product is unused. `shap_values()` also emits
a `UserWarning` on every call (`_tree.py:586-589`); `pred_contrib` emits none.

### `check_additivity` is inert here

`shap_values(..., check_additivity=True)` verifies nothing for LightGBM.
`model_output_vals` is assigned only inside the xgboost branch
(`_tree.py:573-576`) and stays `None` otherwise (`_tree.py:556`), so the guard

```python
if check_additivity and model_output_vals is not None:   # _tree.py:618
```

is short-circuited on every call this repo makes. Demonstrated: a frame with
`fico_n` set to all-NaN passes `check_additivity=True` without raising.

`src/explain.py` claimed in a comment that shap performed this check "for
free". It never did. **Fixed** in commit `5705e10` -- `_assert_additivity()` now
compares `base + contributions.sum(axis=1)` against
`booster.predict(..., raw_score=True)` and raises past `ADDITIVITY_ATOL = 1e-9`
(measured float64 noise: worst `7.77e-15` over 200 rows; shap's own tolerance,
when it does run, is `atol=rtol=1e-2` at `_tree.py:845`). The guard runs on the
serving path, costs one extra predict (0.73-1.70 ms for a single row), and
catches a `tree_limit` mismatch -- explain 240 trees, score 10 -- which was
previously silent.

### What a migration would buy, and what it would cost

Calling `booster.predict(X, num_iteration=best_iteration, pred_contrib=True)`
directly and slicing `phi[:, :-1]` / `phi[:, -1]` would remove **both** the
~65 ms construction and the shared-mutable-`expected_value` hazard: the base
value would arrive inside each call's return value, so two concurrent requests
could not cross-contaminate. Under a FastAPI service with a cached explainer,
that hazard is a real silent-wrong-answer race -- request A's write at
`_tree.py:615` can land between request B's call and B's read at
`src/explain.py:219`, and no exception is raised.

The price is that three LightGBM output conventions, currently absorbed by shap,
become ours to uphold:

1. **last-column-is-base** -- `phi[:, -1]` is the base value, `phi[:, :-1]` the
   contributions.
2. **single-output-means-2D** -- true only because `objective="binary"` and
   `num_model_per_iteration() == 1`. A multiclass booster returns
   `(n, n_classes * (n_features + 1))`, and the naive slice would silently mix
   classes. shap reshapes for this at `_tree.py:590-598`.
3. **`num_iteration=0` means all trees**, not none. Measured: `0`, `-1`, and
   `None` all return the full 240-tree contributions. A booster that ever saved
   `best_iteration = 0` would silently be explained in full.

It would also touch public API -- `explain_applicants(..., explainer=)`,
`global_importance(..., explainer=)`, `_get_explainer()`, and the
one-explainer amortization in `run_explanation()` all become dead -- and the
stub-explainer tests in `tests/test_explain.py` lose their subject. `shap`
would become droppable from `pyproject.toml`, since `src/explain.py:67` is the
only `import shap` in the repo.

**Deliberately not done.** Recorded, not decided. The equivalence is now pinned
by a test, so whoever takes this on is making a decision rather than a leap.
