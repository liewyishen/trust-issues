   # trust-issues

   **A loan risk model that did the boring parts right.**

   Most loan-risk projects make the same fatal mistakes — wrong target definition, data leakage,
   random splits, and trusting inflated AUCs. This project has trust issues with all of that.
   It defines the target carefully, audits for leakage, splits by time, calibrates its
   probabilities, and checks fairness before trusting any score.

   Run the whole chain — load, validate, train, calibrate, evaluate, fairness-audit — with a
   single command:

   ```bash
   uv run python pipelines/training_flow.py run
   ```

   The honest result is a test AUC around **0.67**. That is the feature-limited ceiling when
   you allow yourself only application-time signals and no bureau credit history — and a 0.67
   you can trust beats a 0.99 you can't.

   ---

   ## What this project does right

   | Stage | What |
   |-------|------|
   | **1. Problem framing** | Binary default target taken from the dataset's own label (fully paid vs. defaulted) — no invented horizon; the cost-sensitive operating threshold is set later, in evaluation (LGD 0.65 / margin 0.12), not assumed upfront |
   | **2. Leakage-aware data prep** | Programmatic blocklist of post-decision fields + a pandera schema gate + temporal-consistency checks — leakage *fails* the pipeline, it isn't just warned about |
   | **3. Point-in-time-safe features** | All features constructed from data available at application date only |
   | **4. Baseline-first modeling** | Logistic-regression baseline before LightGBM; no premature complexity |
   | **5. Calibration + cost-based thresholds** | Isotonic calibration on a disjoint calibration slice; operating threshold chosen by expected profit, not 0.5 |
   | **6. Explainability + fairness** | SHAP explanations; a hand-rolled three-layer fairness audit — bootstrap CIs on the Equal-Opportunity ratio, a threshold sweep, and an ablation that retrains without `addr_state`. The audit caught `addr_state` acting as a digital-redlining shortcut, so the production model drops it: Mississippi's EO ratio recovers 0.73 → 0.99 for an AUC cost of just 0.0035 |

   ---

   ## Setup

   ```bash
   uv sync                                              # install dependencies
   uv run pytest                                        # run the test suite (75 passing)
   uv run python pipelines/training_flow.py run         # end-to-end training pipeline
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
   notebooks/     Exploratory analysis notebook (+ HTML export)
   pipelines/     Metaflow end-to-end training pipeline
   src/           Modeling layer: data loading, validation, features, leakage checks,
                  training, calibration, evaluation, fairness
   models/        Trained model + calibrator artifacts (gitignored)
   figures/       Generated plots (gitignored)
   tests/         pytest suite (75 passing)
   ```
