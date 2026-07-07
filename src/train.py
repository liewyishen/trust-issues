"""
Model training for the LendingClub granting-model dataset.

Fixes three pieces of notebooks/analysis.ipynb in place as reusable, testable
code:
  - Cell 21 : Logistic Regression baseline (sklearn Pipeline).
  - Cell 23 : LightGBM, early-stopped on Val, never on Test.
  - Cell 25 : the scale_pos_weight A/B test that justified NOT using it.

Feature contract comes from src/features.py (FEATURES/NUMERIC/CATEGORICAL/
TARGET/add_features) -- this module does not redeclare it, so the notebook
and this module can never quietly drift apart on "what the model's inputs
are". Splits come from src/data_loader.py's temporal_split() output.

MLflow experiment tracking: every training run (baseline and LightGBM) is
logged as its own MLflow run against a local SQLite database
(sqlite:///mlflow.db) -- no remote server needed. This module sets that
tracking URI explicitly (see below) rather than relying on MLflow's own
default, so training always writes to the same place regardless of which
directory it's invoked from. Once a few runs exist, run
`mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repo root to
browse and compare them by metric, instead of scrolling back through
terminal output.

Model packaging: the production LightGBM model is saved with joblib as a
dict bundling the Booster together with its full feature contract (feature
list, categorical columns, train-derived category mappings, chosen boosting
round, hyperparameters, and a training timestamp) -- not as a bare model
file. See train_and_save()'s docstring for why a bare model is a train/serve
skew hazard waiting to happen.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import RANDOM_SEED
from .features import CATEGORICAL, FEATURES, NUMERIC, TARGET, add_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# MLflow tracking setup: SQLite backend, not the plain filesystem store.
#
# An earlier version of this module pinned MLflow to a file-store tracking
# URI (file://.../mlruns) and had to set the environment variable
# MLFLOW_ALLOW_FILE_STORE=true to opt back into it, because newer MLflow
# (this project pins mlflow>=3.14) puts that plain filesystem backend into
# "maintenance mode" and refuses to create a new file-based run store
# otherwise. That opt-out flag was a workaround, not the recommended path --
# MLflow's own docs recommend a database backend for anything beyond a
# quick, throwaway experiment: faster queries as run count grows, and no
# maintenance-mode warning to suppress. SQLite is the simplest such backend
# that still needs no server process, so this module points MLflow at
# sqlite:///mlflow.db instead. `mlflow ui --backend-store-uri
# sqlite:///mlflow.db` (run from the repo root) now starts cleanly with no
# extra flags or environment variables required.
#
# An absolute path is used (not a bare "sqlite:///mlflow.db", which SQLAlchemy
# would resolve relative to the current working directory) so that training
# always writes to the same database file at PROJECT_ROOT/mlflow.db, no
# matter which directory the training script or notebook is run from.
#
# This only takes effect if the caller hasn't already configured a tracking
# URI (via the MLFLOW_TRACKING_URI env var, or an explicit
# mlflow.set_tracking_uri() call -- tests/test_train.py makes exactly that
# call, pointing at a tmp_path database, so test runs never touch the repo's
# real mlflow.db).
# ---------------------------------------------------------------------------
if "MLFLOW_TRACKING_URI" not in os.environ:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")

# LightGBM hyperparameters -- identical to notebooks/analysis.ipynb Cell 23.
# scale_pos_weight is deliberately absent here: it is added conditionally in
# train_lgb() only when use_spw=True, because the A/B test in
# run_spw_ablation() (Cell 25) found it does not improve ranking (Val AUC is
# unchanged) but roughly doubles the mean predicted probability -- a cost
# with no corresponding benefit. The shipped model omits it.
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": RANDOM_SEED,
}


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply feature engineering and split a split-frame into (X, y)."""
    engineered = add_features(df)
    return engineered[FEATURES], engineered[TARGET]


def _train_categories(X_train: pd.DataFrame) -> dict[str, pd.Index]:
    """
    Derive each categorical column's category set from TRAIN ONLY.

    Val, Test, and serving inputs must all reuse this exact mapping (see
    _to_lgb_frame) instead of deriving their own. If each split independently
    inferred its own categories, the same string value could be assigned a
    different internal integer code in different splits -- LightGBM encodes
    categoricals by integer code under the hood, so that mismatch would
    silently corrupt the model's learned category-to-risk associations
    without raising any error.
    """
    return {c: pd.Categorical(X_train[c]).categories for c in CATEGORICAL}


def _to_lgb_frame(X: pd.DataFrame, category_maps: dict[str, pd.Index]) -> pd.DataFrame:
    """
    Cast every categorical column in X to the train-derived categories.

    Any category value not present in category_maps[c] becomes NaN rather
    than raising -- this is what lets a split (or a live serving request)
    contain a category LightGBM's training data never saw (e.g. a new
    addr_state) without crashing prediction. LightGBM's native categorical
    handling treats NaN as its own bin, so this degrades gracefully instead
    of failing closed the way data_validation.py's schema gate does -- the
    two modules make different tradeoffs on purpose: a malformed/missing
    *feature value* at inference time shouldn't take down serving, whereas a
    malformed *training frame* should stop the pipeline.
    """
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=category_maps[c])
    return X


def _report_metrics(y: pd.Series, p) -> dict[str, float]:
    """
    Both ROC-AUC and PR-AUC (average precision) are computed for every split.
    Under this dataset's ~20% default rate, ROC-AUC alone can look better
    than the model deserves -- PR-AUC is the more honest number to trust
    under class imbalance, per the notebook's own framing.
    """
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def train_baseline(splits: dict[str, pd.DataFrame]) -> tuple[Pipeline, dict]:
    """
    Logistic Regression baseline: median/most-frequent imputation + scaling +
    one-hot, class_weight="balanced" to handle the ~20% imbalance via
    reweighting (not resampling). Mirrors notebooks/analysis.ipynb Cell 21.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        temporal_split() output; must contain "train", "val", "test".

    Returns
    -------
    (Pipeline, dict)
        The fitted sklearn Pipeline, and a metrics dict:
        {"train": {"auc":..., "pr_auc":...}, "val": {...}, "test": {...}}.
    """
    X_train, y_train = _xy(splits["train"])
    X_val, y_val = _xy(splits["val"])
    X_test, y_test = _xy(splits["test"])

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])

    lr = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED,
        )),
    ])
    # Pipeline.fit() fits the preprocessor's imputers/scaler/encoder ONLY on
    # X_train. Val and Test only ever go through .transform() inside
    # predict_proba() below -- their statistics never leak into training.
    lr.fit(X_train, y_train)

    metrics = {
        "train": _report_metrics(y_train, lr.predict_proba(X_train)[:, 1]),
        "val": _report_metrics(y_val, lr.predict_proba(X_val)[:, 1]),
        "test": _report_metrics(y_test, lr.predict_proba(X_test)[:, 1]),
    }
    return lr, metrics


def train_lgb(
    splits: dict[str, pd.DataFrame],
    use_spw: bool = False,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, int, dict, dict[str, pd.Index]]:
    """
    Train LightGBM with early stopping evaluated on Val, never on Test.
    Mirrors notebooks/analysis.ipynb Cell 23.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        temporal_split() output; must contain "train", "val", "test".
    use_spw : bool
        If True, adds scale_pos_weight = n_negative / n_positive (computed
        from Train) to LGB_PARAMS. Default False -- see run_spw_ablation()
        for why the shipped model omits it.
    num_boost_round, early_stopping_rounds : int
        Forwarded to lgb.train(); defaults match the notebook (1000, 50).
        Exposed as parameters so tests can use small synthetic data without
        waiting on the notebook's full-scale rounds.

    Returns
    -------
    (Booster, best_iteration, metrics, category_maps)
        best_iteration : the iteration early stopping selected using Val.
        metrics : {"train": {...}, "val": {...}, "test": {...}}, each with
            "auc" and "pr_auc", computed at best_iteration.
        category_maps : train-derived categories for each categorical
            column (see _train_categories) -- needed downstream to encode
            new data (val/test here, serving requests eventually) the exact
            same way train was encoded.
    """
    X_train, y_train = _xy(splits["train"])
    X_val, y_val = _xy(splits["val"])
    X_test, y_test = _xy(splits["test"])

    category_maps = _train_categories(X_train)
    X_train_lgb = _to_lgb_frame(X_train, category_maps)
    X_val_lgb = _to_lgb_frame(X_val, category_maps)
    X_test_lgb = _to_lgb_frame(X_test, category_maps)

    params = dict(LGB_PARAMS)
    if use_spw:
        params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()

    dtrain = lgb.Dataset(X_train_lgb, y_train, categorical_feature=CATEGORICAL)
    dval = lgb.Dataset(X_val_lgb, y_val, reference=dtrain)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        # valid_sets holds ONLY train and val, by construction -- X_test_lgb
        # is never wrapped in a Dataset passed here. Early stopping (and
        # therefore the choice of best_iteration, i.e. model selection)
        # structurally has no way to see Test. Test is only used below, once
        # best_iteration is already fixed, purely for reporting.
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100)],
    )

    best_iteration = booster.best_iteration

    metrics = {
        "train": _report_metrics(
            y_train, booster.predict(X_train_lgb, num_iteration=best_iteration)
        ),
        "val": _report_metrics(
            y_val, booster.predict(X_val_lgb, num_iteration=best_iteration)
        ),
        "test": _report_metrics(
            y_test, booster.predict(X_test_lgb, num_iteration=best_iteration)
        ),
    }
    return booster, best_iteration, metrics, category_maps


def run_spw_ablation(splits: dict[str, pd.DataFrame], rounds: int | None = None) -> dict:
    """
    A/B test: scale_pos_weight ON vs OFF, evaluated on Val only (Test is not
    involved). Mirrors notebooks/analysis.ipynb Cell 25.

    The notebook's finding, reproduced here: spw does not improve ranking
    (Val AUC is essentially unchanged) but it roughly triples the mean
    predicted probability away from the actual Val default rate -- i.e. it
    makes the raw scores look like calibrated probabilities when they
    aren't, without helping the model separate goods from bads. That is why
    train_lgb()/train_and_save() default to use_spw=False.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        temporal_split() output; must contain "train", "val".
    rounds : int or None
        Fixed boosting-round count used for BOTH variants -- no early
        stopping here, since early stopping on two runs with different
        objectives would confound the comparison. If None, this first calls
        train_lgb(splits, use_spw=False) once to obtain an early-stopped
        best_iteration and reuses that count, exactly as the notebook does
        (Cell 23 establishes best_rounds; Cell 25 reuses it for the A/B).

    Returns
    -------
    dict
        {"with_spw": {"auc":..., "mean_pred":...},
         "without_spw": {"auc":..., "mean_pred":...},
         "actual_val_default_rate": float,
         "rounds": int}
    """
    if rounds is None:
        _, best_iteration, _, _ = train_lgb(splits, use_spw=False)
        rounds = best_iteration

    X_train, y_train = _xy(splits["train"])
    X_val, y_val = _xy(splits["val"])
    category_maps = _train_categories(X_train)
    X_train_lgb = _to_lgb_frame(X_train, category_maps)
    X_val_lgb = _to_lgb_frame(X_val, category_maps)

    dtrain_base = lgb.Dataset(X_train_lgb, y_train, categorical_feature=CATEGORICAL)

    results: dict = {}
    for use_spw, key, tag in [(True, "with_spw", "WITH spw"), (False, "without_spw", "WITHOUT spw")]:
        params = dict(LGB_PARAMS)
        if use_spw:
            params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()
        booster = lgb.train(params, dtrain_base, num_boost_round=rounds)
        p_val = booster.predict(X_val_lgb)
        results[key] = {"auc": float(roc_auc_score(y_val, p_val)), "mean_pred": float(p_val.mean())}
        print(f"{tag:<14} Val AUC={results[key]['auc']:.4f}  "
              f"mean_pred={results[key]['mean_pred']:.4f}  actual={y_val.mean():.4f}")

    results["actual_val_default_rate"] = float(y_val.mean())
    results["rounds"] = rounds
    return results


def train_and_save(
    splits: dict[str, pd.DataFrame],
    model_dir: str | Path | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> Path:
    """
    Train the LR baseline and the production LightGBM model (no
    scale_pos_weight), log each as its own MLflow run, then package the
    LightGBM model with its full feature contract into a single joblib
    artifact.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        temporal_split() output; must contain "train", "val", "test".
    model_dir : str, Path, or None
        Directory the packaged model is written to. Defaults to
        DEFAULT_MODEL_DIR (PROJECT_ROOT / "models"), created if missing.
    num_boost_round, early_stopping_rounds : int
        Forwarded to train_lgb().

    Returns
    -------
    Path
        Path to the saved .pkl file.
    """
    model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- MLflow run 1: the Logistic Regression baseline.
    #
    # Logging it as its own run (rather than just printing its numbers) means
    # it shows up as a first-class, comparable record next to the LightGBM
    # run below in `mlflow ui` -- the "does the complex model actually earn
    # its keep over the simple one" comparison becomes something you can
    # click through later, not a number that scrolled off the terminal.
    with mlflow.start_run(run_name="lr_baseline"):
        lr_pipeline, lr_metrics = train_baseline(splits)

        # log_param: records the CONFIGURATION that produced this run, so a
        # run_id found later in `mlflow ui` can be traced back to exactly
        # what was run, without needing to reconstruct it from git history.
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)

        # log_metric: records the OUTCOME of that configuration. Logging
        # both auc and pr_auc for all three splits means the full
        # train/val/test picture (including the generalization gap) is
        # queryable later, not just whatever single number got printed.
        for split_name, m in lr_metrics.items():
            mlflow.log_metric(f"{split_name}_auc", m["auc"])
            mlflow.log_metric(f"{split_name}_pr_auc", m["pr_auc"])

    # --- MLflow run 2: the production LightGBM model.
    with mlflow.start_run(run_name="lgbm_production"):
        booster, best_iteration, lgb_metrics, category_maps = train_lgb(
            splits,
            use_spw=False,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("learning_rate", LGB_PARAMS["learning_rate"])
        mlflow.log_param("num_leaves", LGB_PARAMS["num_leaves"])
        mlflow.log_param("seed", LGB_PARAMS["seed"])
        # best_iteration is itself a modeling decision (chosen by early
        # stopping on Val) -- logging it as a param, not just a metric,
        # records "which iteration IS the model" for reproducibility.
        mlflow.log_param("best_iteration", best_iteration)
        mlflow.log_param("use_scale_pos_weight", False)

        for split_name, m in lgb_metrics.items():
            mlflow.log_metric(f"{split_name}_auc", m["auc"])
            mlflow.log_metric(f"{split_name}_pr_auc", m["pr_auc"])

        # --- Packaging: dump a DICT, not the bare Booster.
        #
        # A bare LightGBM Booster only knows how to turn a matrix of numbers
        # into a prediction -- it has no memory of which column is which,
        # what order they must arrive in, or which string maps to which
        # category code. If serving code has to reconstruct that context
        # independently (e.g. re-deriving "the" categories from whatever
        # happens to be in a serving request, or re-typing the feature list
        # from a different source than the one training used), any drift
        # between training-time and serving-time feature engineering --  a
        # reordered column, a category encountered in a different order, a
        # renamed field -- produces silently WRONG predictions instead of a
        # loud error. This is classic train/serve skew, and by the time it's
        # noticed it's usually already a production incident.
        #
        # Bundling the model together with its feature contract (which
        # columns, in what role, which categories each categorical column
        # was trained on, which boosting round is "the model", and the
        # hyperparameters used) makes the artifact self-describing: whoever
        # loads this .pkl later has everything needed to reproduce the exact
        # inference path the model was trained and validated on, without
        # relying on an unwritten convention living only in this script.
        trained_at = datetime.now(timezone.utc).isoformat()
        artifact = {
            "model": booster,
            "features": FEATURES,
            "categorical": CATEGORICAL,
            "category_maps": category_maps,
            "best_iteration": best_iteration,
            "params": {**LGB_PARAMS, "scale_pos_weight": None},
            "trained_at": trained_at,
        }

        model_path = model_dir / "lgbm_model.pkl"
        joblib.dump(artifact, model_path)

        # log_artifact: MLflow copies the actual serialized file into this
        # run's artifact store, so the exact binary that produced the metrics
        # logged above stays retrievable from the MLflow UI/API alongside
        # them -- not just the numbers that describe it, but the model
        # itself.
        mlflow.log_artifact(str(model_path))

    return model_path
