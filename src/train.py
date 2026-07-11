"""
MLflow orchestration and model packaging for the LendingClub granting-model
LightGBM model.

train_and_save() trains the LR baseline and the production LightGBM model
(via src/model_io.py's train_baseline/train_lgb, no scale_pos_weight -- see
model_io.py's run_spw_ablation() docstring for the A/B test that justified
NOT using it), logs each as its own MLflow run, then packages the LightGBM
model with its full feature contract into a single joblib artifact.

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

The feature-encoding helpers, the LightGBM training loop, and
load_model_artifact() live in src/model_io.py, not here: every consumer that
only needs to score or re-validate an already-trained model (calibrate /
evaluate / fairness / explain / serving) imports from there instead, so that
importing this module -- and therefore mlflow -- is reserved for whoever is
actually about to train.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from .features import CATEGORICAL, FEATURES
from .model_io import DEFAULT_MODEL_DIR, LGB_PARAMS, PROJECT_ROOT, train_baseline, train_lgb

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
