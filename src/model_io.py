"""
Model I/O for the LendingClub granting-model dataset: feature encoding for
inference, LightGBM training, and packaged-artifact loading -- the half of
what used to be src/train.py that every consumer (calibrate / evaluate /
fairness / explain / serving) actually needs, and that does not import
mlflow.

Feature contract comes from src/features.py (FEATURES/NUMERIC/CATEGORICAL/
TARGET/add_features) -- this module does not redeclare it, so the notebook
and this module can never quietly drift apart on "what the model's inputs
are". Splits come from src/data_loader.py's temporal_split() output.

src/train.py imports train_baseline/train_lgb/LGB_PARAMS back from here for
train_and_save(), which packages a trained model, logs it to MLflow, and is
the one thing this module does not do -- see its docstring for the MLflow
experiment-tracking setup.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
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


def _x(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering and return X alone -- no target required.

    _xy() indexes engineered[TARGET], so it raises KeyError on any frame
    without a `Default` column. Every split-frame has one; a live serving
    request does not. This is the same feature-engineering path with the
    target extraction removed, so a caller that has no label (src/explain.py)
    encodes through THIS function rather than writing its own
    add_features(df)[FEATURES] -- a second, independently-written encoding
    path is exactly the train/serve skew calibrate.py's module docstring
    warns about.

    Pair with _to_lgb_frame() to get a frame the model can score: this
    function derives and selects columns, that one casts categoricals to the
    train-derived category maps.
    """
    return add_features(df)[FEATURES]


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
        # Booster.predict() is typed to cover every mode it has (raw_score,
        # pred_leaf, pred_contrib, sparse input), so its return is a union that
        # includes list. Under these params -- objective="binary" (LGB_PARAMS),
        # a dense frame, and none of those flags -- it is always an ndarray, so
        # .mean() is real. Nothing here can narrow the stub's union, and a cast
        # would restate this paragraph without checking it.
        results[key] = {
            "auc": float(roc_auc_score(y_val, p_val)),
            "mean_pred": float(p_val.mean()),  # type: ignore[union-attr]
        }
        print(f"{tag:<14} Val AUC={results[key]['auc']:.4f}  "
              f"mean_pred={results[key]['mean_pred']:.4f}  actual={y_val.mean():.4f}")

    results["actual_val_default_rate"] = float(y_val.mean())
    results["rounds"] = rounds
    return results


def load_model_artifact(model_path: str | Path) -> dict:
    """
    Load a packaged model dict AND enforce that the feature contract it was
    trained under still matches features.py's live FEATURES / CATEGORICAL --
    the exact globals the inference path (_xy / _to_lgb_frame) will use to
    encode whatever gets scored next.

    train_and_save() bundles that contract INTO the artifact precisely so a
    later scoring run cannot silently use a different feature set than the
    model was trained on (features.py's INCLUDE_ADDR_STATE flipped, or the
    feature list edited, between train and serve). But bundling it was only
    half the job -- nothing consumed it. This is the other half: fail-closed
    enforcement, the same discipline as the schema / leakage gates. A contract
    mismatch STOPS the run rather than letting _to_lgb_frame encode against a
    misaligned column set and hand LightGBM a frame that does not match what it
    was trained on -- a KeyError at best, silently wrong predictions at worst.

    Every packaged-model consumer (calibrate / evaluate / fairness / drift)
    loads through here instead of a bare joblib.load, so the contract is
    verified everywhere a shipped model re-enters inference.

    Raises
    ------
    ValueError
        If the artifact's stored features/categorical differ from the current
        features.py FEATURES/CATEGORICAL.
    """
    artifact = joblib.load(model_path)
    saved_features = artifact.get("features")
    saved_categorical = artifact.get("categorical")
    if saved_features != FEATURES or saved_categorical != CATEGORICAL:
        raise ValueError(
            "Model/feature-contract mismatch -- refusing to score.\n"
            f"  packaged model was trained on:\n"
            f"    features    = {saved_features}\n"
            f"    categorical = {saved_categorical}\n"
            f"  features.py now declares:\n"
            f"    FEATURES    = {FEATURES}\n"
            f"    CATEGORICAL = {CATEGORICAL}\n"
            "Inference encodes with the live features.py globals (_xy / "
            "_to_lgb_frame), so scoring this model now would use a different "
            "feature set than it was trained on. Retrain (train_and_save) or "
            "restore the matching features.py configuration (e.g. "
            "INCLUDE_ADDR_STATE) before scoring."
        )
    return artifact
