"""
Tests for src/data_loader.py

Two things are locked down here:
  1. The four-way temporal split (Train/Val/Calib/Test/2018 holdout) never
     leaks rows across boundaries and is reproducible under a fixed seed --
     this is the numeric backbone the whole project depends on.
  2. load_raw()'s validate=True gate actually fires (and validate=False
     actually bypasses it) -- proving the data_validation wiring is live,
     not decoration.

All split tests run on a small synthetic DataFrame (never the real CSV), so
they stay fast and don't depend on data/ being populated. 2015 is sized at
2*N_VAL rows so Val and Calib can be checked for their exact, disjoint size.

Run:  pytest tests/test_splits.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors as pa_errors
import pytest

from src.data_loader import (
    HOLDOUT_YEAR_2018,
    N_VAL,
    TEST_YEAR_HI,
    TEST_YEAR_LO,
    TRAIN_YEAR_HI,
    TRAIN_YEAR_LO,
    VAL_CALIB_YEAR,
    load_raw,
    temporal_split,
)


def _build_synthetic_df(n_2015: int = 2 * N_VAL, seed: int = 123) -> pd.DataFrame:
    """A small frame spanning every split boundary, with unique ids throughout."""
    train_years = list(range(TRAIN_YEAR_LO, TRAIN_YEAR_HI + 1))
    test_years = list(range(TEST_YEAR_LO, TEST_YEAR_HI + 1))

    years = (
        [y for y in train_years for _ in range(50)]
        + [VAL_CALIB_YEAR] * n_2015
        + [y for y in test_years for _ in range(60)]
        + [HOLDOUT_YEAR_2018] * 70
    )
    n = len(years)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n),
        "issue_year": years,
        "Default": rng.integers(0, 2, size=n),
    })


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return _build_synthetic_df()


# ---------------------------------------------------------------------------
# 1. No row overlap between any pair of splits.
# ---------------------------------------------------------------------------
def test_no_row_overlap_across_splits(synthetic_df):
    splits = temporal_split(synthetic_df.copy())
    id_sets = {name: set(d["id"]) for name, d in splits.items()}
    names = list(id_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = id_sets[names[i]] & id_sets[names[j]]
            assert not overlap, f"{names[i]} and {names[j]} share ids: {overlap}"


# ---------------------------------------------------------------------------
# 2. Year boundaries are respected by every split.
# ---------------------------------------------------------------------------
def test_year_boundaries_correct(synthetic_df):
    splits = temporal_split(synthetic_df.copy())

    train_years = splits["train"]["issue_year"]
    assert train_years.between(TRAIN_YEAR_LO, TRAIN_YEAR_HI).all()

    assert splits["val"]["issue_year"].eq(VAL_CALIB_YEAR).all()
    assert splits["calib"]["issue_year"].eq(VAL_CALIB_YEAR).all()

    test_years = splits["test"]["issue_year"]
    assert test_years.between(TEST_YEAR_LO, TEST_YEAR_HI).all()

    assert splits["holdout_2018"]["issue_year"].eq(HOLDOUT_YEAR_2018).all()


# ---------------------------------------------------------------------------
# 3. Val and Calib are each exactly N_VAL rows, and disjoint.
# ---------------------------------------------------------------------------
def test_val_calib_are_40000_and_disjoint(synthetic_df):
    splits = temporal_split(synthetic_df.copy())
    assert len(splits["val"]) == N_VAL
    assert len(splits["calib"]) == N_VAL
    assert set(splits["val"]["id"]).isdisjoint(set(splits["calib"]["id"]))


# ---------------------------------------------------------------------------
# 4. Reproducible: same seed, same Val ids, every time.
# ---------------------------------------------------------------------------
def test_reproducible_with_same_seed(synthetic_df):
    splits_a = temporal_split(synthetic_df.copy())
    splits_b = temporal_split(synthetic_df.copy())
    assert set(splits_a["val"]["id"]) == set(splits_b["val"]["id"])


# ---------------------------------------------------------------------------
# 5. load_raw()'s validation gate: fires when validate=True, bypassed when
#    validate=False. Proves the data_validation wiring is live, not decoration.
# ---------------------------------------------------------------------------
def _write_invalid_csv(tmp_path) -> str:
    """One row with fico_n=1200 -- out of LOAN_SCHEMA's [300, 900] band."""
    bad = pd.DataFrame({
        "id": [1],
        "issue_d": ["Jan-2016"],
        "revenue": [50000.0],
        "dti_n": [12.0],
        "loan_amnt": [5000.0],
        "fico_n": [1200.0],
        "home_ownership_n": ["MORTGAGE"],
        "purpose": ["debt_consolidation"],
        "emp_length": ["10+ years"],
        "addr_state": ["PA"],
        "Default": [0],
    })
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    return str(path)


def test_validate_gate_fires_on_bad_schema(tmp_path):
    path = _write_invalid_csv(tmp_path)
    with pytest.raises(pa_errors.SchemaErrors):
        load_raw(data_path=path, validate=True)


def test_validate_false_bypasses_gate(tmp_path):
    path = _write_invalid_csv(tmp_path)
    df = load_raw(data_path=path, validate=False)
    assert len(df) == 1
    assert df.loc[0, "fico_n"] == 1200.0
