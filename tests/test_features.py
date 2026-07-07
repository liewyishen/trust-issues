"""
Tests for src/features.py

Locks down notebook Cell 17's feature engineering: the emp_length ordinal
map, the explicit missingness flag, and the feature/target contract every
downstream module imports. All tests use small synthetic DataFrames, never
the real CSV.

Run:  pytest tests/test_features.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    CATEGORICAL,
    CATEGORICAL_BASE,
    FEATURES,
    INCLUDE_ADDR_STATE,
    NUMERIC,
    TARGET,
    add_features,
    build_categorical,
    emp_order,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "emp_length": ["< 1 year", "5 years", "10+ years", "NI", "3 years"],
        "revenue": [40000.0, 60000.0, 90000.0, 55000.0, 70000.0],
    })


# ---------------------------------------------------------------------------
# 1. Known emp_length strings map to the right ordinal.
# ---------------------------------------------------------------------------
def test_known_emp_length_values_map_correctly(raw_df):
    out = add_features(raw_df)
    mapped = dict(zip(out["emp_length"], out["emp_length_ord"]))
    assert mapped["< 1 year"] == 0
    assert mapped["5 years"] == 5
    assert mapped["10+ years"] == 10


# ---------------------------------------------------------------------------
# 2. "NI" -> NaN ordinal, missing flag = 1.
# ---------------------------------------------------------------------------
def test_ni_maps_to_nan_ordinal_and_flags_missing(raw_df):
    out = add_features(raw_df)
    ni_rows = out[out["emp_length"] == "NI"]
    assert ni_rows["emp_length_ord"].isna().all()
    assert (ni_rows["emp_length_missing"] == 1).all()


# ---------------------------------------------------------------------------
# 3. Normal (non-"NI") tenure -> missing flag = 0.
# ---------------------------------------------------------------------------
def test_normal_tenure_is_not_flagged_missing(raw_df):
    out = add_features(raw_df)
    non_ni = out[out["emp_length"] != "NI"]
    assert (non_ni["emp_length_missing"] == 0).all()


# ---------------------------------------------------------------------------
# 4. add_features does not mutate its input.
# ---------------------------------------------------------------------------
def test_add_features_does_not_mutate_input(raw_df):
    before_cols = set(raw_df.columns)
    add_features(raw_df)
    assert set(raw_df.columns) == before_cols
    assert "emp_length_ord" not in raw_df.columns
    assert "emp_length_missing" not in raw_df.columns


# ---------------------------------------------------------------------------
# 5. Feature contract: exactly 8, no experience_c, addr_state EXCLUDED by
# default (INCLUDE_ADDR_STATE=False -- the fairness-audit conclusion,
# see docs/data-decisions.md).
# ---------------------------------------------------------------------------
def test_features_contract():
    assert INCLUDE_ADDR_STATE is False
    assert len(FEATURES) == 8
    assert "experience_c" not in FEATURES
    assert "addr_state" not in FEATURES
    assert "addr_state" not in CATEGORICAL
    assert FEATURES == NUMERIC + CATEGORICAL
    assert TARGET == "Default"
    assert emp_order["10+ years"] == 10


# ---------------------------------------------------------------------------
# 7. INCLUDE_ADDR_STATE toggle: build_categorical() reproduces both the
# current default (False -> excluded) and the with-state variant (True ->
# included) -- the fairness-audit ablation's reproducibility depends on
# this still working even though production defaults to no-state.
# ---------------------------------------------------------------------------
def test_build_categorical_default_matches_current_toggle():
    assert build_categorical() == CATEGORICAL
    assert "addr_state" not in build_categorical()


def test_build_categorical_true_includes_addr_state():
    with_state = build_categorical(True)
    assert "addr_state" in with_state
    assert with_state == CATEGORICAL_BASE + ["addr_state"]


def test_build_categorical_false_excludes_addr_state():
    no_state = build_categorical(False)
    assert "addr_state" not in no_state
    assert no_state == CATEGORICAL_BASE


# ---------------------------------------------------------------------------
# 6. add_features output includes the two derived columns.
# ---------------------------------------------------------------------------
def test_add_features_adds_expected_columns(raw_df):
    out = add_features(raw_df)
    assert "emp_length_ord" in out.columns
    assert "emp_length_missing" in out.columns
