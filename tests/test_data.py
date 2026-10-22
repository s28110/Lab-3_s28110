import pandas as pd
import pytest

from src import data, features
from src.config import BINARY_MAP, NOMINAL_COLS, NUMERIC_COLS, TARGET


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


def test_schema(raw):
    expected = set(NUMERIC_COLS + list(BINARY_MAP) + NOMINAL_COLS + [TARGET])
    assert expected <= set(raw.columns), f"Brakujące kolumny: {expected - set(raw.columns)}"


def test_enough_rows(raw):
    assert len(raw) > 4000


def test_target_range(raw):
    assert raw[TARGET].between(0, 100).all()


def test_categories_known(raw):
    for col, mapping in BINARY_MAP.items():
        unknown = set(raw[col].dropna().str.lower()) - set(mapping)
        assert not unknown, f"{col}: nieznane kategorie {unknown}"


def test_clean_has_no_missing(raw):
    df, report = data.clean(raw.copy())
    assert df.isna().sum().sum() == 0
    assert report["rows_clean"] > 0


def test_imputation_handles_missing(raw):
    df = raw.copy()
    df.loc[df.index[:20], "wage"] = None
    df.loc[df.index[20:40], "gender"] = None
    df.loc[df.index[40:45], TARGET] = None
    clean, report = data.clean(df)
    assert clean.isna().sum().sum() == 0
    assert report["dropped_missing_target"] == 5


def test_split_no_overlap(raw):
    df, _ = data.clean(raw.copy())
    X_tr, X_te, y_tr, y_te = features.split(df)
    assert len(set(X_tr.index) & set(X_te.index)) == 0
    assert abs(y_tr.mean() - y_te.mean()) < 1.0
    assert TARGET not in X_tr.columns
