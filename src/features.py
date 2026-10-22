import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_COLS, BINARY_MAP, NOMINAL_COLS, RANDOM_STATE, TARGET, TEST_SIZE

ENGINEERED = ["parents_college", "log_distance", "wage_unemp_ratio", "edu_beyond_hs"]

NUM_FEATURES = ["unemp", "wage", "distance", "tuition", "education"] + ENGINEERED


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c, m in BINARY_MAP.items():
        df[c] = df[c].map(m).astype(float)

    df["parents_college"] = df["fcollege"] + df["mcollege"]
    df["log_distance"] = np.log1p(df["distance"])
    df["wage_unemp_ratio"] = df["wage"] / (df["unemp"] + 1e-6)
    df["edu_beyond_hs"] = (df["education"] - 12).clip(lower=0)
    return df


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binr = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
    nom = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num, NUM_FEATURES), ("bin", binr, BINARY_COLS), ("nom", nom, NOMINAL_COLS)],
        verbose_feature_names_out=False,
    )


def split(df: pd.DataFrame):
    df = add_features(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    # Stratyfikacja po kwintylach score -> podobny rozkład celu w train i test
    bins = pd.qcut(y, q=5, labels=False)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=bins)
