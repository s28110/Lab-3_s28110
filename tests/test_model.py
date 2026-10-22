import json

import joblib
import pandas as pd
import pytest

from src import data, features
from src.config import MODELS_DIR, REPORTS_DIR

MIN_R2 = 0.25


@pytest.fixture(scope="module")
def results():
    return pd.read_csv(REPORTS_DIR / "model_results.csv")


@pytest.fixture(scope="module")
def metrics():
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


def test_best_model_r2_above_threshold(metrics):
    assert metrics["test_metrics"]["R2"] >= MIN_R2


def test_beats_baseline(results, metrics):
    base_mae = results.loc[results.Model.str.startswith("Baseline"), "Test_MAE"].iloc[0]
    assert metrics["test_metrics"]["MAE"] < base_mae


def test_at_least_three_algorithms(results):
    assert results.loc[~results.Model.str.startswith("Baseline"), "Model"].nunique() >= 3


def test_no_overfitting(metrics):
    assert abs(metrics["cv_metrics"]["CV_R2"] - metrics["test_metrics"]["R2"]) < 0.08


def test_saved_model_predicts():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    df, _ = data.clean(data.load_raw())
    X = features.add_features(df).drop(columns=["score"]).head(10)
    pred = model.predict(X)
    assert len(pred) == 10 and pred.min() > 20 and pred.max() < 80


def test_artifacts_exist():
    from src.config import FIG_DIR, ROOT
    assert len(list(FIG_DIR.glob("*.png"))) >= 8
    assert (ROOT / "Readme.pdf").exists()
