import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, FIG_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from .features import build_preprocessor

CV = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"}


def candidate_models():
    return {
        "Baseline (średnia)": DummyRegressor(strategy="mean"),
        "Regresja liniowa": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Las losowy": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Sieć neuronowa (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000,
                                             early_stopping=True, random_state=RANDOM_STATE),
    }


SEARCH_SPACES = {
    "Ridge": {"model__alpha": np.logspace(-3, 3, 30)},
    "Las losowy": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 6, 8, 10, 14],
        "model__min_samples_leaf": [1, 3, 5, 10, 20],
        "model__max_features": ["sqrt", 0.33, 0.5, 1.0],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__min_samples_leaf": [5, 10, 20, 40],
    },
    "Sieć neuronowa (MLP)": {
        "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
        "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        "model__learning_rate_init": [1e-3, 3e-3, 1e-2],
    },
}
N_ITER = 15


def make_pipe(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": float(np.sqrt(mse))}


def cv_summary(pipe, X, y) -> dict:
    res = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    return {"CV_R2": res["test_r2"].mean(), "CV_R2_std": res["test_r2"].std(),
            "CV_MAE": -res["test_mae"].mean(), "CV_RMSE": float(np.sqrt(-res["test_mse"].mean()))}


def run(X_train, X_test, y_train, y_test) -> dict:
    rows, fitted = [], {}

    for name, model in candidate_models().items():
        t0 = time.time()
        pipe = make_pipe(model)
        cvs = cv_summary(pipe, X_train, y_train)
        pipe.fit(X_train, y_train)
        test = metrics(y_test, pipe.predict(X_test))
        rows.append({"Model": name, "Wariant": "domyślny", **cvs,
                     **{f"Test_{k}": v for k, v in test.items()}, "Czas_s": time.time() - t0})
        fitted[(name, "domyślny")] = pipe
        print(f"[base]  {name:24s} CV R2={cvs['CV_R2']:.3f}  test R2={test['R2']:.3f}")

    best_params = {}
    for name, space in SEARCH_SPACES.items():
        t0 = time.time()
        base = candidate_models()[name]
        n_iter = min(N_ITER, int(np.prod([len(v) for v in space.values()])))
        search = RandomizedSearchCV(make_pipe(base), space, n_iter=n_iter, cv=CV,
                                    scoring="neg_mean_squared_error", random_state=RANDOM_STATE,
                                    n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        cvs = cv_summary(clone(pipe), X_train, y_train)
        test = metrics(y_test, pipe.predict(X_test))
        best_params[name] = {k.replace("model__", ""): (v if not isinstance(v, np.generic) else v.item())
                             for k, v in search.best_params_.items()}
        rows.append({"Model": name, "Wariant": "tuning", **cvs,
                     **{f"Test_{k}": v for k, v in test.items()}, "Czas_s": time.time() - t0})
        fitted[(name, "tuning")] = pipe
        print(f"[tuned] {name:24s} CV R2={cvs['CV_R2']:.3f}  test R2={test['R2']:.3f}  {best_params[name]}")

    results = pd.DataFrame(rows)
    results.round(4).to_csv(REPORTS_DIR / "model_results.csv", index=False)

    # Wybór modelu wyłącznie na podstawie CV - zbiór testowy służy tylko do końcowej oceny
    best_row = results.loc[results["CV_RMSE"].idxmin()]
    best_key = (best_row["Model"], best_row["Wariant"])
    best = fitted[best_key]
    joblib.dump(best, MODELS_DIR / "best_model.joblib")
    y_pred = best.predict(X_test)
    print(f"Najlepszy model: {best_key}")

    ablation = _ablation_without_education(best, X_train, X_test, y_train, y_test)

    _plots(results, best, best_key, X_test, y_test, y_pred)

    summary = {
        "best_model": best_key[0], "best_variant": best_key[1],
        "best_params": best_params.get(best_key[0]) if best_key[1] == "tuning" else "domyślne",
        "test_metrics": metrics(y_test, y_pred),
        "cv_metrics": {k: float(best_row[k]) for k in ["CV_R2", "CV_R2_std", "CV_MAE", "CV_RMSE"]},
        "all_best_params": best_params,
        "ablation_without_education": ablation,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    return summary


def _ablation_without_education(best, X_train, X_test, y_train, y_test):
    # education = lata nauki ukończone PO teście; sprawdzamy, ile trafności od niej zależy
    from . import features as F
    drop = ["education", "edu_beyond_hs"]
    old = F.NUM_FEATURES
    F.NUM_FEATURES = [c for c in old if c not in drop]
    try:
        pipe = Pipeline([("prep", F.build_preprocessor()), ("model", clone(best.named_steps["model"]))])
        pipe.fit(X_train.drop(columns=drop), y_train)
        return metrics(y_test, pipe.predict(X_test.drop(columns=drop)))
    finally:
        F.NUM_FEATURES = old


def _plots(results, best, best_key, X_test, y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_df = results.copy()
    for ax, metric, title in [(axes[0], "Test_R2", "R² na zbiorze testowym (więcej = lepiej)"),
                              (axes[1], "Test_RMSE", "RMSE na zbiorze testowym (mniej = lepiej)")]:
        sns.barplot(data=plot_df, y="Model", x=metric, hue="Wariant", ax=ax, orient="h")
        ax.set_title(title)
        ax.set_ylabel("")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_model_comparison.png", dpi=130)
    plt.close(fig)

    resid = y_test - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(y_test, y_pred, s=8, alpha=0.4)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set(xlabel="score rzeczywisty", ylabel="score przewidziany",
                title=f"{best_key[0]} ({best_key[1]}): predykcja vs rzeczywistość")
    axes[1].scatter(y_pred, resid, s=8, alpha=0.4)
    axes[1].axhline(0, color="r", ls="--", lw=1)
    axes[1].set(xlabel="score przewidziany", ylabel="reszta", title="Reszty vs predykcja")
    sns.histplot(resid, kde=True, ax=axes[2])
    axes[2].set_title("Rozkład reszt")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_best_model_diagnostics.png", dpi=130)
    plt.close(fig)

    pi = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE,
                                scoring="r2", n_jobs=-1)
    imp = pd.DataFrame({"feature": X_test.columns, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp.round(4).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp, x="importance", y="feature", ax=ax, color="#3b6ea5")
    ax.set_title("Permutation importance (spadek R² po przetasowaniu cechy)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_feature_importance.png", dpi=130)
    plt.close(fig)
