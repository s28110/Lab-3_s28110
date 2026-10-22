import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import BINARY_COLS, BINARY_MAP, FIG_DIR, NOMINAL_COLS, NUMERIC_COLS, REPORTS_DIR, TARGET

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    cat_cols = BINARY_COLS + NOMINAL_COLS

    num_stats = df[NUMERIC_COLS + [TARGET]].describe().T
    num_stats["skew"] = df[NUMERIC_COLS + [TARGET]].skew()
    num_stats.round(3).to_csv(REPORTS_DIR / "stats_numeric.csv")

    cat_rows = []
    for c in cat_cols:
        for level, cnt in df[c].value_counts().items():
            cat_rows.append({
                "variable": c, "level": level, "count": int(cnt),
                "share_%": round(100 * cnt / len(df), 1),
                "mean_score": round(df.loc[df[c] == level, TARGET].mean(), 2),
            })
    cat_stats = pd.DataFrame(cat_rows)
    cat_stats.to_csv(REPORTS_DIR / "stats_categorical.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], kde=True, ax=ax[0], color="#3b6ea5")
    ax[0].set_title("Rozkład zmiennej score")
    sns.boxplot(x=df[TARGET], ax=ax[1], color="#3b6ea5")
    ax[1].set_title("Boxplot score")
    _save(fig, "01_target_distribution.png")

    fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(16, 3.4))
    for ax, c in zip(axes, NUMERIC_COLS):
        sns.histplot(df[c], kde=True, ax=ax, color="#5a9e6f")
        ax.set_title(c)
    _save(fig, "02_numeric_distributions.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.flat, cat_cols):
        sns.boxplot(data=df, x=c, y=TARGET, ax=ax, hue=c, legend=False)
        ax.set_title(f"score wg {c}")
        ax.set_xlabel("")
    _save(fig, "03_score_by_category.png")

    fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(16, 3.4))
    for ax, c in zip(axes, NUMERIC_COLS):
        sns.regplot(data=df, x=c, y=TARGET, ax=ax, scatter_kws={"s": 5, "alpha": 0.25},
                    line_kws={"color": "#c44e52"})
        ax.set_title(f"score vs {c}")
    _save(fig, "04_score_vs_numeric.png")

    enc = df.copy()
    for c, m in BINARY_MAP.items():
        enc[c] = enc[c].map(m)
    enc = pd.get_dummies(enc, columns=NOMINAL_COLS, prefix="eth", dtype=float)
    corr = enc.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Macierz korelacji (Pearson)")
    _save(fig, "05_correlation_matrix.png")

    corr_target = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    corr_target.round(3).to_csv(REPORTS_DIR / "corr_with_target.csv", header=["corr"])

    return {"corr_with_target": corr_target.round(3).to_dict()}
