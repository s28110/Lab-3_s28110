import json
import urllib.request

import numpy as np
import pandas as pd

from .config import BINARY_COLS, DATA_PATH, DATA_URL, NUMERIC_COLS, NOMINAL_COLS, REPORTS_DIR, TARGET


def download(force: bool = False) -> None:
    if DATA_PATH.exists() and not force:
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pobieranie {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def load_raw() -> pd.DataFrame:
    download()
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["rownames"], errors="ignore")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"rows_raw": int(len(df))}

    for c in BINARY_COLS + NOMINAL_COLS:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})

    report["missing_per_column"] = {k: int(v) for k, v in df.isna().sum().items()}

    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    report["dropped_missing_target"] = int(n_before - len(df))

    for c in NUMERIC_COLS:
        df[c] = df[c].fillna(df[c].median())
    for c in BINARY_COLS + NOMINAL_COLS:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    dup = int(df.duplicated().sum())
    report["duplicates_removed"] = dup
    df = df.drop_duplicates().reset_index(drop=True)

    # Outliery (1.5*IQR) tylko raportujemy - to realne obserwacje, nie błędy
    outliers = {}
    for c in NUMERIC_COLS + [TARGET]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    report["outliers_iqr"] = outliers
    report["rows_clean"] = int(len(df))

    (REPORTS_DIR / "data_cleaning.json").write_text(json.dumps(report, indent=2))
    return df, report
