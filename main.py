import argparse
import json

from src import data, eda, features, train
from src.config import REPORTS_DIR


def main():
    p = argparse.ArgumentParser(description="Lab3 - Analizator wyników (predykcja score)")
    p.add_argument("--download", action="store_true", help="wymuś ponowne pobranie datasetu")
    p.add_argument("--no-report", action="store_true", help="pomiń generowanie Readme.pdf")
    args = p.parse_args()

    if args.download:
        data.download(force=True)

    print("== 1. Wczytanie i czyszczenie danych ==")
    df, clean_report = data.clean(data.load_raw())
    print(json.dumps(clean_report, indent=2))

    print("== 2. Eksploracja danych ==")
    eda_summary = eda.run(df)

    print("== 3. Inżynieria cech i podział train/test ==")
    X_train, X_test, y_train, y_test = features.split(df)
    print(f"train={len(X_train)}  test={len(X_test)}")

    print("== 4-5. Trening, walidacja krzyżowa, tuning, ocena ==")
    summary = train.run(X_train, X_test, y_train, y_test)
    summary["eda"] = eda_summary
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))

    if not args.no_report:
        from src import report
        report.build()
        report.update_readme()

    print(f"\nNajlepszy model: {summary['best_model']} ({summary['best_variant']})")
    print(json.dumps(summary["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
