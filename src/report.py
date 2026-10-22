import json
import re
from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from .config import CV_FOLDS, FIG_DIR, REPORTS_DIR, ROOT, TEST_SIZE

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="DejaVu", fontSize=9.5, leading=13.5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DejaVu-Bold", fontSize=15, spaceBefore=6,
                    textColor=colors.HexColor("#1f3b5c"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DejaVu-Bold", fontSize=11.5,
                    textColor=colors.HexColor("#1f3b5c"))
TITLE = ParagraphStyle("title", parent=H1, fontSize=22, alignment=TA_CENTER, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=BODY, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))

W = A4[0] - 3.6 * cm


def _p(text, style=BODY):
    return Paragraph(text, style)


def _bullets(items):
    return [_p(f"• {t}") for t in items]


def _table(df: pd.DataFrame, col_widths=None, font=7.5):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c0c8d2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _fig(name, caption, width=W):
    path = FIG_DIR / name
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), _p(caption, SMALL), Spacer(1, 8)])


def _load():
    s = json.loads((REPORTS_DIR / "summary.json").read_text())
    clean = json.loads((REPORTS_DIR / "data_cleaning.json").read_text())
    res = pd.read_csv(REPORTS_DIR / "model_results.csv")
    return s, clean, res


def _results_table(res):
    t = res[["Model", "Wariant", "CV_R2", "CV_RMSE", "Test_R2", "Test_MAE", "Test_MSE", "Test_RMSE"]].copy()
    for c in t.columns[2:]:
        t[c] = t[c].map(lambda v: f"{v:.3f}")
    return t


def build(out: Path = ROOT / "Readme.pdf"):
    s, clean, res = _load()
    tm, cvm, abl = s["test_metrics"], s["cv_metrics"], s["ablation_without_education"]
    num = pd.read_csv(REPORTS_DIR / "stats_numeric.csv", index_col=0)
    imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    corr = s["eda"]["corr_with_target"]
    top_corr = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:4]
    o = clean["outliers_iqr"]

    st = [_p("Lab3 – Analizator wyników", TITLE),
          _p(f"Predykcja zmiennej <b>score</b> · zbiór CollegeDistance (AER) · {date.today().isoformat()}", SUB),
          Spacer(1, 10),
          _table(pd.DataFrame([
              ["Najlepszy model", f"{s['best_model']} ({s['best_variant']})"],
              ["Test: R² / MAE / MSE / RMSE",
               f"{tm['R2']:.3f} / {tm['MAE']:.2f} / {tm['MSE']:.2f} / {tm['RMSE']:.2f}"],
              [f"R² ({CV_FOLDS}-fold CV)", f"{cvm['CV_R2']:.3f} ± {cvm['CV_R2_std']:.3f}"],
              ["Train / test", f"{s['n_train']} / {s['n_test']}"],
          ], columns=["Pozycja", "Wartość"]), col_widths=[6 * cm, 8 * cm], font=9),
          Spacer(1, 6)]

    st += [_p("1. Eksploracja danych", H1),
           _p(f"{clean['rows_raw']} obserwacji, 13 cech: 5 liczbowych (unemp, wage, distance, tuition, "
              "education), 7 binarnych (gender, fcollege, mcollege, home, urban, income, region) i ethnicity "
              "(3 kategorie). Cel <b>score</b> to wynik testu kompetencji w ostatniej klasie szkoły średniej "
              f"(średnia {num.loc['score', 'mean']:.1f}, odch. std. {num.loc['score', 'std']:.1f}, "
              "rozkład zbliżony do normalnego)."),
           _p(f"Braki: <b>{sum(clean['missing_per_column'].values())}</b>, duplikaty: "
              f"<b>{clean['duplicates_removed']}</b>. Proces i tak obsługuje braki (usunięcie wierszy bez score, "
              "imputacja medianą/modą w pipeline). Outliery wg 1,5·IQR (unemp "
              f"{o['unemp']}, wage {o['wage']}, distance {o['distance']}) pozostawiono – to realne obserwacje."),
           _fig("01_target_distribution.png", "Rys. 1. Rozkład score.", width=W * 0.75),
           _fig("03_score_by_category.png",
                "Rys. 2. score wg kategorii – największe różnice: etniczność, wykształcenie rodziców, dochód."),
           _fig("05_correlation_matrix.png", "Rys. 3. Korelacje. Najsilniejsze ze score: " +
                ", ".join(f"{k} ({v:+.2f})" for k, v in top_corr) + ".", width=W * 0.7)]

    st += [_p("2. Inżynieria cech", H1), *_bullets([
        "zmienne binarne → 0/1, ethnicity → one-hot (drop='first'),",
        "nowe cechy: parents_college (fcollege + mcollege), log_distance (prawoskośność), "
        "wage_unemp_ratio (rynek pracy), edu_beyond_hs (lata nauki po szkole średniej),",
        "standaryzacja zmiennych liczbowych (wymagana przez Ridge i MLP), imputacja w pipeline,",
        f"podział {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} ze stratyfikacją po kwintylach score; "
        "transformacje dopasowywane tylko na train (brak wycieku danych).",
    ])]

    st += [_p("3. Modele", H1),
           _p("Baseline (średnia), regresja liniowa, Ridge (L2 – stabilna przy skorelowanych cechach), las "
              "losowy i Gradient Boosting (nieliniowości, interakcje) oraz sieć neuronowa MLP. Każdy oceniony "
              f"{CV_FOLDS}-fold CV na train, potem na teście.")]

    base_rf = res[(res.Model == "Las losowy") & (res.Wariant == "domyślny")].iloc[0]
    tuned_rf = res[(res.Model == "Las losowy") & (res.Wariant == "tuning")].iloc[0]
    base_lr = res[res.Model == "Regresja liniowa"].iloc[0]
    st += [_p("4. Ocena i optymalizacja", H1),
           _p("Tuning: RandomizedSearchCV (5-fold, kryterium MSE). Model finalny wybrano po wyniku CV, "
              "test służy tylko do końcowej oceny."),
           KeepTogether([_table(_results_table(res), font=7.3), Spacer(1, 6)]),
           _fig("06_model_comparison.png", "Rys. 4. Porównanie modeli na zbiorze testowym.", width=W * 0.75),
           _p(f"<b>{s['best_model']}</b> wyjaśnia ok. {tm['R2'] * 100:.0f}% wariancji (MAE {tm['MAE']:.2f} pkt). "
              f"CV ≈ test, więc brak przeuczenia. Tuning najbardziej pomógł lasowi losowemu "
              f"(R² {base_rf.Test_R2:.3f} → {tuned_rf.Test_R2:.3f}). Przewaga nad regresją liniową "
              f"(R² {base_lr.Test_R2:.3f}) jest niewielka – zależności są w dużej mierze addytywne."),
           _p(f"<b>Uwaga:</b> education to lata nauki ukończone po teście. Bez niej model osiąga "
              f"R² = {abl['R2']:.3f}. Umiarkowane R² wynika z braku w zbiorze kluczowych zmiennych "
              "(np. wcześniejszych ocen), a nie z wyboru modelu."),
           _fig("07_best_model_diagnostics.png", "Rys. 5. Diagnostyka najlepszego modelu.", width=W * 0.7),
           _fig("08_feature_importance.png", "Rys. 6. Permutation importance – dominują: " +
                ", ".join(imp.feature.head(3)) + ".", width=W * 0.45)]

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm, title="Lab3 – Analizator wyników")

    def footer(canvas, d):
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 1.8 * cm, 0.8 * cm, f"str. {d.page}")

    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print(f"Zapisano {out}")


def update_readme(path: Path = ROOT / "README.md"):
    if not path.exists():
        return
    s, _, res = _load()
    tm = s["test_metrics"]
    t = _results_table(res)
    md = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in t.values]
    block = "\n".join([
        "<!-- RESULTS:START -->",
        f"**Najlepszy model:** {s['best_model']} ({s['best_variant']}) — "
        f"R² = {tm['R2']:.3f}, MAE = {tm['MAE']:.3f}, MSE = {tm['MSE']:.3f}, RMSE = {tm['RMSE']:.3f} "
        f"(zbiór testowy, n = {s['n_test']}).",
        "", *md, "",
        f"_Zaktualizowano automatycznie: {date.today().isoformat()}_",
        "<!-- RESULTS:END -->",
    ])
    txt = path.read_text(encoding="utf-8")
    txt = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", block, txt, flags=re.S)
    path.write_text(txt, encoding="utf-8")


def ci_summary():
    s, clean, res = _load()
    tm = s["test_metrics"]
    t = _results_table(res)
    print("## 📊 Lab3 – Analizator wyników\n")
    print(f"**Najlepszy model:** {s['best_model']} ({s['best_variant']})\n")
    print("| R² | MAE | MSE | RMSE |\n|---|---|---|---|")
    print(f"| {tm['R2']:.3f} | {tm['MAE']:.3f} | {tm['MSE']:.3f} | {tm['RMSE']:.3f} |\n")
    print("<details><summary>Wszystkie modele</summary>\n")
    print("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns))
    for r in t.values:
        print("| " + " | ".join(map(str, r)) + " |")
    print("\n</details>")


if __name__ == "__main__":
    import sys
    if "--ci-summary" in sys.argv:
        ci_summary()
    else:
        build()
        update_readme()
