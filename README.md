# Lab3 – Analizator wyników

Model przewidujący zmienną `score` ze zbioru [CollegeDistance](https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv).
Cały proces (EDA → cechy → trening → tuning → ocena → dokumentacja) uruchamia GitHub Actions.

📄 **Dokumentacja z wykresami: [Readme.pdf](Readme.pdf)**

## Wyniki

<!-- RESULTS:START -->
**Najlepszy model:** Gradient Boosting (tuning) — R² = 0.341, MAE = 5.727, MSE = 49.052, RMSE = 7.004 (zbiór testowy, n = 948).

| Model | Wariant | CV_R2 | CV_RMSE | Test_R2 | Test_MAE | Test_MSE | Test_RMSE |
|---|---|---|---|---|---|---|---|
| Baseline (średnia) | domyślny | -0.001 | 8.721 | -0.000 | 7.263 | 74.402 | 8.626 |
| Regresja liniowa | domyślny | 0.333 | 7.121 | 0.337 | 5.738 | 49.343 | 7.024 |
| Ridge | domyślny | 0.333 | 7.121 | 0.337 | 5.738 | 49.344 | 7.024 |
| Las losowy | domyślny | 0.250 | 7.548 | 0.264 | 6.018 | 54.748 | 7.399 |
| Gradient Boosting | domyślny | 0.340 | 7.085 | 0.342 | 5.719 | 48.987 | 6.999 |
| Sieć neuronowa (MLP) | domyślny | 0.305 | 7.268 | 0.296 | 5.940 | 52.392 | 7.238 |
| Ridge | tuning | 0.333 | 7.121 | 0.337 | 5.739 | 49.344 | 7.025 |
| Las losowy | tuning | 0.337 | 7.100 | 0.340 | 5.729 | 49.093 | 7.007 |
| Gradient Boosting | tuning | 0.343 | 7.070 | 0.341 | 5.727 | 49.052 | 7.004 |
| Sieć neuronowa (MLP) | tuning | 0.310 | 7.240 | 0.292 | 5.901 | 52.646 | 7.256 |

_Zaktualizowano automatycznie: 2026-10-22_
<!-- RESULTS:END -->
