# Independent SARIMA Rolling Replay Report

## Scope

Independent local refit of the archived Chinese unified expanding-window SARIMA(1,0,0)x(1,0,0,12) forecasts. The 116 origin-specific histories were reconstructed from the bundled first-release CPI panel using each recorded origin cutoff. The replay did not modify the sealed archive.

## Verification

- Archive CRC: PASS; sealed manifest entries verified: 39.
- Environment: Python 3.12.14, statsmodels 0.14.4, NumPy 2.2.6, SciPy 1.15.3.
- Refit origins: 116; forecast rows compared: 420.
- Parameter max absolute difference: 9.57159462492e-05; origin records exceeding 1e-8: 45 of 116; fit status mismatches: 0.
- Maximum CPI input-value difference versus the archived point-in-time snapshots: 0 pp; release timestamps and monthly history keys matched.
- Optimizer convergence flags differed at 5 origins (CN_2019-03, CN_2019-12, CN_2020-01, CN_2020-04, CN_2025-11); both fits returned finite coefficients and forecasts.
- Forecast max absolute difference: 0.00018590651429 percentage points.
- Forecast rows exceeding 1e-8: 176 of 420; absolute-error max difference: 0.000185906514123 pp.
- Largest absolute native/common MAE difference across horizons: 5.91742e-06 pp.
- Common-origin counts by horizon: {'1': 99, '3': 99, '6': 99, '12': 99}.
- Overall status: **REPLAY_COMPLETE_WITH_NUMERICAL_DIFFERENCES**.

## MAE summary

| Horizon | Native n | Archived MAE | Replay MAE | Difference | Common n | Archived common MAE | Replay common MAE | Difference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 105 | 0.303345603 | 0.303345110 | -0.000000493 | 99 | 0.301685677 | 0.301685106 | -0.000000570 |
| 3 | 105 | 0.611312466 | 0.611313101 | +0.000000634 | 99 | 0.614076415 | 0.614077220 | +0.000000805 |
| 6 | 105 | 0.893315326 | 0.893317596 | +0.000002270 | 99 | 0.882100029 | 0.882102758 | +0.000002729 |
| 12 | 105 | 1.199324868 | 1.199330163 | +0.000005295 | 99 | 1.185830483 | 1.185836400 | +0.000005917 |

Convergence-flag differences (archived vs replay; all were recorded as fit-complete and returned finite forecasts):

| Origin | Archived flag / iterations | Replay flag / iterations |
|---|---:|---:|
| CN_2019-03 | True / 31 | False / 30 |
| CN_2019-12 | False / 35 | True / 36 |
| CN_2020-01 | False / 37 | True / 37 |
| CN_2020-04 | False / 34 | True / 32 |
| CN_2025-11 | True / 24 | False / 23 |

The independently recomputed common-sample SARIMA MAEs round to the same four-decimal values as the manuscript (0.3017, 0.6141, 0.8821, 1.1858). The four-horizon sample minima therefore remain: SARIMA at h=1 and h=3, Fixed candidate + gate at h=6, and AR(1) at h=12.

The archive records statsmodels but does not fully bind the original operating-system and BLAS build. The source histories and release cutoffs match exactly, but the remaining small coefficient and forecast differences cannot be attributed to a single runtime cause from the archived metadata; this is not an exact numerical reproduction.

The CSV contains the matching RMSE values and sample denominators. `archived` values come from the scored origin-loss archive; `replay` values are recomputed from the independently refitted forecasts and the same first-release target mapping.

## Row-level outputs

- `sarima_parameters_by_origin.csv`: archive and replay parameter vectors, convergence and history metadata for each origin.
- `sarima_forecasts_by_origin_horizon.csv`: original and replayed forecast, prediction difference, score-derived original absolute error, and replay absolute error for every horizon key.
- `replay_metadata.json`: input/archive hashes, environment, tolerances, counts, and summary verdict.

## Interpretation boundary

This is a reproducibility check of one archived historical rolling run using the same frozen specification and point-in-time histories. It is not a new model-selection exercise or an independent prospective evaluation. The raw target values are not written to the row-level output.
