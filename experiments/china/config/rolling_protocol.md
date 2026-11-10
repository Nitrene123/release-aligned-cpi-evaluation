# Unified expanding-window rolling evaluation, 2018–2026

## Frozen scope

This protocol uses the original evaluation start, January 2018, and extends the target window through the latest released month, September 2026. It covers 105 target months at each of four horizons (1, 3, 6, and 12 months), or 420 planned target–origin–horizon keys. The origin month is the target month minus the horizon. Each origin clock is the official first-release time for that origin-month CPI announcement plus one minute.

This is a retrospective rolling-origin reconstruction. It is not blind, prospective, or an independent confirmation: the earlier 2026 results and target window have already been examined. The TimesFM-3 checkpoint also postdates the historical origins, and sequence-level pretraining exposure is unknown.

## What “one configuration” means

All methods share the same target definition, target months, origin clock, four evaluation horizons, strict point-in-time cutoff, coverage reporting, and scoring rules. Method-specific estimation remains appropriate: AR(1), SARIMA, and learned correction heads are refit at each origin using only information available before that origin; the official TimesFM-3 checkpoint remains frozen. Their fitted coefficients and head weights therefore change by origin, while the protocol and hyperparameters do not.

The ten fixed method families are No-change, Seasonal-naive, expanding-origin AR(1), expanding-origin SARIMA, TimesFM-3 target-only, multivariate TimesFM-3 (B3), BEAH, Direct, Fixed candidate, and Fixed candidate plus a four-parameter gate. BEAH, Direct, and GateOnly use seeds 417, 42, 123, 2027, and 3407. The last 12 eligible historical h=3 examples select the epoch count; a fresh same-seed model is then fitted on all eligible pre-origin rows.

## Point-in-time controls

Every input must have a documented availability timestamp strictly earlier than the origin clock. Historical training labels enter a later origin's fit only after their first official release and only if their release timestamp is strictly earlier than that later origin. A label may be a legitimate training observation for a later forecast even when it was an evaluation target at an earlier origin; this is the standard expanding-window information flow and is logged per use.

For origins from January 2025 through August 2026, the run uses the archived G3 origin-specific inputs. The September 2026 target value is not part of the forecast-building input bundle. Its release-time metadata may be used to validate that the forecast precedes the target release; its numeric value is reserved for the separate scoring stage. Missing or unresolved covariates remain explicitly masked. No interpolation, substitution, or fallback run is permitted.

## Model settings

- AR(1): OLS with an intercept on adjacent monthly first-release CPI YoY pairs, refit at every origin, with recursive forecasts.
- SARIMA: `SARIMAX(order=(1,0,0), seasonal_order=(1,0,0,12), trend='c')`, with stationary initialization, stationarity and invertibility constraints, no differencing, no measurement error, and `statsmodels 0.14.4`; refit at every origin using `lbfgs`, `maxiter=500`, `pgtol=1e-8`, and `factr=1e7`.
- TimesFM-3 target-only and B3: same official checkpoint SHA-256 `a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`, repository commit `e51928e27119cb17bebc005be2696b75e0a9e688`, direct decode, 12-step output, q0.5 after monotone quantile rearrangement. The checkpoint is not updated.
- BEAH, Direct, and GateOnly: pinned G1/G4 implementation, origin-specific expanding fit, 12 most recent eligible h=3 validation origins, selected h=3 validation MAE, then fresh same-seed final fit on all eligible labels released before the outer origin. No hyperparameter search.

## Evaluation and reporting

The primary metric is MAE, reported for all methods and all four horizons. Report each method's native coverage, pairwise common-origin coverage, the all-method common-origin intersection, origin-level paired losses, and win/loss/tie counts. Five seeds are averaged within an origin before aggregation; they are not independent time observations. Missing forecasts are not zero errors. No p-values are used; any resampling intervals are descriptive and cannot create additional independent months.

This protocol does not modify prior G6 outputs, prior 2018–2024 runs, P1, or the prospective queue. Scores are generated only after the forecast archive is completed and sealed.
