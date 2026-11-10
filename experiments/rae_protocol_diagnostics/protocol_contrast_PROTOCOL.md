# Matched information-set contrast on the China CPI cohort

## Design and scope

This is a retrospective, same-model, same-origin protocol contrast for the frozen TimesFM-3 multivariate configuration (B3). It uses the 77 origin months from 2018-01 through 2024-11 that are present at all four registered horizons (h=1, 3, 6, 12), for 308 fixed origin-horizon keys. The target definition, CPI context, checkpoint, inference API, feature order, and scoring keys are held fixed. The 2026-11-09 captured source-vintage registry is the archive cutoff for the value scenarios.

The four input constructions are:

| Label | Construction | Coverage status |
|---|---|---|
| P0 latest captured value | For each feature and source month, use the last value in the captured registry, without filtering by its release time relative to the forecast origin. Missing registry cells remain masked. | A latest-value snapshot as captured through the registry cutoff, not a current all-source database. |
| P1 release-lag only | Use the same last captured value as P0, but expose it only when the first recorded publication time for that source month is no later than the origin timestamp. Later revisions may therefore remain in the value while release lag is respected. | Timing-only diagnostic; it does not recreate a historical vintage. |
| P2 captured vintage as of origin | Select the last registry version whose recorded availability is no later than the forecast origin. If no such version is recorded, mask the cell. | Partial vintage coverage: the registry is revision-rich for Brent and contains one recorded version per month for the other three features; completeness of their revision histories is not established. |
| P3 archived RAE input | Use the original B3 input snapshot and its archived mask without alteration. | The strict reference input configuration used by the archived B3 forecasts. |

P0, P1, and P2 are therefore explicitly archive-bounded diagnostics. They are not a complete four-level ablation of all source versions. The P2 label refers to the captured registry selection rule and does not imply complete source-history coverage.

## Frozen processing and analysis

1. Select the 2018-01 to 2024-11 origin range from `scored_origin_horizon_keys.csv`; require exactly the same 77 origin IDs at each horizon.
2. Build all four protocol inputs over the same archived CPI context month grid and four features. Values unavailable under a protocol are zero-filled and represented by the official TimesFM-3 mask tensor; no protocol may drop an origin.
3. Generate q10--q90 outputs for each protocol from the official TimesFM-3 checkpoint and the source revision recorded in `inference_metadata.json`. P3 q50 must reproduce the archived B3 forecast within 0.002 CPI percentage points before scoring proceeds.
4. Freeze and hash the target-free forecast file. Then score the 308 fixed origin-horizon keys against the Chinese first-release CPI panel. Do not export target values.
5. Report MAE, RMSE, paired origin-level absolute-loss changes against P3, direction counts, and descriptive circular moving-block intervals (30,000 draws, seed 417, block lengths 3, 6, and 12 calendar months). Resample on the complete 2018-01--2024-11 monthly origin axis, retaining months without a common origin as missing positions; each bootstrap draw averages over its observed paired origins. These are retrospective descriptive contrasts, not confirmatory tests.

No model is retrained or tuned. No separate delayed P1 queue is used. Forecast generation itself does not read target-month outcomes; the scoring script accesses only the target series required for the fixed keys after forecasts are frozen.

## Provenance limits

The registry covers 2012-01 to 2024-11 and is not a complete current snapshot. It contains 9,516 Brent vintage records matched to 13,002 observed Brent cells in the full scored-origin union; the remaining 3,486 cells have no matching registry record. The three other macro series have one archived record per source month in this registry, so the experiment cannot quantify their full revision histories. Any interpretation must retain these source-specific coverage limits.
