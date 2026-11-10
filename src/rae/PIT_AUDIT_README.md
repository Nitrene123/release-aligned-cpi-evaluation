# Point-in-time audit contract

`audit_point_in_time.py` checks a target-free JSON bundle containing forecast origins, model-specific required features, vintage-tagged inputs, planned forecast keys, and forecast/status rows. It writes an issue report and one eligibility row per planned model–origin–horizon key. The implementation uses only the Python standard library.

## Run

From the repository root:

```bash
python -m unittest discover -s tests -p "test_point_in_time.py" -v
python src/rae/audit_point_in_time.py src/rae/example_pit_bundle.json --out-dir audit_output
```

The output directory may exist, but the command refuses to overwrite either output file. Use a new directory for each run. Exit status is `0` for `PASS` and `2` for an audit result of `FAIL`.

## Bundle structure

The smallest complete example is [`example_pit_bundle.json`](example_pit_bundle.json). A production bundle uses these top-level fields:

| Field | Required content |
|---|---|
| `origins` | Unique `origin_id`, `origin_month` in `YYYY-MM`, and `origin_time_utc` as an ISO-8601 timestamp with UTC offset. |
| `required_features_by_model` | Mapping from each model name to the feature names that model requires at each origin. |
| `inputs` | Rows keyed by `origin_id`, `feature`, and `reference_month`; include `available_at_utc`, `vintage_id`, optional `expected_vintage_id`, `vintage_status`, `masked`, and a finite numeric `value` when unmasked. |
| `planned_forecasts` | One row per intended model/origin/horizon with `model`, `origin_id`, integer `horizon_months`, and `target_month`. |
| `forecasts` | One row per planned key with the same key and target month, a `status`, and a finite `prediction` when status is `FORECAST_COMPLETE`. |
| `expected_origin_ids_by_model_horizon` | Optional frozen coverage contract, keyed as `model|horizon`, listing the expected origin IDs. |

Allowed forecast statuses are `FORECAST_COMPLETE`, `INPUT_INELIGIBLE`, `FIT_FAILED`, and `PREDICTION_FAILED`. Failed forecasts remain explicit rows and are not treated as zero error. Rows in `inputs` or `forecasts` must not contain numeric target-value fields named `target_value`, `observed_target`, `actual_target`, or `label_value`.

## Checks performed

- Availability timestamps for unmasked inputs must be strictly earlier than the origin timestamp; reference months cannot follow the origin month.
- Selected vintages are checked against a supplied expected vintage ID, and statuses marked overwritten, superseded, or revoked are rejected.
- Unmasked input values and completed predictions must be finite numeric values.
- Origin, input, planned-forecast, and forecast keys must be valid and unique; every planned forecast must have a matching output/status row.
- Each target month must equal the origin month plus the forecast horizon. An optional expected-origin contract checks model-specific origin coverage.
- Per-key `input_eligible`, `forecast_complete`, and `evaluation_eligible` flags and reason codes are written to `eligibility.csv`.

## Outputs and interpretation

`audit_report.json` contains the overall status, counts, policy, and issue details. `eligibility.csv` records the decision for each planned key. The audit does not read or require target values and does not calculate forecast scores.

The result applies only to the metadata supplied in the bundle. The program does not independently authenticate publisher timestamps, enumerate an external vintage history, validate task-specific transformation/lag dependencies, or score forecasts. Those checks require source-specific evidence or separate modules.
