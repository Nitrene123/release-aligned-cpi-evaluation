# Point-in-time audit contract

`audit_point_in_time.py` checks a target-free JSON bundle containing forecast origins, model-specific required features, vintage-tagged inputs, planned forecast keys, and forecast/status rows. It writes an issue report and one eligibility row per planned model–origin–horizon key. The implementation uses only the Python standard library.

## Run

From the repository root:

```bash
python -m unittest discover -s tests -p "test_point_in_time.py" -v
python -m unittest discover -s tests -p "test_dependency_graph.py" -v
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

For exact I4 eligibility, prefer `required_inputs_by_model_origin` over the legacy feature-only contract:

```json
"required_inputs_by_model_origin": {
  "TimesFM": {
    "O1": [
      {"feature": "CPI", "lag": 1, "reference_month": "2023-12"},
      {"feature": "PPI", "lag": 2, "reference_month": "2023-11"}
    ]
  }
}
```

Each requirement is the exact tuple $(feature, lag, reference\_month)$ for one model and origin. `lag` is a non-negative calendar-month distance from `origin_month`; the auditor checks that the declared reference month equals `origin_month - lag`. Every planned origin for a model using this schema must have an explicit contract, including an empty list when no context inputs are required. An input row may be global (no `model`) or scoped to one model; a model-scoped row cannot satisfy another model's contract. A requirement is eligible only when exactly one matching input row exists and is unmasked, finite, active, vintage-consistent, dependency-valid, and available strictly before the origin. Missing or ambiguous rows make the origin ineligible. Duplicate feature-lag rows are rejected.

Allowed forecast statuses are `FORECAST_COMPLETE`, `INPUT_INELIGIBLE`, `FIT_FAILED`, and `PREDICTION_FAILED`. Failed forecasts remain explicit rows and are not treated as zero error. Rows in `inputs` or `forecasts` must not contain numeric target-value fields named `target_value`, `observed_target`, `actual_target`, or `label_value`.

## Checks performed

- Availability timestamps for unmasked inputs must be strictly earlier than the origin timestamp; reference months cannot follow the origin month.
- Selected vintages are checked against a supplied expected vintage ID, and statuses marked overwritten, superseded, or revoked are rejected.
- Unmasked input values and completed predictions must be finite numeric values.
- Origin, input, planned-forecast, and forecast keys must be valid and unique; every planned forecast must have a matching output/status row.
- Each target month must equal the origin month plus the forecast horizon. An optional expected-origin contract checks model-specific origin coverage.
- Per-key `input_eligible`, `forecast_complete`, and `evaluation_eligible` flags and reason codes are written to `eligibility.csv`.
- A completed forecast whose input contract is ineligible is reported as `FORECAST_FROM_INELIGIBLE_INPUT` and is excluded from evaluation eligibility.
- `coverage_by_model_horizon` reports the complete planned-origin denominator, input-eligible origins, completed forecasts on eligible and ineligible inputs, and eligible coverage. Ineligible origins are never silently removed from the denominator.

The I4 contract suite covers a missing required lag cell, a row without lag metadata, NaN/Inf values, masking, duplicate feature-lag rows, lag/reference inconsistency, release cutoffs, forecasts produced from ineligible inputs, model-specific input scope, explicit per-origin requirements, and denominator retention. Run all three audit suites with:

```bash
python -m unittest discover -s tests -p "test_point_in_time.py" -v
python -m unittest discover -s tests -p "test_dependency_graph.py" -v
python -m unittest discover -s tests -p "test_model_input_eligibility.py" -v
```

The current machine-readable combined run is `tests/current_audit_validation_results.json` (56 tests across I1/I2/I5, I3, and I4 suites). The test results are software-contract evidence for the supplied bundle schema; they do not establish that an unprovided model manifest lists every input a model actually consumes.

## Derived-input dependency DAG

An unmasked input is treated as derived when `is_derived` is `true`, `dependency_node_id` is supplied, or an input-level `transformation_id` is supplied. Such a row must identify a derived root in the bundle-level `dependency_graph`. The graph uses `nodes`, `edges`, and `transformation_registry`:

- Each edge has `child_node_id` and `parent_node_id`; traversal follows every transitive parent from the selected root.
- A `source` node records `reference_month`, `available_at_utc`, `source_id`, `series_id`, `vintage_id`, `source_sha256`, and optional `vintage_status`.
- A `derived` node records its own parents, `reference_month`, `available_at_utc`, `transformation_id`, JSON-object `parameters`, `parameters_sha256`, and `code_sha256`.
- `transformation_registry` maps each transformation ID to the expected parameter and code hashes. Parameter hashes use UTF-8 JSON with sorted keys, compact separators, `ensure_ascii=false`, and NaN disallowed. The input row's optional `transformation_id` must match the root node.

The checker performs an iterative ancestor walk and a Kahn topological cycle check. It rejects cycles, absent nodes or parents, malformed source identity/hash metadata, invalid or unregistered transformations, parameter/code hash mismatches, future reference months, and any ancestor whose availability is equal to or later than the origin. An invalid derived root makes that input ineligible. The regression suite includes all 12 transitive future-parent cases missed by the earlier frozen S22 checker.

This is a provenance-metadata check. It assumes the supplied graph and registries are complete and trusted, and that hashes were generated from the intended artifacts. It does not independently authenticate the registry, retrieve omitted dependencies, or replay the transformation to establish numerical correctness.

## Outputs and interpretation

`audit_report.json` contains the overall status, counts, policy, and issue details. `eligibility.csv` records the decision for each planned key. The audit does not read or require target values and does not calculate forecast scores.

The result applies only to the metadata supplied in the bundle. The program does not independently authenticate publisher timestamps or transformation registries, establish that the supplied graph is complete, enumerate an external vintage history, replay transformations, or score forecasts. Full historical-vintage maximality and forecast-to-input fingerprint checks require source-specific evidence or separate modules.
