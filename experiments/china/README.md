# China rolling CPI evaluation

This folder contains source for the common expanding-window protocol, correction-head training, statistical baselines, scoring, coverage, and target-free prediction/loss archives. Raw CPI values, macroeconomic input panels, release-response files, and model weights are not included.

The runners resolve the project root through `code/portable_paths.py` and the `ECOFM_CPI_ROOT` environment variable. Reconstruct the external inputs from the official sources in `DATA_SOURCES.md`, preserving release/vintage metadata and the frozen protocol layout. TimesFM-3 inference requires a locally obtained checkpoint. Use the exact configuration associated with the archived outputs when checking the original results.

Use `score_unified_rolling.py` and `audit_unified_score.py` for score reconstruction and audit after supplying an official target table. The included target-free score summaries and per-origin loss table permit aggregation checks without distributing the target values.
