# Release-Aligned Auditing for High-Confidence Evaluation of Time-Series Forecasts: Evidence from CPI and Industrial Production

This repository contains the study's point-in-time audit implementation, synthetic information-violation benchmark, rolling forecast and scoring code, experiment configurations, validation tests, and target-free derived outputs. Chao Liu and Yanbin Zhang contributed equally.

## Contents

- A reusable release-aligned input-contract auditor, its [input/output schema](src/rae/PIT_AUDIT_README.md), synthetic example, and 19 contract tests.
- An independently generated 800-case information-violation benchmark, a Feast feature-retrieval comparison, and a complete 2^5 computational ablation (32 configurations and 25,600 audit calls).
- China and U.S. CPI experiment code, frozen protocols, target-free per-origin losses, forecast archives, coverage summaries, and paired uncertainty outputs.
- IPT vintage-stress, P0–P3 diagnostic, ETS/Theta, and Timestral experiment code with derived results.
- Figure-generation scripts and the derived metric tables used by them.

## Quick start

Core audit tests use only the Python standard library:

```bash
python -m unittest discover -s tests -p "test_point_in_time.py" -v
python src/rae/audit_point_in_time.py src/rae/example_pit_bundle.json --out-dir audit_output
```

The 800-case benchmark and 32-configuration ablation are documented in `experiments/rae_injection_800/README.md`. CPI scoring and model rerun instructions are in `REPRODUCIBILITY.md`.

## Data and model access

No TimesFM-3 checkpoint weights, third-party raw economic observations, private input tensors, credentials, or internal server logs are included. Official data sources and local reconstruction paths are listed in `DATA_SOURCES.md`. Forecast outputs and loss summaries are supplied as derived artifacts where needed to verify reported tables and figures.

The repository distinguishes score reconstruction from full input-to-inference reruns. Each experiment README describes the inputs needed for a full rerun and the checks supported by the released derived files.

## Citation and license

Please cite the associated article as specified in `CITATION.cff`. Chao Liu and Yanbin Zhang are co-first authors. Original source code is released under the MIT License. The code license does not grant rights to third-party datasets or model checkpoints; see `DATA_LICENSE.md`.
