# Release-Aligned Forecast Evaluation Toolkit

This repository contains software and reproducibility materials for auditing time-dependent forecasting evaluations.

## Contents

- Release, vintage, dependency, model-eligibility, target-mapping, and provenance checks with synthetic examples.
- Information-violation benchmarks, computational ablations, and held-out fault tests.
- Historical CPI and industrial-production experiment configurations, forecast archives, derived scores, and figure-generation scripts.
- Data-source, licensing, and reproduction instructions.

## Quick start

Core audit tests use only the Python standard library:

```bash
python -m unittest discover -s tests -p "test_point_in_time.py" -v
python -m unittest discover -s tests -p "test_dependency_graph.py" -v
python -m unittest discover -s tests -p "test_model_input_eligibility.py" -v
python tests/record_current_audit_validation_results.py --output-dir audit_validation_output
python src/rae/audit_point_in_time.py src/rae/example_pit_bundle.json --out-dir audit_output
```

Experiment-specific README files describe how to reproduce audit benchmarks, forecasting results, and derived tables.

## Data and model access

No TimesFM-3 checkpoint weights, third-party raw economic observations, private input tensors, credentials, or internal server logs are included. Official data sources and local reconstruction paths are listed in `DATA_SOURCES.md`. Forecast outputs and loss summaries are supplied as derived artifacts where needed to verify reported tables and figures.

The repository distinguishes score reconstruction from full input-to-inference reruns. Each experiment README describes the inputs needed for a full rerun and the checks supported by the released derived files.

## Citation and license

Original source code is released under the MIT License. The code license does not grant rights to third-party datasets or model checkpoints; see `DATA_LICENSE.md`.
