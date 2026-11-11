# Release-Aligned Evaluation for High-Confidence Auditing of Time-Series Forecasts: Evidence from CPI and Industrial Production

This repository contains the study's point-in-time audit implementation, synthetic information-violation benchmark, rolling forecast and scoring code, experiment configurations, validation tests, and target-free derived outputs. Chao Liu and Yanbin Zhang contributed equally.

## Contents

- A reusable release-aligned input-contract auditor, its [input/output schema](src/rae/PIT_AUDIT_README.md), and synthetic examples. The current contract suite includes 19 general input-contract tests, 24 dependency-DAG tests (including 12 S22 regression cases), and 13 exact model-origin eligibility tests; the combined run is recorded in `tests/current_audit_validation_results.json`.
- An independently generated 800-case information-violation benchmark, a Feast feature-retrieval comparison, and a complete 2^5 computational ablation (32 configurations and 25,600 audit calls).
- In the separate 900-case comparison reported in Supplementary Note S21, RAE and the modular Feast--Pandera stack produced the same 1,704 location keys. Of these, 900 match generator-designated primary labels and 804 are unadjudicated secondary findings without independent record-level truth; the 52.8% primary-label match fraction is not independently established record-level precision. The modular-stack materials are supplied in the manuscript supplementary archive.
- A case-held-out fault-generalization suite with 200 synthetic bundles, plus 500 seeded property and metamorphic trials. The checker snapshot frozen before generation flagged 138/150 faulted cases (92.0% recall), with 100% precision and no false alarms among 50 clean controls. A later portable-checker update adds a DAG audit and regression tests for the 12 previously missed transitive-parent cases; historical suite metrics remain unchanged.
- China and U.S. CPI experiment code, frozen protocols, target-free per-origin losses, forecast archives, coverage summaries, and paired uncertainty outputs.
- IPT vintage-stress, P0–P3 diagnostic, ETS/Theta, and Timestral experiment code with derived results.
- Figure-generation scripts and the derived metric tables used by them.

## Quick start

Core audit tests use only the Python standard library:

```bash
python -m unittest discover -s tests -p "test_point_in_time.py" -v
python -m unittest discover -s tests -p "test_dependency_graph.py" -v
python -m unittest discover -s tests -p "test_model_input_eligibility.py" -v
python tests/record_current_audit_validation_results.py --output-dir audit_validation_output
python src/rae/audit_point_in_time.py src/rae/example_pit_bundle.json --out-dir audit_output
```

The 800-case benchmark and 32-configuration ablation are documented in `experiments/rae_injection_800/README.md`. CPI scoring and model rerun instructions are in `REPRODUCIBILITY.md`.

The held-out generalization experiment, its frozen checker, independent case generator, case bundles, truth manifest, and archived results are in `experiments/rae_heldout_fault_generalization/`. Reproduction commands and interpretation are provided in that directory's `README.md` and `RESULTS.md`.

## Data and model access

No TimesFM-3 checkpoint weights, third-party raw economic observations, private input tensors, credentials, or internal server logs are included. Official data sources and local reconstruction paths are listed in `DATA_SOURCES.md`. Forecast outputs and loss summaries are supplied as derived artifacts where needed to verify reported tables and figures.

The repository distinguishes score reconstruction from full input-to-inference reruns. Each experiment README describes the inputs needed for a full rerun and the checks supported by the released derived files.

## Citation and license

Please cite the associated article as specified in `CITATION.cff`. Chao Liu and Yanbin Zhang are co-first authors. Original source code is released under the MIT License. The code license does not grant rights to third-party datasets or model checkpoints; see `DATA_LICENSE.md`.
