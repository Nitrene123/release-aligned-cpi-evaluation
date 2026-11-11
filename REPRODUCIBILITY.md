# Reproduction guide

## Verify the released files

```bash
python scripts/verify_public_release.py
```

After a deliberate file update and review, regenerate the file manifest with `python scripts/verify_public_release.py --write-manifest`, then run the verification command again.

The root manifest normalizes text-file line endings to LF before hashing, so the same text files verify on Windows and POSIX. Binary artifacts are hashed byte-for-byte.

## Run the point-in-time auditor tests

```bash
python -m unittest discover -s tests -p "test_point_in_time.py" -v
python src/rae/audit_point_in_time.py src/rae/example_pit_bundle.json --out-dir audit_output
```

The example is synthetic and contains no target values.

## Rebuild the information-violation benchmark

Follow `experiments/rae_injection_800/README.md`. A clean run generates its own synthetic cases, runs the auditor and Feast retrieval modes, and writes outputs to the experiment workspace. The full factorial ablation then evaluates 32 configurations on the same cases.

## Rebuild reported score tables and figures

Target-free forecast archives and origin-level loss tables are included in `experiments/*/reference_results/` and `experiments/figures/data/`. Score aggregation and figure scripts are included. Where an experiment needs target values to recompute a score, obtain them from the cited official provider and supply a local target file to the scoring script. Target values are not exported into the derived result tables here.

## Re-run model inference

CPI rolling runners require locally reconstructed input bundles and, for TimesFM-3, an authorized local checkpoint. Raw input panels and model weights are not distributed. Runner code, protocols, model forecasts, input fingerprints, and target-free loss summaries are provided for source review and score verification. See each experiment README for external inputs and configuration.

## Re-run the IPT vintage experiment

Obtain the IPT source workbooks from the Philadelphia Fed link in `DATA_SOURCES.md`, then run the input-rebuild script with local workbook and output paths. Set the TimesFM-3 checkpoint via `--checkpoint` or `TIMESFM3_CHECKPOINT`. Do not put source workbooks or checkpoint weights in the public repository.

## Rebuild figures

Install `requirements/figures.txt` and run the scripts under `experiments/figures/code/`. Their inputs are the derived tables under `experiments/figures/data/`.

## Reproduction scope

Released materials support (1) exact reruns of the synthetic auditor tests, (2) regeneration of the synthetic injection and ablation experiment, (3) metric aggregation checks from target-free loss tables, and (4) figure rebuilding from derived metric tables. Full inference from source inputs requires the external data and official checkpoint listed in `DATA_SOURCES.md`.
