# RAE computational ablation protocol

## Frozen inputs

The runner reads the 800-case independent truth table, the fixture manifest, the case bundles, and the audited implementation under `baseline_source/`. These inputs are read-only. Each fixture is SHA-256 checked before and after the ablation.

## Output-directory contract

Every run must write to a new output directory. The runner fails before reading the fixtures if the requested output directory already exists. It also rejects output paths inside `cases/` or `baseline_source/`. CSV, JSON, and manifest files are opened in exclusive-create mode, so a pre-existing output file is never replaced.

The archived result directory `results/computational_ablation/` already exists and is preserved. A rerun must specify a different path, for example:

```powershell
python run_computational_ablation.py --root . --out results/computational_ablation_run_20261110T150000Z
```

The requested path must not exist before the command starts. If a run stops after creating its output directory, that partial directory remains protected; choose another new path for a subsequent run. The runner does not delete or clean prior outputs.

## Runner provenance

The archived `benchmark_report.json` records the runner SHA-256 used for the completed benchmark as `f53958ddc0acd648f8f6720987adb4d1ea89e65b175cc4f5c6bd1271ac04cdd9`. That exact historical runner is preserved as `provenance/runner_as_executed.py`. The current `run_computational_ablation.py` received an output-safety maintenance change: it now refuses an existing output directory and creates CSV, JSON, and manifest files exclusively. Its current source SHA-256 is `26ded2b7001dc7f5d5fbcf52a1ce069cd4ad24d27618182108d9ef2fa7baa325`. The change is limited to output-path and file-creation behavior; the archived result files were not rerun or modified. The current source hash applies to future runs.

## Output set

A successful run writes the eight result CSVs, `benchmark_report.json`, and `SHA256SUMS.txt` into the new directory. The manifest contains exact raw-file hashes for that run. Environment and timing fields are recorded as metadata and can differ across machines; they are not evidence of numerical disagreement by themselves.
