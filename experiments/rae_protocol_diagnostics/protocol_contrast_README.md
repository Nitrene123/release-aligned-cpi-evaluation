# Matched four-policy protocol contrast

This directory contains the frozen inputs, prepared panels, four-policy TimesFM-3 forecasts, derived scores, and scripts for Supplementary Note S15. The comparison uses the same B3 checkpoint, feature set, CPI history, 77 Chinese origins, and 308 origin-horizon keys for P0 latest captured, P1 release-lag only, P2 captured vintage as of origin, and P3 archived RAE.

The source-vintage registry is archive-bounded at 2024-12-09 UTC. It does not establish complete revision histories for all features. The P0--P2 labels therefore describe the available registry transformations, not a complete reconstruction of all historical data releases. No target values are included here. `scores/origin_errors_target_free.csv` contains derived per-origin errors, not target observations.

## Included files

- `inputs/`: source snapshots, vintage registry, archived B3 forecasts, and the fixed origin-horizon key file.
- `prepared/`: protocol-specific inputs, common keys, coverage summaries, and input hashes.
- `forecasts/`: frozen forecasts and inference metadata. The q50 replay differs from the archived B3 forecasts by at most 0.0016325 percentage points, within the 0.002-point tolerance fixed before scoring.
- `scores/`: MAE, RMSE, paired changes, descriptive block intervals, target-free origin errors, and score metadata.
- `prepare_protocol_inputs.py`, `run_protocol_contrast.py`, `score_protocol_contrast.py`: portable preparation, inference, and scoring steps.
- `PROTOCOL.md`: design, policy definitions, and limits.

## Reproduction

Preparation requires Python with NumPy and pandas:

```bash
python prepare_protocol_inputs.py
```

Inference requires CUDA, PyTorch, the official TimesFM source at commit `e51928e27119cb17bebc005be2696b75e0a9e688`, and the separately obtained checkpoint and configuration. The expected checkpoint SHA-256 is `a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`; the expected `config.json` SHA-256 is `ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6`.

Set these paths before running the inference script (PowerShell):

```powershell
$env:TIMESFM_SOURCE_DIR = 'C:\path\to\timesfm-at-pinned-commit'
$env:TIMESFM_CHECKPOINT_DIR = 'C:\path\to\timesfm3-checkpoint'
python run_protocol_contrast.py
```

The official checkpoint is available from [Google Research TimesFM](https://github.com/google-research/timesfm) and [the TimesFM-3.0 PyTorch model page](https://huggingface.co/google/timesfm-3.0-pytorch). Check and follow the weight license before use. This package does not redistribute the checkpoint or the official source tree.

Scoring reads only the target month and first-release CPI columns needed for the fixed keys. Supply the CPI panel used for this study and choose a new score-output directory:

```powershell
$env:CPI_TARGET_PANEL = 'C:\path\to\unified_cpi_panel.csv'
$env:RAE_PROTOCOL_SCORE_OUTPUT_DIR = 'C:\path\to\new-score-output'
python score_protocol_contrast.py
```

The target panel is not bundled. Scoring does not write target values to output files. The saved bootstrap intervals are descriptive circular moving-block intervals with 30,000 draws and block lengths of 3, 6, and 12 calendar months. They use the complete 83-month origin calendar and retain its six months without common origins as missing positions.

The reference inference run used Python 3.11.9, PyTorch 2.12.0.dev20260315+cu128, CUDA 12.8, and an NVIDIA RTX 5070 Laptop GPU; full environment metadata is in `forecasts/inference_metadata.json` and the package lock listing is `environment-lock.txt`.
