# PIT mask counterfactual: unified China CPI cohort (2018–2026)

## What this bundle supports

This bundle makes the B3 inference step replayable from the frozen input snapshots for the unified Chinese CPI rolling cohort. It includes 157 origin contexts, the strict point-in-time covariate masks, the conservative post-origin recovery map, the archived q50 reference, code pinned to the official TimesFM repository commit, and the complete strict/backfill forecast outputs. The reference execution used an NVIDIA A800; the runner now permits another CUDA GPU and records its identity, while enforcing the checkpoint hash, official source commit, immutable input hashes, complete forecast keys, and a 0.002 percentage-point q50 parity tolerance.

The model checkpoint is not redistributed. Obtain `google/timesfm-3.0-pytorch` through Google's official model distribution, accept the applicable separate TimesFM Non-Commercial License, and verify `model.safetensors` has SHA-256:

`a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`

The official inference source is pinned at commit `e51928e27119cb17bebc005be2696b75e0a9e688`. The code checks the installed Git checkout at runtime and stops on a different commit. The inference process uses historical CPI values as model context; it does not read target outcomes for scoring. The bundle does not contain a raw target panel.

## Rebuild the recovery map

Use Python with NumPy and pandas installed:

```bash
python code/build_recovery_map.py --input-dir inputs --output-dir rebuilt_recovery_map
```

This reads only the supplied source audit, point-in-time ledger, and frozen input snapshots. It should produce 104 verified post-origin cells. Compare `rebuilt_recovery_map/mask_recovery_map.csv` byte-for-byte with `inputs/mask_recovery_map.csv`; do not use unverified snapshot-only recoveries.

## Re-run TimesFM-3 inference

1. Create an environment matching `inputs/environment-lock.txt` (Python 3.12.3 and CUDA-enabled PyTorch 2.5.1+cu121 in the reference run).
2. Install the pinned official source and dependencies from `inputs/environment-lock.txt`.
3. Download the official checkpoint to a local directory containing `model.safetensors` and verify its SHA-256 as above.
4. From this bundle's root, run:

```bash
python code/run_pit_mask_counterfactual.py \
  --input-dir inputs \
  --checkpoint-dir /path/to/official/timesfm3-checkpoint \
  --output-dir outputs_rebuilt \
  --device cuda:0 \
  --batch-size 4
```

The output directory must not already exist. The runner never reads a scoring target, never accesses P1, and writes both strict replay and the deliberately invalid post-origin-backfill diagnostic. The strict q50 values must match `inputs/archived_b3_q50.csv` within 0.002 pp. The reference A800 run differed by at most `8.9e-16` pp over 157 × 12 strict q50 forecasts. Other CUDA hardware may differ numerically, but the runner fails if the declared tolerance is exceeded.

## Main retrospective finding

The expanded score covers 99 common origin–target pairs per horizon (h=1, 3, 6, 12), with 104 verified post-origin recoveries among 177 masked cells in the 110-origin union; 73 cells remain masked. Backfilling changed B3 MAE by −0.000208, +0.004396, +0.006825, and +0.011592 percentage points at h=1, 3, 6, and 12. The direction is horizon-dependent. These paired origin and year summaries are descriptive, post-score analyses; there are no p-values or inferential intervals. The post-origin variant deliberately uses information unavailable at the forecast origin and is not a valid forecasting protocol.

`derived_scores/` contains target-free per-origin losses and summaries. `code/verify_derived_outputs.py` checks hashes, key counts, strict q50 parity, and score arithmetic without loading target values:

```bash
python code/verify_derived_outputs.py
```

It verifies the archived derived results; it does not independently validate the external first-release target values. The separate scoring source `code/score_mask_counterfactual.py` is included for transparency and requires a locally obtained, independently validated first-release target panel plus the archived signed-loss input. Neither is bundled. The score report notes that one final target month was transiently reconstructed from a previously scored signed residual; raw target values were not written to the output files.

## Interpretation and remaining boundary

This diagnostic is retrospective, after target releases and earlier project results were known. It does not provide blind, prospective, or independent confirmation, and it does not estimate a general effect of point-in-time screening. It covers one model configuration, one CPI series, and two mask states (strict versus a deliberately invalid post-origin backfill). It broadens the earlier 36-origin diagnostic to 99 pairs per horizon but does not turn overlapping monthly forecasts into independent observations.

The bundle supports a portable **frozen-snapshot-to-inference replay**, conditional on licensed checkpoint access and the pinned official source. It does not reconstruct every raw source response into the frozen snapshots, rerun every original Chinese/U.S. model cohort, or provide an external operating-system-level access audit. P1 is outside scope and was not accessed.
