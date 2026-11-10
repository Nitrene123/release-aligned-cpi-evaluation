#!/usr/bin/env python3
"""Replay frozen TimesFM-3 B3 inputs under strict and partial Brent-vintage variants.

This target-free runner uses only the packaged origin snapshots, recovery map,
and archived B3 q50 reference. It never reads target-month outcomes or scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FEATURES = (
    "PPI_YOY_PCT",
    "M2_YOY_PCT",
    "BRENT_LOG_YOY_PCT",
    "USDCNY_LOG_YOY_PCT",
)
EXPECTED_CHECKPOINT_SHA256 = "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"
EXPECTED_OFFICIAL_SOURCE_COMMIT = "e51928e27119cb17bebc005be2696b75e0a9e688"
Q_NAMES = (10, 20, 30, 40, 50, 60, 70, 80, 90)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)


def verify_official_source_commit(module_file: Path) -> str:
    """Require the installed official inference source to match the pin."""
    for root in (module_file.parent, *module_file.parents):
        if (root / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            )
            commit = result.stdout.strip()
            if commit != EXPECTED_OFFICIAL_SOURCE_COMMIT:
                raise RuntimeError(f"TimesFM source commit mismatch: {commit}")
            return commit
    raise RuntimeError(
        "Cannot verify the pinned TimesFM source commit. Install the official "
        "repository at the commit recorded in environment-lock.txt."
    )


def read_inputs(input_dir: Path):
    b3 = pd.read_csv(input_dir / "b3_input_snapshots.csv", dtype={"context_reference_month": str})
    cpi = pd.read_csv(input_dir / "cpi_input_snapshots.csv", dtype={"observation_month": str})
    rec = pd.read_csv(input_dir / "mask_recovery_map.csv", dtype={"context_reference_month": str})
    ref = pd.read_csv(input_dir / "archived_b3_q50.csv", dtype={"target_month": str})
    scored = pd.read_csv(input_dir / "scored_origin_horizon_keys.csv", dtype={"target_month": str})
    if set(b3.feature_name.unique()) != set(FEATURES):
        raise ValueError("Covariate feature set or ordering contract changed")
    if b3.duplicated(["origin_id", "context_reference_month", "feature_name"]).any():
        raise ValueError("Duplicate B3 input key")
    if cpi.duplicated(["origin_id", "observation_month"]).any():
        raise ValueError("Duplicate CPI context key")
    if rec.duplicated(["origin_id", "context_reference_month", "feature_name"]).any():
        raise ValueError("Duplicate mask-recovery key")
    if ref.duplicated(["origin_id", "horizon_months"]).any():
        raise ValueError("Duplicate archived B3 reference key")
    if scored.duplicated(["origin_id", "horizon_months"]).any():
        raise ValueError("Duplicate scored origin-horizon key")

    origin_rows = b3[["origin_id", "origin_month", "origin_timestamp_utc"]].drop_duplicates()
    if origin_rows.origin_id.nunique() != len(origin_rows) or len(origin_rows) != 157:
        raise ValueError("Expected the original 157-origin B3 batching cohort")
    origin_rows = origin_rows.sort_values(["origin_month", "origin_id"])
    b3_groups = {k: v for k, v in b3.groupby("origin_id", sort=False)}
    cpi_groups = {k: v for k, v in cpi.groupby("origin_id", sort=False)}
    rec_map = {
        (str(r.origin_id), str(r.feature_name), str(r.context_reference_month)): np.float32(r.recovered_value)
        for r in rec.itertuples(index=False)
    }

    examples = []
    for origin in origin_rows.itertuples(index=False):
        oid = str(origin.origin_id)
        cov = b3_groups[oid].sort_values(["context_reference_month", "feature_name"])
        target = cpi_groups[oid].sort_values("observation_month")
        months = target.observation_month.astype(str).tolist()
        cov_months = sorted(cov.context_reference_month.astype(str).unique().tolist())
        if months != cov_months:
            raise ValueError(f"CPI/covariate month grids differ at {oid}")
        if not months or months[-1] != str(origin.origin_month):
            raise ValueError(f"CPI context does not end at its origin: {oid}")
        target_values = target.value.to_numpy(dtype=np.float32)
        if not np.isfinite(target_values).all():
            raise ValueError(f"Non-finite CPI context value at {oid}")
        value_lookup = {}
        mask_lookup = {}
        for row in cov.itertuples(index=False):
            key = (oid, str(row.feature_name), str(row.context_reference_month))
            is_masked = bool(row.masked)
            value_lookup[key] = np.float32(0.0 if is_masked else row.value)
            mask_lookup[key] = is_masked
        cov_values = np.stack(
            [[value_lookup[(oid, feat, month)] for month in months] for feat in FEATURES], axis=0
        ).astype(np.float32, copy=False)
        cov_mask = np.stack(
            [[mask_lookup[(oid, feat, month)] for month in months] for feat in FEATURES], axis=0
        ).astype(np.bool_, copy=False)
        # A later captured vintage may replace an observed Brent cell only for a scored B3 origin.
        if oid in set(scored.origin_id.astype(str)):
            for fi, feat in enumerate(FEATURES):
                for ti, month in enumerate(months):
                    key = (oid, feat, month)
                    if key in rec_map:
                        if cov_mask[fi, ti]:
                            raise ValueError(f"Vintage map points to a masked cell: {key}")
                        cov_values[fi, ti] = rec_map[key]
                        cov_mask[fi, ti] = False
        examples.append(
            {
                "origin_id": oid,
                "origin_month": str(origin.origin_month),
                "origin_timestamp_utc": str(origin.origin_timestamp_utc),
                "months": months,
                "target": target_values,
                "strict_cov": np.stack(
                    [[value_lookup[(oid, feat, month)] for month in months] for feat in FEATURES], axis=0
                ).astype(np.float32, copy=False),
                "strict_mask": np.stack(
                    [[mask_lookup[(oid, feat, month)] for month in months] for feat in FEATURES], axis=0
                ).astype(np.bool_, copy=False),
                "backfill_cov": cov_values,
                "backfill_mask": cov_mask,
            }
        )
    return examples, rec, ref, scored


def run_variant(model, examples, variant: str, batch_size: int, device: str):
    outputs = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        max_len = max(len(x["target"]) for x in batch)
        target = np.zeros((len(batch), 1, max_len), dtype=np.float32)
        cov = np.zeros((len(batch), len(FEATURES), max_len), dtype=np.float32)
        cov_mask = np.ones((len(batch), len(FEATURES), max_len), dtype=np.bool_)
        global_mask = np.ones((len(batch), max_len), dtype=np.bool_)
        for i, item in enumerate(batch):
            n = len(item["target"])
            sl = slice(max_len - n, max_len)
            target[i, 0, sl] = item["target"]
            if variant == "strict_replay":
                cov[i, :, sl] = item["strict_cov"]
                cov_mask[i, :, sl] = item["strict_mask"]
            elif variant == "latest_archived_brent":
                cov[i, :, sl] = item["backfill_cov"]
                cov_mask[i, :, sl] = item["backfill_mask"]
            else:
                raise ValueError(variant)
            global_mask[i, sl] = False
        with torch.inference_mode():
            raw = model.model.decode(
                target=torch.from_numpy(target).to(device),
                horizon=12,
                past_only_covariates=torch.from_numpy(cov).to(device),
                past_only_mask=torch.from_numpy(cov_mask).to(device),
                mask=torch.from_numpy(global_mask).to(device),
            )
        q = raw[:, 0, :12, :].detach().float().cpu().numpy()
        if q.shape != (len(batch), 12, 9) or not np.isfinite(q).all():
            raise RuntimeError(f"Invalid output for {variant}: shape={q.shape}")
        q = np.sort(q, axis=-1)
        for item, path in zip(batch, q):
            for h in range(1, 13):
                row = {
                    "variant": variant,
                    "origin_id": item["origin_id"],
                    "origin_month": item["origin_month"],
                    "origin_timestamp_utc": item["origin_timestamp_utc"],
                    "horizon_months": h,
                    "point_forecast_q50": float(path[h - 1, 4]),
                }
                row.update({f"q{qn}": float(path[h - 1, qi]) for qi, qn in enumerate(Q_NAMES)})
                outputs.append(row)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--q50-tolerance-pp", type=float, default=0.002)
    args = ap.parse_args()
    require_new_directory(args.output_dir)
    input_dir = args.input_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint = checkpoint_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {checkpoint_sha}")

    # Verify every immutable input file before loading tensors.
    input_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.name != "files_sha256.json")
    input_hashes = {p.name: sha256(p) for p in input_files}
    file_manifest_path = input_dir / "files_sha256.json"
    if file_manifest_path.exists():
        frozen = json.loads(file_manifest_path.read_text(encoding="utf-8"))
        if frozen != input_hashes:
            raise RuntimeError("Input file hash manifest mismatch")

    examples, recovery, reference, scored = read_inputs(input_dir)
    expected_keys = {(str(r.origin_id), int(r.horizon_months)) for r in reference.itertuples(index=False)}
    if len(expected_keys) != len(reference):
        raise RuntimeError("Archived B3 q50 keys are not unique")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this frozen inference replay")
    # The reference execution was performed on an A800. Permit another CUDA
    # device for independent replay; the pinned checkpoint, inference API,
    # inputs, deterministic settings, and q50 parity check remain enforced.
    try:
        selected_device = torch.device(args.device)
        if selected_device.type != "cuda":
            raise RuntimeError("This runner requires a CUDA device")
        actual_gpu = torch.cuda.get_device_name(selected_device)
    except (RuntimeError, AssertionError, IndexError) as exc:
        raise RuntimeError(f"Requested CUDA device is unavailable: {args.device}") from exc
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
    import timesfm3

    installed_source_commit = verify_official_source_commit(Path(timesfm3.__file__).resolve())

    t_load = time.perf_counter()
    model = TimesFM3Forecaster.from_pretrained(
        str(checkpoint_dir), device=args.device, per_core_batch_size=args.batch_size, local_files_only=True
    )
    load_seconds = time.perf_counter() - t_load
    if tuple(model.config.quantiles) != (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        raise RuntimeError(f"Unexpected quantiles: {model.config.quantiles}")
    if model.config.quantiles[model.config.median_quantile_index] != 0.5:
        raise RuntimeError("The configured median quantile is not q50")

    t_inf = time.perf_counter()
    strict_rows = run_variant(model, examples, "strict_replay", args.batch_size, args.device)
    backfill_rows = run_variant(model, examples, "latest_archived_brent", args.batch_size, args.device)
    inference_seconds = time.perf_counter() - t_inf
    all_rows = strict_rows + backfill_rows
    forecast_path = args.output_dir / "forecast_predictions.csv"
    pd.DataFrame(all_rows).to_csv(forecast_path, index=False, encoding="utf-8", lineterminator="\n")

    strict = pd.DataFrame(strict_rows)
    ref = reference.copy()
    ref["horizon_months"] = ref.horizon_months.astype(int)
    joined = strict.merge(ref, on=["origin_id", "origin_month", "origin_timestamp_utc", "horizon_months"],
                          how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise RuntimeError("Strict replay origin/horizon keys differ from archived B3 q50 keys")
    q50_max_abs = float(np.max(np.abs(joined.point_forecast_q50.to_numpy() - joined.archived_q50.to_numpy())))
    q50_mean_abs = float(np.mean(np.abs(joined.point_forecast_q50.to_numpy() - joined.archived_q50.to_numpy())))
    q50_within = q50_max_abs <= args.q50_tolerance_pp

    # Verify full origin-horizon coverage for each originally scored horizon.
    expected_scored = {(str(r.origin_id), int(r.horizon_months)) for r in scored.itertuples(index=False)}
    produced = {(str(r.origin_id), int(r.horizon_months)) for r in strict.itertuples(index=False)}
    if not expected_scored.issubset(produced):
        raise RuntimeError("Scored origin-horizon keys are missing from inference output")

    package_env = json.loads((input_dir / "environment.json").read_text(encoding="utf-8"))
    metadata = {
        "status": "STRICT_REPLAY_WITHIN_PREDECLARED_Q50_TOLERANCE" if q50_within else "STRICT_REPLAY_OUTSIDE_TOLERANCE",
        "cohort": "China CPI unified 2018-2026 retrospective rolling evaluation",
        "variants": ["strict_replay", "latest_archived_brent"],
        "forecast_rows": len(all_rows),
        "batch_origin_count": len(examples),
        "scored_origin_horizon_count_by_horizon": {
            str(h): int(n) for h, n in scored.groupby("horizon_months").size().items()
        },
        "observed_brent_cells_revised": len(recovery),
        "P1_accessed": False,
        "scored_target_outcomes_read": False,
        "historical_cpi_context_values_used_as_model_input": True,
        "weights_sha256": checkpoint_sha,
        "official_source_commit": EXPECTED_OFFICIAL_SOURCE_COMMIT,
        "runtime_source_commit_verified": installed_source_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": actual_gpu,
        "reference_gpu": "NVIDIA A800-SXM4-80GB",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": args.device,
        "batch_size": args.batch_size,
        "features_order": list(FEATURES),
        "quantiles": list(model.config.quantiles),
        "point_forecast": "q50",
        "strict_replay_q50_max_abs_difference_pp": q50_max_abs,
        "strict_replay_q50_mean_abs_difference_pp": q50_mean_abs,
        "strict_replay_q50_tolerance_pp": args.q50_tolerance_pp,
        "strict_replay_within_tolerance": q50_within,
        "checkpoint_load_seconds": load_seconds,
        "two_variant_inference_seconds": inference_seconds,
        "input_file_sha256": input_hashes,
        "forecast_predictions_sha256": sha256(forecast_path),
        "forecast_predictions_path": forecast_path.name,
        "scoring_scope": "not performed by this runner; outcome values are not read",
        "retrospective_status": "generated after all targets were released and earlier results were known",
        "archived_environment_reference": package_env,
    }
    (args.output_dir / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in sorted(args.output_dir.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"),
        encoding="ascii",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if not q50_within:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
