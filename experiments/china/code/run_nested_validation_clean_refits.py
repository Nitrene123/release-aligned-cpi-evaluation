#!/usr/bin/env python3
"""Re-fit only rolling correction heads with a target-disjoint inner split.

This consumes the original target-free run's sealed training packets and frozen
B3 forecasts. It does not rerun TimesFM-3, statistical baselines, or scoring.
The original run remains immutable; every output is written to a new run folder.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import time
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
EXP = PROJECT / "reproducibility/matched_origin_2018_2026_unified_v1"
RESULTS = EXP / "results/complete_run_v1"
DEFAULT_RUN = RESULTS / "runs/run_20261030_unified_full_v2"
SOURCE_DATA = RESULTS / "source_data"
HORIZONS = (1, 3, 6, 12)
SEEDS = (417, 42, 123, 2027, 3407)
KINDS = ("BEAH", "Direct", "GateOnly")
POLICY = (
    "inner training origin timestamp and every used label release strictly precede "
    "the first h=3 validation-origin timestamp; no inner-training target-reference "
    "month occurs among the 12 h=3 validation target months; choose epochs on h=3; "
    "then refit using all labels eligible strictly before the outer origin"
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8"))


def stamp(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value}")
    return dt.astimezone(timezone.utc)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output: {path.name}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_original_bundle(run_dir: Path) -> dict:
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "FORECASTS_SEALED_NO_TARGET_SCORING":
        raise RuntimeError("source run is not a sealed unscored forecast run")
    if state.get("scoring_targets_loaded") is not False or state.get("P1_accessed") is not False:
        raise RuntimeError("source run does not declare target/P1 isolation")
    sealed = json.loads((run_dir / "sealed_manifest.json").read_text(encoding="utf-8"))
    files = sealed.get("files", {})
    if not files:
        raise RuntimeError("source run has no sealed file manifest")
    mismatch = []
    for rel, expected in files.items():
        path = run_dir / rel
        if not path.is_file() or sha(path) != expected:
            mismatch.append(rel)
    if mismatch:
        raise RuntimeError(f"source run seal mismatch: {mismatch[:5]}")
    source_manifest = json.loads((SOURCE_DATA / "source_bundle_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "SOURCE_BUNDLE_VERIFIED_NO_FORECASTS_NO_SCORE_TARGETS":
        raise RuntimeError("source bundle is not the frozen target-free bundle")
    normalized = source_manifest.get("normalized_files", {})
    if not normalized:
        raise RuntimeError("source bundle manifest has no normalized file hash inventory")
    for rel, record in normalized.items():
        expected = record.get("sha256") if isinstance(record, dict) else record
        path = SOURCE_DATA / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"source bundle file hash mismatch: {rel}")
    return {"source_run_files_verified": len(files),
            "source_bundle_files_verified": len(normalized),
            "source_run_manifest_sha256": sha(run_dir / "sealed_manifest.json"),
            "source_bundle_manifest_sha256": sha(SOURCE_DATA / "source_bundle_manifest.json")}


def load_source_metadata():
    origin_rows = [r for r in read_csv(SOURCE_DATA / "unified_origins.csv") if r["country"] == "CN"]
    origins = {}
    for r in origin_rows:
        release = stamp(r["origin_timestamp_utc"])
        origins[r["origin_id"]] = {
            "origin_month": r["origin_month"],
            "origin_release_timestamp": release,
            "origin_timestamp": release + timedelta(minutes=1),
        }
    label_rows = [r for r in read_csv(SOURCE_DATA / "unified_label_eligibility.csv") if r["country"] == "CN"]
    label_meta = {(r["origin_id"], int(r["horizon"])): r for r in label_rows}
    if len(label_meta) != len(label_rows):
        raise RuntimeError("duplicate label eligibility key")
    return origins, label_meta


def read_npz_member(zf: zipfile.ZipFile, member: str):
    with np.load(io.BytesIO(zf.read(member)), allow_pickle=False) as pack:
        return {k: pack[k].copy() for k in pack.files}


def clean_packet(raw: dict, origins: dict, label_meta: dict, outer_id: str,
                 outer_time: datetime):
    val_ids = raw["validation_origin_ids"].astype(str).tolist()
    if len(val_ids) != 12:
        raise RuntimeError(f"expected 12 validation origins at {outer_id}")
    val_start = origins[val_ids[0]]["origin_timestamp"]
    validation_target_to_value = {}
    for i, oid in enumerate(val_ids):
        key = (oid, 3)
        if key not in label_meta or not bool(raw["validation_valid"][i, 1]):
            raise RuntimeError(f"missing eligible h=3 validation label: {outer_id}/{oid}")
        meta = label_meta[key]
        target_month = meta["target_reference_month"][:7]
        y, m = map(int, origins[oid]["origin_month"].split("-"))
        idx = y * 12 + (m - 1) + 3
        expected_target = f"{idx // 12:04d}-{idx % 12 + 1:02d}"
        if target_month != expected_target:
            raise RuntimeError(f"validation target map mismatch {outer_id}/{oid}: {target_month} != {expected_target}")
        if stamp(meta["label_release_timestamp"]) >= outer_time:
            raise RuntimeError(f"validation target unavailable by outer origin: {outer_id}/{oid}")
        if target_month in validation_target_to_value:
            raise RuntimeError(f"duplicate validation target month at {outer_id}: {target_month}")
        validation_target_to_value[target_month] = float(raw["validation_target"][i, 1])
    validation_months = set(validation_target_to_value)

    # Keep only inner-training rows with at least one label that passes both
    # the inner-window release cutoff and the cross-horizon target disjointness.
    train_ids = raw["train_origin_ids"].astype(str)
    keep_cells = np.zeros_like(raw["train_valid"], dtype=np.bool_)
    counts = {"candidate_train_label_cells": int(raw["train_valid"].sum()),
              "dropped_release_at_or_after_validation_start": 0,
              "dropped_validation_target_month_overlap": 0,
              "dropped_both_conditions": 0,
              "exact_value_matches_on_overlapping_cells": 0,
              "overlapping_training_label_cells": 0}
    dropped_seen = set()
    overlap_train_months = set()
    train_months = set()
    for i, oid in enumerate(train_ids):
        origin = origins[str(oid)]
        if origin["origin_timestamp"] >= val_start:
            raise RuntimeError(f"inner training origin is not before validation start: {outer_id}/{oid}")
        for j, h in enumerate(HORIZONS):
            if not raw["train_valid"][i, j]:
                continue
            key = (str(oid), h)
            meta = label_meta.get(key)
            if meta is None:
                raise RuntimeError(f"training packet label missing from eligibility ledger: {outer_id}/{key}")
            target_month = meta["target_reference_month"][:7]
            release = stamp(meta["label_release_timestamp"])
            y, m = map(int, origin["origin_month"].split("-"))
            idx = y * 12 + (m - 1) + h
            expected_target = f"{idx // 12:04d}-{idx % 12 + 1:02d}"
            if target_month != expected_target:
                raise RuntimeError(f"training target map mismatch {outer_id}/{key}")
            after_cutoff = release >= val_start
            overlaps = target_month in validation_months
            if overlaps:
                counts["overlapping_training_label_cells"] += 1
                overlap_train_months.add(target_month)
                val_value = validation_target_to_value[target_month]
                if np.float32(raw["train_target"][i, j]).tobytes() == np.float32(val_value).tobytes():
                    counts["exact_value_matches_on_overlapping_cells"] += 1
            if after_cutoff and overlaps:
                counts["dropped_both_conditions"] += 1
            elif after_cutoff:
                counts["dropped_release_at_or_after_validation_start"] += 1
            elif overlaps:
                counts["dropped_validation_target_month_overlap"] += 1
            if after_cutoff or overlaps:
                dropped_seen.add((str(oid), h))
                continue
            keep_cells[i, j] = True
            train_months.add(target_month)

    keep_rows = keep_cells.any(axis=1)
    clean = {}
    for key, value in raw.items():
        if key.startswith("train_") and key != "train_origin_ids":
            clean[key] = value[keep_rows].copy()
        else:
            clean[key] = value.copy()
    clean["train_origin_ids"] = train_ids[keep_rows].copy()
    clean["train_valid"] = keep_cells[keep_rows].copy()
    clean["train_target"][~clean["train_valid"]] = 0

    # Epoch selection is explicitly h=3 only; no other validation labels are
    # placed in the validation objective packet.
    clean["validation_valid"][:] = False
    clean["validation_valid"][:, 1] = raw["validation_valid"][:, 1]
    clean["validation_target"][:] = 0
    clean["validation_target"][:, 1] = raw["validation_target"][:, 1]

    kept_months = {label_meta[(str(oid), h)]["target_reference_month"][:7]
                   for i, oid in enumerate(clean["train_origin_ids"])
                   for j, h in enumerate(HORIZONS) if clean["train_valid"][i, j]}
    overlap_after = kept_months & validation_months
    if overlap_after:
        raise RuntimeError(f"clean split still overlaps target months: {outer_id}/{sorted(overlap_after)}")
    if any(stamp(label_meta[(str(oid), h)]["label_release_timestamp"]) >= val_start
           for i, oid in enumerate(clean["train_origin_ids"])
           for j, h in enumerate(HORIZONS) if clean["train_valid"][i, j]):
        raise RuntimeError(f"clean split retains post-window label at {outer_id}")
    if int(clean["validation_valid"][:, 1].sum()) != 12 or clean["validation_valid"][:, [0, 2, 3]].any():
        raise RuntimeError(f"validation mask is not exactly the 12 h=3 targets at {outer_id}")
    counts.update({
        "retained_inner_training_label_cells": int(clean["train_valid"].sum()),
        "retained_inner_training_origins": int(len(clean["train_origin_ids"])),
        "retained_inner_training_target_months": len(kept_months),
        "validation_target_months": ";".join(sorted(validation_months)),
        "overlapping_training_target_months_before_filter": ";".join(sorted(overlap_train_months)),
        "train_validation_overlap_after_filter": 0,
        "validation_start_utc": val_start.isoformat(),
        "outer_origin_timestamp_utc": outer_time.isoformat(),
    })
    return clean, counts, val_start


def load_modules(source_dir: Path):
    heads_path = source_dir / "heads.py"
    training_path = source_dir / "training.py"
    beah_path = source_dir / "beah.py"
    heads = ModuleType("nested_clean_pinned_heads")
    heads.__file__ = str(heads_path)
    tree = ast.parse(heads_path.read_bytes(), filename=str(heads_path))
    changed = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "REFERENCE_SOURCE" for t in node.targets
        ):
            node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                  args=[ast.Constant(str(beah_path))], keywords=[])
            changed += 1
    if changed != 1:
        raise RuntimeError("could not bind exact pinned base-effect source")
    ast.fix_missing_locations(tree)
    sys.modules[heads.__name__] = heads
    exec(compile(tree, str(heads_path), "exec"), heads.__dict__)
    training = ModuleType("nested_clean_pinned_training")
    training.__file__ = str(training_path)
    sys.modules[training.__name__] = training
    exec(compile(training_path.read_bytes(), str(training_path), "exec"), training.__dict__)
    return heads, training


class GateOnly(torch.nn.Module):
    """Frozen fixed candidate and exactly four learned horizon gates."""
    def __init__(self, heads):
        super().__init__()
        self.raw_gate = torch.nn.Parameter(torch.zeros(4))
        self.heads = heads

    def forward(self, anchor, monthly, b3, month):
        candidate = self.heads.fixed_candidate(anchor, monthly)
        return {"yoy_pct": self.heads.blend(b3, torch.log1p(candidate / 100), self.raw_gate),
                "candidate_yoy_pct": candidate, "gates": self.raw_gate.clamp(0, 1)}


def model_create(heads, kind: str, seed: int):
    if kind in ("BEAH", "Direct"):
        return heads.create(kind, seed)
    if kind == "GateOnly":
        torch.manual_seed(seed)
        return GateOnly(heads)
    raise ValueError(kind)


def fingerprint(model) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensors(pack: dict, prefix: str, device: str):
    x = tuple(torch.as_tensor(pack[f"{prefix}_{key}"], device=device)
              for key in ("anchor", "monthly", "b3", "month"))
    y = torch.as_tensor(pack[f"{prefix}_target"], device=device)
    valid = torch.as_tensor(pack[f"{prefix}_valid"], device=device)
    return x, y, valid


def run(args):
    run_dir = args.run_dir.resolve()
    output_root = args.output_root.resolve()
    source_dir = run_dir / "source"
    packets_path = run_dir / "rolling_training_packets.zip"
    state_check = verify_original_bundle(run_dir)
    origins, label_meta = load_source_metadata()
    if not packets_path.is_file():
        raise FileNotFoundError(packets_path)

    all_fit_rows, trace_rows, audit_rows, forecast_rows = [], [], [], []
    original_packets = zipfile.ZipFile(packets_path, "r")
    members = sorted(x for x in original_packets.namelist() if x.endswith(".npz"))
    if len(members) != 110:
        raise RuntimeError(f"expected 110 original outer-origin packets, found {len(members)}")
    planned = {(r["origin_id"], int(r["horizon_months"])): r
               for r in read_csv(run_dir / "planned_origin_horizon_keys.csv")}
    b3_rows = read_csv(run_dir / "b3_internal_horizon_forecasts.csv")
    b3_map = {(r["origin_id"], int(r["horizon_months"])): float(r["prediction"])
              for r in b3_rows if r["model"] == "B3 multivariate"}
    old_forecasts = read_csv(run_dir / "model_forecasts_unscored.csv")
    base_hash = {}
    for r in old_forecasts:
        if r["model"] in ("Fixed candidate", "B3 multivariate"):
            base_hash.setdefault(r["origin_id"], {})[r["model"]] = r["input_hash"]
    panel_rows = read_csv(SOURCE_DATA / "unified_bace_panel.csv")
    panel = {r["origin_id"]: r for r in panel_rows if r["country"] == "CN"}
    input_index = {r["origin_id"]: r for r in read_csv(run_dir / "evaluation_origin_input_index.csv")}
    ready_origins = {m.removeprefix("packets/").removesuffix(".npz") for m in members}
    if len(ready_origins) != 110:
        raise RuntimeError("outer packet origin inventory is not unique")
    missing = sorted(oid for oid in ready_origins
                     if oid not in panel or oid not in origins or
                     any((oid, h) not in b3_map for h in HORIZONS))
    if missing:
        raise RuntimeError(f"frozen input or B3 path missing for outer origins: {missing[:5]}")
    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("GPU is required for the frozen adapter refit protocol")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    heads, training = load_modules(source_dir)
    if tuple(training.SEEDS) != SEEDS or tuple(heads.HORIZONS) != HORIZONS:
        raise RuntimeError("pinned training seeds/horizons differ from frozen protocol")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    out = output_root / f"{'preflight_' if args.preflight_only else 'run_'}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    state = {"status": "STARTED", "started_at_utc": started,
             "source_run_seal": state_check, "source_run_id": run_dir.name,
             "nested_validation_policy": POLICY, "outer_origin_count": 110,
             "models": list(KINDS), "seeds": list(SEEDS),
             "planned_fit_invocations": 0 if args.preflight_only else 110 * len(KINDS) * len(SEEDS),
             "scoring_targets_loaded": False, "target_values_scored": False,
             "P1_accessed": False, "B3_recomputed": False,
             "statistics_recomputed": False}
    (out / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    try:
        packet_buffer = io.BytesIO()
        new_zip = zipfile.ZipFile(packet_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6)
        for member in members:
            outer_id = member.removeprefix("packets/").removesuffix(".npz")
            raw = read_npz_member(original_packets, member)
            outer_time = origins[outer_id]["origin_timestamp"]
            clean, audit, val_start = clean_packet(raw, origins, label_meta, outer_id, outer_time)
            audit_rows.append({"outer_origin_id": outer_id, **audit})
            npbuf = io.BytesIO()
            np.savez_compressed(npbuf, **clean)
            new_zip.writestr(member, npbuf.getvalue())

            if args.preflight_only:
                continue

            pack_hash = canonical_hash({
                "outer_origin_id": outer_id,
                "outer_origin_timestamp_utc": outer_time.isoformat(),
                "validation_start_utc": val_start.isoformat(),
                "nested_validation_policy": POLICY,
                "train_origin_ids": clean["train_origin_ids"].astype(str).tolist(),
                "validation_origin_ids": clean["validation_origin_ids"].astype(str).tolist(),
                "final_origin_ids": clean["final_origin_ids"].astype(str).tolist(),
                "train": [clean[f"train_{k}"].tolist() for k in ("anchor", "monthly", "b3", "month", "target", "valid")],
                "validation": [clean[f"validation_{k}"].tolist() for k in ("anchor", "monthly", "b3", "month", "target", "valid")],
                "final": [clean[f"final_{k}"].tolist() for k in ("anchor", "monthly", "b3", "month", "target", "valid")],
            })
            train = tensors(clean, "train", device)
            validation = tensors(clean, "validation", device)
            final = tensors(clean, "final", device)
            initial_audit = {"outer_origin_id": outer_id, "training_packet_hash": pack_hash,
                             "eligible_train_origins": int(len(clean["train_origin_ids"])),
                             "inner_validation_h3_origins": int(len(clean["validation_origin_ids"])),
                             "final_fit_origins": int(len(clean["final_origin_ids"])),
                             "eligible_final_label_cells": int(clean["final_valid"].sum()),
                             "inner_train_label_cells": int(clean["train_valid"].sum()),
                             "validation_label_cells": int(clean["validation_valid"].sum()),
                             "validation_start_utc": val_start.isoformat()}
            for kind in KINDS:
                for seed in SEEDS:
                    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
                    select_model = model_create(heads, kind, seed).to(device)
                    init_sha = fingerprint(select_model)
                    select_start = time.perf_counter()
                    selection = training.select_epochs(select_model, train[:3], validation[:3])
                    select_seconds = time.perf_counter() - select_start
                    for point in selection["trace"]:
                        trace_rows.append({"outer_origin_id": outer_id, "model": kind, "seed": seed,
                                           "epoch": int(point["epoch"]),
                                           "validation_h3_mae_pp": float(point["h3_mae"])})
                    selected_epochs = int(selection["selected_epoch"])
                    final_model = model_create(heads, kind, seed).to(device)
                    if fingerprint(final_model) != init_sha:
                        raise RuntimeError(f"initialization mismatch {outer_id}/{kind}/{seed}")
                    optimizer = training.optimizer_for(final_model)
                    fit_start = time.perf_counter()
                    for _ in range(selected_epochs):
                        training.step(final_model, optimizer, final[0], final[1], final[2])
                    fit_seconds = time.perf_counter() - fit_start
                    final_model.eval()
                    final_sha = fingerprint(final_model)
                    n_params = sum(p.numel() for p in final_model.parameters())
                    expected = 4 if kind == "GateOnly" else 688
                    if n_params != expected:
                        raise RuntimeError(f"parameter count mismatch for {kind}: {n_params}")

                    row = panel[outer_id]
                    anchor = torch.tensor([float(row["anchor_yoy_pct"])], device=device, dtype=torch.float32)
                    monthly = torch.tensor([[float(row[f"mom_{i:02d}_pct"]) for i in range(12)]],
                                            device=device, dtype=torch.float32)
                    b3 = torch.tensor([[b3_map[(outer_id, h)] for h in HORIZONS]],
                                      device=device, dtype=torch.float32)
                    month = torch.tensor([int(row["anchor_month"])], device=device, dtype=torch.int64)
                    inf_start = time.perf_counter()
                    with torch.inference_mode():
                        result = final_model(anchor, monthly, b3, month)
                    inf_seconds = time.perf_counter() - inf_start
                    pred = result["yoy_pct"].detach().cpu().numpy()[0]
                    if not np.isfinite(pred).all() or fingerprint(final_model) != final_sha:
                        raise RuntimeError(f"invalid or mutated final model at {outer_id}/{kind}/{seed}")
                    gates = result["gates"].detach().cpu().numpy()
                    if outer_id not in base_hash or set(base_hash[outer_id]) != {"Fixed candidate", "B3 multivariate"}:
                        raise RuntimeError(f"frozen input hashes missing at {outer_id}")
                    eval_input_hash = canonical_hash({
                        "candidate_input_hash": base_hash[outer_id]["Fixed candidate"],
                        "b3_input_hash": base_hash[outer_id]["B3 multivariate"],
                        "training_packet_hash": pack_hash})
                    for j, h in enumerate(HORIZONS):
                        key = (outer_id, h)
                        if key not in planned:
                            continue
                        key_row = planned[key]
                        forecast_rows.append({
                            "model": {"BEAH": "BEAH rolling", "Direct": "Direct rolling",
                                      "GateOnly": "Fixed candidate + gate"}[kind],
                            "seed": seed, **key_row, "prediction": float(pred[j]),
                            "status": "FORECAST_COMPLETE", "input_hash": eval_input_hash,
                            "training_packet_hash": pack_hash, "selected_epochs": selected_epochs,
                            "parameter_count": n_params, "parameter_sha256": final_sha,
                            "gate_value": float(gates[j] if np.ndim(gates) else gates)})
                    fit_record = {
                        **initial_audit, "model": kind, "seed": seed,
                        "status": "FIT_AND_FORECAST_COMPLETE",
                        "selection_train_origins": int(len(clean["train_origin_ids"])),
                        "selection_train_label_cells": int(clean["train_valid"].sum()),
                        "selection_validation_h3_origins": 12,
                        "selection_validation_label_cells": 12,
                        "selection_validation_target_months": audit["validation_target_months"],
                        "selection_train_target_months": audit["retained_inner_training_target_months"],
                        "selected_epochs": selected_epochs,
                        "selection_optimizer_steps": int(selection["optimizer_steps"]),
                        "selection_wall_seconds": select_seconds,
                        "best_validation_h3_mae_pp": float(selection["best_h3_mae"]),
                        "final_fit_wall_seconds": fit_seconds,
                        "inference_wall_seconds": inf_seconds,
                        "initial_parameters_sha256": init_sha,
                        "final_parameters_sha256": final_sha,
                        "parameter_count": n_params,
                        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
                        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device.startswith("cuda") else 0,
                        "training_packet_hash": pack_hash,
                        "status_detail": "clean inner target-disjoint epoch selection; final refit on all labels released before outer origin"}
                    all_fit_rows.append(fit_record)
                    del select_model, final_model, optimizer
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                if len(all_fit_rows) % 75 == 0:
                    print(f"clean rolling head fits complete: {len(all_fit_rows)} / 1650; origin={outer_id}", flush=True)

        new_zip.close()
        write_csv(out / "inner_validation_target_audit.csv", audit_rows)
        (out / "rolling_training_packets_nested_clean.zip").write_bytes(packet_buffer.getvalue())
        if args.preflight_only:
            state.update(status="PREFLIGHT_PASSED_NO_GPU_FITS", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                         outer_origin_count=len(audit_rows), total_validation_target_months=sum(12 for _ in audit_rows),
                         train_validation_target_month_overlap_after_filter=0,
                         labels_released_at_or_after_validation_start_after_filter=0,
                         validation_target_month_occurrences_with_train_overlap_before_filter=sum(
                             len(x["overlapping_training_target_months_before_filter"].split(";"))
                             if x["overlapping_training_target_months_before_filter"] else 0
                             for x in audit_rows),
                         overlapping_training_label_cells_before_filter=sum(int(x["overlapping_training_label_cells"]) for x in audit_rows),
                         exact_value_matches_before_filter=sum(int(x["exact_value_matches_on_overlapping_cells"]) for x in audit_rows),
                         scoring_targets_loaded=False, P1_accessed=False, fits_run=0,
                         elapsed_seconds=time.perf_counter() - t0)
        else:
            if len(all_fit_rows) != 110 * 3 * 5:
                raise RuntimeError(f"expected 1,650 clean fits, got {len(all_fit_rows)}")
            if not forecast_rows:
                raise RuntimeError("no forecasts were produced")
            write_csv(out / "adapter_fit_manifest.csv", all_fit_rows)
            write_csv(out / "adapter_validation_traces.csv", trace_rows)
            write_csv(out / "corrected_head_forecasts_unscored.csv", forecast_rows)
            resource = {
                "fit_invocations": len(all_fit_rows),
                "fit_and_prediction_wall_seconds": sum(float(r["selection_wall_seconds"]) +
                                                        float(r["final_fit_wall_seconds"]) +
                                                        float(r["inference_wall_seconds"]) for r in all_fit_rows),
                "mean_fit_and_prediction_wall_seconds_per_invocation": float(np.mean([
                    float(r["selection_wall_seconds"]) + float(r["final_fit_wall_seconds"]) +
                    float(r["inference_wall_seconds"]) for r in all_fit_rows])),
                "max_gpu_peak_allocated_bytes": max(int(r["gpu_peak_allocated_bytes"]) for r in all_fit_rows),
                "max_gpu_peak_reserved_bytes": max(int(r["gpu_peak_reserved_bytes"]) for r in all_fit_rows),
                "device": torch.cuda.get_device_name(torch.cuda.current_device()) if device.startswith("cuda") else device,
                "torch": torch.__version__, "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__,
                "models": {kind: {"fit_count": sum(r["model"] == kind for r in all_fit_rows),
                                  "parameter_count": 4 if kind == "GateOnly" else 688} for kind in KINDS},
                "note": "B3 and statistical predictions reused byte-for-byte from the sealed source run; no backbone inference was performed"}
            (out / "resource_metrics.json").write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")
            environment = {
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "torch": torch.__version__,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "device": torch.cuda.get_device_name(torch.cuda.current_device()) if device.startswith("cuda") else device,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "deterministic_algorithms": True, "deterministic_warn_only": True,
                "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "environment_note": "This run executes the configured correction-head refit procedure; baseline forecasts and model checkpoints are supplied separately.",
            }
            (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
            write_csv(out / "adapter_fit_manifest.csv", all_fit_rows)
            state.update(status="CORRECTED_HEAD_FORECASTS_SEALED_NO_TARGET_SCORING",
                         completed_at_utc=datetime.now(timezone.utc).isoformat(),
                         outer_origin_count=len(audit_rows), fit_invocations=len(all_fit_rows),
                         corrected_forecast_rows=len(forecast_rows),
                         scoring_targets_loaded=False, P1_accessed=False, B3_recomputed=False,
                         statistics_recomputed=False, elapsed_seconds=time.perf_counter() - t0,
                         protocol_amendment="CLEAN_NESTED_VALIDATION_TARGET_DISJOINT_V1")
            (out / "source_hashes.json").write_text(json.dumps({
                "original_run_manifest_sha256": state_check["source_run_manifest_sha256"],
                "source_bundle_manifest_sha256": state_check["source_bundle_manifest_sha256"],
                "original_training_packets_sha256": sha(packets_path),
                "heads_py_sha256": sha(source_dir / "heads.py"),
                "training_py_sha256": sha(source_dir / "training.py"),
                "beah_py_sha256": sha(source_dir / "beah.py"),
                "corrected_runner_sha256": sha(Path(__file__)),
                "nested_validation_policy_sha256": sha_bytes(POLICY.encode("utf-8")),
            }, indent=2) + "\n", encoding="utf-8")
        (out / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                 if p.is_file() and p.name != "sealed_manifest.json"}
        (out / "sealed_manifest.json").write_text(json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": state["status"], "files": files,
            "target_values_scored": False, "P1_accessed": False}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": state["status"], "run_dir": str(out),
                          "elapsed_seconds": state["elapsed_seconds"],
                          "outer_origin_count": len(audit_rows),
                          "fit_invocations": state.get("fit_invocations", 0),
                          "corrected_forecast_rows": state.get("corrected_forecast_rows", 0),
                          "overlapping_training_label_cells_before_filter": state.get("overlapping_training_label_cells_before_filter",
                              sum(int(x["overlapping_training_label_cells"]) for x in audit_rows)),
                          "train_validation_target_month_overlap_after_filter": 0,
                          "scoring_targets_loaded": False, "P1_accessed": False}, indent=2), flush=True)
    except BaseException as exc:
        state.update(status="FAILED_NO_TARGET_SCORING", failed_at_utc=datetime.now(timezone.utc).isoformat(),
                     error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                     elapsed_seconds=time.perf_counter() - t0,
                     scoring_targets_loaded=False, P1_accessed=False)
        (out / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        original_packets.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-root", type=Path,
                        default=EXP / "results/nested_validation_clean_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true",
                        help="audit all 110 splits and write clean packets without training or scoring")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
