#!/usr/bin/env python3
"""Generate sealed CN 2018-2026 unified expanding-window forecasts; never score targets.

Run only on the pinned EconoFM_CPI server environment. All adapter fits are
re-created at each outer origin from labels whose first release precedes that
origin. The historical evaluation target values are not joined in this run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import resource
import subprocess
import shutil
import sys
import time
import traceback
import zipfile
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo
from portable_paths import project_root

import numpy as np
import pandas as pd
import torch

ROOT = project_root(__file__)
EXP = ROOT / "revision_experiments_v2/rolling_foundation_matched_v2"
DATA = EXP / "source_data"
PROTOCOL_ID = "CN_UNIFIED_EXPANDING_WINDOW_2018_2026_V1"
HORIZONS = (1, 3, 6, 12)
ALL_H = tuple(range(1, 13))
SEEDS = (417, 42, 123, 2027, 3407)
FEATURES = ("PPI_YOY_PCT", "M2_YOY_PCT", "BRENT_LOG_YOY_PCT", "USDCNY_LOG_YOY_PCT")
WEIGHT_SHA = "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"
SOURCE_COMMIT = "e51928e27119cb17bebc005be2696b75e0a9e688"
EXPECTED = {
    "data/processed/bace_panel_v1.csv": "536a6c689d6d95ac6d51e76bc5b9ba8c1585a95d28ecde45ad7d86c36c535b54",
    "results/timesfm3_multivariate_predictions_v1.csv": "42c1c64b408a11972b15c923b4597774fef95fa618d8ea7a8207e5e65d232dbf",
    "results/timesfm3_multivariate_run_metadata_v1.json": "d8ca5a04c73793c6a6f33db106c1f6cb520e2b32dd943cec372be90ebeda7cca",
    "data/processed/rolling_origin_horizon_manifest_v1.csv": "5bc23b4729a5a90f71b2e4f809a00fdbce1920bcc6f8a7899ec82e679d8a1a76",
    "data/processed/pit_covariate_inputs_v1.csv": "62d927fb613c66dbffdf1c25e873871e354b6aee2ee7004860b6f4309bf1b1be",
    "data/processed/china_cpi_first_release_panel_2012-01_2025-12.csv": "30679a9c9871c7a99473e38dcdcfe31883ba98216ca0b73604a7d1b2ffbc08f8",
    "scripts/run_timesfm3_cpi_multivariate_baseline.py": "4b52f066141bcc54286e6f53219df2fb9823b1c97644a4f5b74bb12a31961266",
    "stage5/g4_history_audit_v3/historical_origin_eligibility_ledger.csv": "2f83fbe1f832b0f56b0afc53fbf8821d78f8f3d8421aea380f62c06b03228d00",
    "stage5/g4_history_audit_v3/historical_label_eligibility_ledger.csv": "a7f8282201c5adec238ea1e1b0e9db8cfa22624174bbb072bbe701c44b3c1878",
    "stage5/g4_history_audit_v3/used_covariate_source_audit.csv": "c948ab266f68b850d91bd3ab7d7d31d10e31da174ac140ff7f88d10a56881399",
    "stage5/g1/heads.py": "276ab8c146f31487393364f8b251df312db8df257d93330266bf821f11706cbb",
    "stage5/g1/training.py": "4f61ecff149d89a287c8acc59adc34a324b47a84251e20beca624ad560e41bd8",
    "imports/base_effect_v1/BaseEffect_TimesFM3_MVP/beah.py": "afe828da757409e20c2b0cbac4e1b2882349d7510ff79a45d579ed383f3303f6",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def stamp(value: str, local: bool = False) -> datetime:
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        if not local:
            raise ValueError(f"naive timestamp without declared source timezone: {value}")
        dt = dt.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return dt.astimezone(timezone.utc)


def add_month(month: str, n: int) -> str:
    y, m = map(int, month[:7].split("-"))
    z = y * 12 + (m - 1) + n
    return f"{z // 12:04d}-{z % 12 + 1:02d}"


def canonical_hash(obj) -> str:
    return sha_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8"))


def load_frozen_b3_cache(path: Path, cache_file_sha: str, b3_items: list[dict]):
    quantile_names = [f"q{x}" for x in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
    cache = pd.read_csv(path, usecols=["country", "origin_id", "horizon_months",
                                       "point_forecast_q50", *quantile_names])
    cache = cache[cache.country == "CN"]
    wanted_origins = {x["origin_id"] for x in b3_items}
    quantiles = {}
    for row in cache.itertuples(index=False):
        if row.origin_id not in wanted_origins:
            continue
        key = (row.origin_id, int(row.horizon_months))
        if key in quantiles:
            raise RuntimeError(f"duplicate frozen B3 cache key: {key}")
        values = np.asarray([getattr(row, n) for n in quantile_names], dtype=np.float32)
        if not np.isfinite(values).all() or not np.array_equal(values, np.sort(values)):
            raise RuntimeError(f"invalid archived B3 quantiles: {key}")
        if np.float32(row.point_forecast_q50).tobytes() != values[4].tobytes():
            raise RuntimeError(f"archived B3 q50 field mismatch: {key}")
        quantiles[key] = values
    outputs = {}
    checks = []
    expected_keys = {(x["origin_id"], h) for x in b3_items for h in ALL_H}
    if set(quantiles) != expected_keys:
        raise RuntimeError(f"frozen B3 cache key mismatch: expected {len(expected_keys)}, found {len(quantiles)}")
    for item in b3_items:
        oid = item["origin_id"]
        rows = []
        for h in ALL_H:
            q = quantiles[(oid, h)]
            rows.append(q)
            checks.append({"origin_id": oid, "horizon_months": h,
                           "cache_source_sha256": cache_file_sha,
                           "point_forecast_q50": float(q[4]), "cache_row_present": True,
                           "quantiles_finite_and_ordered": True})
        outputs[oid] = np.stack(rows, axis=0)
    return outputs, checks


def load_pinned_heads():
    head_path = ROOT / "stage5/g1/heads.py"
    train_path = ROOT / "stage5/g1/training.py"
    reference = ROOT / "imports/base_effect_v1/BaseEffect_TimesFM3_MVP/beah.py"
    h = ModuleType("rolling_pinned_heads")
    h.__file__ = str(head_path)
    import ast
    tree = ast.parse(head_path.read_bytes(), filename=str(head_path))
    changed = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REFERENCE_SOURCE" for t in node.targets):
            node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(reference))], keywords=[])
            changed += 1
    if changed != 1:
        raise RuntimeError("could not bind the pinned local BEAH source exactly once")
    ast.fix_missing_locations(tree)
    sys.modules[h.__name__] = h
    exec(compile(tree, str(head_path), "exec"), h.__dict__)
    t = ModuleType("rolling_pinned_training")
    t.__file__ = str(train_path)
    sys.modules[t.__name__] = t
    exec(compile(train_path.read_bytes(), str(train_path), "exec"), t.__dict__)
    return h, t


class GateOnly(torch.nn.Module):
    """Frozen candidate + four learned horizon gates; exactly four parameters."""
    def __init__(self, heads):
        super().__init__()
        self.raw_gate = torch.nn.Parameter(torch.zeros(4))
        self.heads = heads

    def forward(self, anchor, monthly, b3, month):
        cand = self.heads.fixed_candidate(anchor, monthly)
        return {"yoy_pct": self.heads.blend(b3, torch.log1p(cand / 100), self.raw_gate),
                "candidate_yoy_pct": cand, "gates": self.raw_gate.clamp(0, 1)}


def model_create(heads, kind: str, seed: int):
    if kind in ("BEAH", "Direct"):
        return heads.create(kind, seed)
    if kind == "GateOnly":
        torch.manual_seed(seed)
        return GateOnly(heads)
    raise ValueError(kind)


def parameter_fingerprint(model) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_vector(model) -> np.ndarray:
    return np.concatenate([p.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).ravel()
                           for _, p in model.named_parameters()])


def make_packet(rows, eligible_by_origin, b3_map, device="cuda:0"):
    rows = sorted(rows, key=lambda r: (stamp(r["origin_timestamp"]), r["origin_id"]))
    n = len(rows)
    a = np.zeros(n, dtype=np.float32)
    m = np.zeros((n, 12), dtype=np.float32)
    b = np.zeros((n, 4), dtype=np.float32)
    mo = np.zeros(n, dtype=np.int64)
    y = np.zeros((n, 4), dtype=np.float32)
    valid = np.zeros((n, 4), dtype=np.bool_)
    ids = []
    for i, row in enumerate(rows):
        oid = row["origin_id"]
        ids.append(oid)
        a[i] = float(row["anchor_yoy_pct"])
        m[i] = [float(row[f"mom_{j:02d}_pct"]) for j in range(12)]
        b[i] = [float(b3_map[(oid, h)]) for h in HORIZONS]
        mo[i] = int(row["anchor_month"])
        for j, h in enumerate(HORIZONS):
            val = eligible_by_origin.get((oid, h))
            if val is not None:
                y[i, j] = val
                valid[i, j] = True
    if not np.isfinite(a).all() or not np.isfinite(m).all() or not np.isfinite(b).all() or not np.isfinite(y).all():
        raise ValueError("non-finite value in an eligible training packet")
    if not valid.any() or not (y[~valid] == 0).all():
        raise ValueError("empty or contaminated training packet")
    x = tuple(torch.as_tensor(v, device=device) for v in (a, m, b, mo))
    return x, torch.as_tensor(y, device=device), torch.as_tensor(valid, device=device), ids, (a, m, b, mo, y, valid)


def make_clean_inner_split(eligible, label_audit, origins, val_ids, outer_time):
    """Build a strictly time-isolated epoch-selection split.

    The validation set is the selected h=3 target months. Every label used to
    optimize the inner training model must have been released before the first
    validation-origin timestamp, come from a pre-window origin, and refer to a
    target month absent from the validation set. This blocks cross-horizon
    reuse of the same realized CPI observation.
    """
    if len(val_ids) != 12:
        raise ValueError(f"expected 12 h=3 validation origins, got {len(val_ids)}")
    val_start = stamp(origins[val_ids[0]]["origin_timestamp"])
    validation_target_months = set()
    validation_eligible = {}
    for oid in val_ids:
        key = (oid, 3)
        if key not in eligible or key not in label_audit:
            raise RuntimeError(f"validation h=3 label is not eligible: {oid}")
        meta = label_audit[key]
        target_month = meta["target_reference_month"][:7]
        if target_month != add_month(origins[oid]["origin_month"], 3):
            raise RuntimeError(f"validation target mapping mismatch: {oid}/h3")
        if stamp(meta["label_release_timestamp"]) >= outer_time:
            raise RuntimeError(f"validation label was unavailable at outer origin: {oid}/h3")
        if target_month in validation_target_months:
            raise RuntimeError(f"duplicate validation target month: {target_month}")
        validation_target_months.add(target_month)
        # Epoch selection scores only h=3. Keep all other validation horizons
        # physically masked so they cannot enter a later objective by accident.
        validation_eligible[key] = eligible[key]

    train_eligible = {}
    dropped_after_window = 0
    dropped_target_overlap = 0
    candidate_cells = 0
    for key, value in eligible.items():
        train_id, horizon = key
        meta = label_audit[key]
        train_time = stamp(origins[train_id]["origin_timestamp"])
        release = stamp(meta["label_release_timestamp"])
        target_month = meta["target_reference_month"][:7]
        if train_time >= val_start:
            continue
        candidate_cells += 1
        if release >= val_start:
            dropped_after_window += 1
            continue
        if target_month in validation_target_months:
            dropped_target_overlap += 1
            continue
        if target_month != add_month(origins[train_id]["origin_month"], horizon):
            raise RuntimeError(f"inner training target mapping mismatch: {train_id}/h{horizon}")
        train_eligible[key] = value

    train_ids = sorted({oid for oid, _ in train_eligible},
                       key=lambda oid: (stamp(origins[oid]["origin_timestamp"]), oid))
    train_target_months = {label_audit[key]["target_reference_month"][:7]
                           for key in train_eligible}
    overlap = train_target_months & validation_target_months
    if overlap:
        raise RuntimeError(f"inner train/validation target-month overlap: {sorted(overlap)}")
    if any(stamp(label_audit[key]["label_release_timestamp"]) >= val_start
           for key in train_eligible):
        raise RuntimeError("inner training contains a label released at/after validation start")
    if not train_ids:
        raise RuntimeError("clean inner training split is empty")
    return train_ids, train_eligible, validation_eligible, val_start, {
        "candidate_label_cells_before_release_and_target_filters": candidate_cells,
        "dropped_labels_released_at_or_after_validation_start": dropped_after_window,
        "dropped_labels_sharing_validation_target_month": dropped_target_overlap,
        "retained_inner_training_label_cells": len(train_eligible),
        "retained_inner_training_target_months": len(train_target_months),
        "validation_target_months": sorted(validation_target_months),
    }


def add_write_csv(path: Path, rows: list[dict]):
    write_csv(path, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
    ap.add_argument("--preflight-only", action="store_true",
                    help="verify frozen sources, origin clocks, statistical keys, and point-in-time inputs without loading TimesFM or fitting models")
    args = ap.parse_args()
    if not args.run_id.startswith("run_") or "/" in args.run_id or "\\" in args.run_id:
        raise ValueError("invalid run id")
    out = EXP / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = now()
    state = {"protocol_id": PROTOCOL_ID, "status": "STARTED_NO_SCORING",
             "started_at_utc": started, "scoring_targets_loaded": False, "P1_accessed": False,
             "prospective_queue_accessed": False, "target_scoring_authorized": False}
    (out / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    t_run = time.perf_counter()
    try:
        src = {k: ROOT / k for k in EXPECTED}
        source_sha = {k: sha(p) for k, p in src.items()}
        for rel, expected in EXPECTED.items():
            if expected and source_sha[rel] != expected:
                raise RuntimeError(f"source checksum changed: {rel} {source_sha[rel]}")
        bundle_manifest_path = DATA / "source_bundle_manifest.json"
        bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        if bundle_manifest.get("status") != "SOURCE_BUNDLE_VERIFIED_NO_FORECASTS_NO_SCORE_TARGETS":
            raise RuntimeError("unified point-in-time source bundle is not in the verified target-free state")
        if bundle_manifest.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("source bundle protocol ID mismatch")
        for rel, expected in bundle_manifest["historical_base_source_sha256"].items():
            actual = sha(ROOT / rel)
            if actual != expected:
                raise RuntimeError(f"hash-bound historical source changed: {rel}")
            source_sha[rel] = actual
        for name, record in bundle_manifest["normalized_files"].items():
            actual = sha(DATA / name)
            if actual != record["sha256"]:
                raise RuntimeError(f"normalized input bundle hash mismatch: {name}")
            source_sha[f"source_data/{name}"] = actual
        source_sha["source_data/source_bundle_manifest.json"] = sha(bundle_manifest_path)
        weight_path = ROOT / "weights/model.safetensors"
        if sha(weight_path) != WEIGHT_SHA:
            raise RuntimeError("official TimesFM-3 checkpoint hash mismatch")
        heads, training = load_pinned_heads()
        if tuple(training.SEEDS) != SEEDS or sha(ROOT / "stage5/g1/heads.py") != EXPECTED["stage5/g1/heads.py"]:
            raise RuntimeError("pinned adapter implementation or seed set mismatch")

        # Retain the exact source code, protocol, and environment for reproducibility.
        source_dir = out / "source"
        source_dir.mkdir()
        for rel in ("stage5/g1/heads.py", "stage5/g1/training.py",
                    "imports/base_effect_v1/BaseEffect_TimesFM3_MVP/beah.py",
                    "scripts/run_timesfm3_cpi_multivariate_baseline.py"):
            dest = source_dir / Path(rel).name
            dest.write_bytes((ROOT / rel).read_bytes())
        stat_ref = EXP / "statistical_reference_v1"
        for name in ("forecast_predictions.csv", "origin_fit_audit.csv", "manifest.json"):
            (source_dir / f"statistical_{name}").write_bytes((stat_ref / name).read_bytes())
        (source_dir / "run_unified_rolling.py").write_bytes(Path(__file__).read_bytes())
        (source_dir / "run_unified_statistics.py").write_bytes((EXP / "code/run_unified_statistics.py").read_bytes())
        (source_dir / "source_bundle_manifest.json").write_bytes(bundle_manifest_path.read_bytes())
        (source_dir / "execution_amendment_b3_replay_v1.md").write_bytes(
            (EXP / "execution_amendment_b3_replay_v1.md").read_bytes())
        (source_dir / "protocol_v1.md").write_bytes((EXP / "protocol_v1.md").read_bytes())
        (source_dir / "protocol_v1.json").write_bytes((EXP / "protocol_v1.json").read_bytes())
        env = {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA",
            "checkpoint_sha256": sha(weight_path), "official_source_commit": SOURCE_COMMIT,
            "statistics_forecasts": "per-origin expanding-window refit; statsmodels 0.14.4; separate target-free run",
        }
        try:
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                                    capture_output=True, text=True).stdout
            (out / "environment-lock.txt").write_text(freeze, encoding="utf-8")
        except Exception as exc:
            (out / "environment-lock.txt").write_text(f"pip freeze unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        (out / "environment.json").write_text(json.dumps(env, indent=2, default=str) + "\n", encoding="utf-8")

        # Read-only normalized metadata and source panels. The numeric 2026-09
        # evaluation target is absent; earlier first-release values are accessed
        # as inputs or training labels only after each origin's release gate.
        manifest = read_csv(DATA / "unified_origins.csv")
        origins = {}
        for r in manifest:
            if r["country"] != "CN":
                continue
            oid = r["origin_id"]
            release_time = stamp(r["origin_timestamp_utc"])
            current = dict(origin_id=oid, origin_month=r["origin_month"],
                           origin_release_timestamp=release_time.isoformat(),
                           origin_timestamp=(release_time + timedelta(minutes=1)).isoformat(),
                           origin_split=r["origin_split"])
            if oid in origins and origins[oid] != current:
                raise RuntimeError(f"origin manifest conflict: {oid}")
            origins[oid] = current

        g3_origin_manifest = json.loads((ROOT / "stage5/g3_cn_v1/origin_input_manifest.json").read_text(encoding="utf-8"))
        for meta in g3_origin_manifest:
            release = stamp(meta["origin_release_timestamp"])
            expected_cutoff = release + timedelta(minutes=1)
            recorded_cutoff = stamp(meta["origin_timestamp"])
            if expected_cutoff != recorded_cutoff:
                raise RuntimeError(f"G3 origin does not follow the frozen release-plus-one-minute clock: {meta['origin_id']}")
            if stamp(origins[meta["origin_id"]]["origin_timestamp"]) != recorded_cutoff:
                raise RuntimeError(f"unified origin clock differs from the G3 archived clock: {meta['origin_id']}")

        origin_audit = {r["origin_id"]: r for r in read_csv(DATA / "unified_origin_eligibility.csv")
                        if r.get("country") == "CN"}
        label_audit = {(r["origin_id"], int(r["horizon"])): r
                       for r in read_csv(DATA / "unified_label_eligibility.csv")
                       if r["country"] == "CN"}
        if set(origins) - set(origin_audit):
            raise RuntimeError("origin eligibility audit lacks registered origins")
        for oid in origins:
            if stamp(origin_audit[oid]["origin_timestamp"]) != datetime.fromisoformat(origins[oid]["origin_release_timestamp"]):
                raise RuntimeError(f"manifest and G2 release timestamp mismatch: {oid}")

        # Target-free panel of CPI inputs, used only through each origin clock.
        cpi_rows = read_csv(DATA / "unified_cpi_panel.csv")
        cpi = {}
        for r in cpi_rows:
            month = r["reference_month"][:7]
            av = stamp(r["available_at_conservative_local"], local=True)
            raw_value = r["first_release_yoy_pct"].strip()
            cpi[month] = {"raw_value": raw_value, "available_at": av,
                          "source": r.get("value_source", ""), "url": r.get("release_url", "")}
        if cpi.get("2026-09", {}).get("raw_value"):
            raise RuntimeError("numeric September 2026 scoring target is present in the forecast bundle")
        cpi_value_access_rows = []

        def cpi_value_asof(month: str, information_cutoff: datetime, purpose: str,
                           request_origin_id: str, training_origin_id: str = "") -> float:
            if month not in cpi:
                raise RuntimeError(f"CPI value source lacks reference month {month}")
            rec = cpi[month]
            if rec["available_at"] >= information_cutoff:
                raise RuntimeError(f"PIT value access rejected: {month} published at/after cutoff")
            if not rec["raw_value"]:
                raise RuntimeError(f"CPI numeric value is not materialized for {month}")
            value = float(rec["raw_value"])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite CPI first-release value for {month}")
            cpi_value_access_rows.append({"reference_month": month,
                                         "request_origin_id": request_origin_id,
                                         "training_origin_id": training_origin_id,
                                         "purpose": purpose,
                                         "simulated_information_cutoff_utc": information_cutoff.isoformat(),
                                         "official_first_release_available_at_utc": rec["available_at"].isoformat(),
                                         "strictly_available_before_cutoff": True,
                                         "value_materialized_after_gate": True,
                                         "actual_process_accessed_at_utc": now()})
            return value

        # Planned target-origin-horizon map. No target values are needed here.
        planned = []
        eval_origins = set()
        for target in pd.period_range("2018-01", "2026-09", freq="M"):
            for h in HORIZONS:
                om = str(target - h)
                oid = "CN_" + om
                if oid not in origins or str(target) not in cpi:
                    raise RuntimeError(f"missing planned origin/target metadata: {oid} {target}")
                ot = datetime.fromisoformat(origins[oid]["origin_timestamp"])
                if cpi[str(target)]["available_at"] <= ot:
                    raise RuntimeError(f"target release is not strictly after origin: {oid} h{h}")
                planned.append({"origin_id": oid, "origin_month": om, "origin_timestamp_utc": ot.isoformat(),
                                "origin_release_timestamp_utc": origins[oid]["origin_release_timestamp"],
                                "target_month": str(target), "horizon_months": h})
                eval_origins.add(oid)
        if len(planned) != 420 or len(eval_origins) != 116:
            raise RuntimeError(f"wrong planned cohort size: {len(planned)} rows/{len(eval_origins)} origins")
        planned_keys = {(r["origin_id"], r["horizon_months"], r["target_month"]) for r in planned}

        # Read audited origin-level model-input eligibility and covariate source
        # upper bounds. Unresolved origins are not imputed or sent to B3/heads.
        cov_audit = {}
        for r in read_csv(DATA / "unified_covariate_audit.csv"):
            if r["country"] != "CN":
                continue
            key = (r["origin_id"], r["context_reference_month"], r["feature_name"])
            cov_audit[key] = r
        ready = {oid: origin_audit[oid]["full_input_provenance_ready"].lower() == "true" for oid in origins}
        source_check_rows = []
        cpi_contexts = {}
        cpi_input_snapshot_rows = []
        for oid in sorted(origins, key=lambda x: (origins[x]["origin_month"], x)):
            om = origins[oid]["origin_month"]
            ot = datetime.fromisoformat(origins[oid]["origin_timestamp"])
            ctx_months = sorted(m for m in cpi if m <= om and cpi[m]["available_at"] < ot)
            if not ctx_months or ctx_months[-1] != om:
                raise RuntimeError(f"CPI context is not available through origin: {oid}")
            context = []
            for m in ctx_months:
                value = cpi_value_asof(m, ot, "historical_origin_input_context", oid)
                available_at = cpi[m]["available_at"].isoformat()
                context.append((m, value, available_at))
                cpi_input_snapshot_rows.append({"origin_id": oid, "origin_month": om,
                                                "origin_release_timestamp_utc": origins[oid]["origin_release_timestamp"],
                                                "origin_timestamp_utc": ot.isoformat(),
                                                "feature": "CPI_YOY_PCT", "observation_month": m,
                                                "value": value, "masked": False, "mask_reason": "",
                                                "source_id": "stats.gov.cn", "series_id": "CN_CPI_YOY_FIRST_RELEASE",
                                                "vintage_id": f"CN_CPI_YOY_FIRST_RELEASE_{m}",
                                                "available_at": available_at,
                                                "source_sha256": source_sha["source_data/unified_cpi_panel.csv"],
                                                "source_url": cpi[m]["url"]})
            cpi_contexts[oid] = context

        # CPI-only fixed candidate inputs are available at all evaluable CPI origins regardless
        # of unresolved macro covariate provenance.
        feature_columns = ["country", "origin_id", "anchor_yoy_pct", "anchor_yoy_released_at", "anchor_month"]
        feature_columns += [f"mom_{j:02d}_{suffix}" for j in range(12) for suffix in ("pct", "released_at")]
        panel_rows = pd.read_csv(DATA / "unified_bace_panel.csv",
                                 usecols=feature_columns, dtype=str, keep_default_na=False).to_dict("records")
        panel = {r["origin_id"]: r for r in panel_rows
                 if r["country"] == "CN" and r["origin_id"] in origins}
        if len(panel) != len(origins):
            raise RuntimeError(f"unified CPI-only panel coverage differs from origin set: {len(panel)} vs {len(origins)}")
        cpi_head_eligible = {}
        for oid, r in panel.items():
            ot = datetime.fromisoformat(origins[oid]["origin_timestamp"])
            checks = []
            for field in ["anchor_yoy_released_at"] + [f"mom_{j:02d}_released_at" for j in range(12)]:
                ts = stamp(r[field])
                checks.append(ts < ot)
            finite = all(math.isfinite(float(r[k])) for k in ["anchor_yoy_pct"] + [f"mom_{j:02d}_pct" for j in range(12)])
            cpi_head_eligible[oid] = all(checks) and finite
            if ready.get(oid, False) and not cpi_head_eligible[oid]:
                raise RuntimeError(f"audited full-input origin fails strict CPI feature gate: {oid}")
        for oid in sorted(eval_origins):
            if not cpi_head_eligible[oid]:
                raise RuntimeError(f"CPI-only candidate input fails strict PIT gate: {oid}")

        # Feature snapshots and point-in-time audit for the macro panel.
        pit_rows = read_csv(DATA / "unified_pit_covariates.csv")
        cov_by = {}
        for r in pit_rows:
            if r["country"] != "CN":
                continue
            oid = r["origin_id"]
            month = r["context_reference_month"]
            feat = r["feature_name"]
            cov_by.setdefault(oid, {}).setdefault(month, {})[feat] = r
        # Recompute the official multivariate baseline at every eligible origin,
        # including historical training origins used by later correction heads.
        b3_origins = [oid for oid in sorted(origins, key=lambda x: (origins[x]["origin_month"], x))
                      if ready.get(oid, False)]

        test_input_rows = []
        origin_input_hash = {}
        for oid in sorted(eval_origins):
            cpi_hash = canonical_hash(cpi_contexts[oid])
            anchor = panel[oid]
            candidate_payload = {"anchor": anchor["anchor_yoy_pct"],
                                 "anchor_release": anchor["anchor_yoy_released_at"],
                                 "anchor_month": anchor["anchor_month"],
                                 "monthly": [(anchor[f"mom_{j:02d}_pct"], anchor[f"mom_{j:02d}_released_at"]) for j in range(12)]}
            cand_hash = canonical_hash(candidate_payload)
            input_hash = {"TimesFM-3 target-only": cpi_hash, "Fixed candidate": cand_hash}
            test_input_rows.append({"origin_id": oid, "origin_month": origins[oid]["origin_month"],
                                    "origin_timestamp_utc": origins[oid]["origin_timestamp"],
                                    "origin_release_timestamp_utc": origins[oid]["origin_release_timestamp"],
                                    "cpi_context_length": len(cpi_contexts[oid]), "cpi_context_sha256": cpi_hash,
                                    "fixed_candidate_input_sha256": cand_hash,
                                    "cpi_only_inputs_eligible": cpi_head_eligible[oid],
                                    "macro_input_provenance_ready": ready[oid],
                                    "macro_unresolved_reasons": origin_audit[oid].get("unresolved_reasons", "")})
            origin_input_hash[oid] = input_hash
        write_csv(out / "evaluation_origin_input_index.csv", test_input_rows)
        write_csv(out / "planned_origin_horizon_keys.csv", planned)

        # Read the newly generated, origin-specific statistical refits. The
        # forecast file contains no evaluation errors or numeric target column.
        stat_ref_path = EXP / "statistical_reference_v1/forecast_predictions.csv"
        stat_manifest_path = EXP / "statistical_reference_v1/manifest.json"
        stat_manifest = json.loads(stat_manifest_path.read_text(encoding="utf-8"))
        if stat_manifest.get("status") != "STATISTICAL_FORECASTS_COMPLETE_NO_SCORING" or stat_manifest.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("unified statistical forecast manifest is not complete or protocol-aligned")
        if sha(stat_ref_path) != stat_manifest["outputs"]["forecast_predictions.csv"]["sha256"]:
            raise RuntimeError("statistical forecast file hash differs from its frozen manifest")
        stat_ref = pd.read_csv(stat_ref_path, dtype={"model": str, "origin_id": str, "origin_month": str,
                                                     "origin_timestamp_utc": str, "target_month": str,
                                                     "status": str})
        if len(stat_ref) != 1680 or set(stat_ref.model) != {"No-change", "Seasonal-naive", "AR(1)", "SARIMA"}:
            raise RuntimeError("new expanding-origin statistical forecast matrix has an unexpected shape")
        stat_rows = []
        stat_parameter_counts = {"No-change": 0, "Seasonal-naive": 0, "AR(1)": 2, "SARIMA": 4}
        for r in stat_ref.itertuples(index=False):
            oid = r.origin_id
            h = int(r.horizon_months)
            target = str(r.target_month)
            if (oid, h, target) not in planned_keys:
                raise RuntimeError(f"statistical forecast key outside frozen cohort: {oid}/h{h}/{target}")
            declared_time = stamp(r.origin_timestamp_utc)
            expected_time = datetime.fromisoformat(origins[oid]["origin_timestamp"])
            if declared_time != expected_time:
                raise RuntimeError(f"statistical origin clock mismatch: {oid}")
            history = cpi_contexts[oid]
            if r.model == "No-change":
                used_history = history[-1:]
            elif r.model == "Seasonal-naive":
                seasonal_month = add_month(target, -12)
                used_history = [x for x in history if x[0] == seasonal_month]
                if len(used_history) != 1:
                    raise RuntimeError(f"seasonal-naive input is absent at {oid}/h{h}")
            else:
                used_history = history
            stat_input_hash = canonical_hash({"model": r.model, "cpi_history": used_history,
                                              "cpi_source_sha256": source_sha["source_data/unified_cpi_panel.csv"]})
            value = float(r.prediction) if pd.notna(r.prediction) and str(r.prediction).strip() else ""
            if value != "" and not math.isfinite(value):
                raise RuntimeError(f"non-finite statistical prediction {oid}/h{h}/{r.model}")
            stat_rows.append({"model": r.model, "seed": "", "origin_id": oid,
                              "origin_month": str(r.origin_month), "origin_timestamp_utc": expected_time.isoformat(),
                              "target_month": target, "horizon_months": h, "prediction": value,
                              "status": r.status, "failure_reason": r.failure_reason or "",
                              "input_hash": stat_input_hash,
                              "training_packet_hash": "", "selected_epochs": "",
                              "parameter_count": stat_parameter_counts[r.model],
                              "training_start": r.training_start, "training_end": r.training_end,
                              "training_n_months": r.training_n_months, "adjacent_pairs": r.adjacent_pairs,
                              "sarima_optimizer_converged": r.sarima_optimizer_converged,
                              "sarima_iterations": r.sarima_iterations})
        if len(stat_rows) != 1680 or len({(r["model"], r["origin_id"], r["horizon_months"]) for r in stat_rows}) != 1680:
            raise RuntimeError("statistical reference keys are duplicated or missing")
        write_csv(out / "statistical_predictions.csv", stat_rows)
        shutil.copy2(EXP / "statistical_reference_v1/origin_fit_audit.csv", out / "statistical_fit_audit.csv")
        shutil.copy2(stat_manifest_path, out / "statistical_reference_manifest.json")
        print("Loaded 1,680 newly refitted statistical rows; evaluation target/error fields are absent.", flush=True)

        # Build target-only CPI contexts and B3 macro contexts. CPI contexts are
        # first-release observations strictly available at the archived origin.
        target_items = []
        for oid in sorted(eval_origins, key=lambda x: (origins[x]["origin_month"], x)):
            target_items.append((oid, cpi_contexts[oid]))
        b3_items = []
        input_snapshot_rows = []
        for oid in b3_origins:
            om = origins[oid]["origin_month"]
            ot = datetime.fromisoformat(origins[oid]["origin_timestamp"])
            month_list = [item[0] for item in cpi_contexts[oid]]
            target_vals = [item[1] for item in cpi_contexts[oid]]
            if not month_list or month_list[-1] != om:
                raise RuntimeError(f"B3 target context incomplete at {oid}")
            cv, cm, cv_payload = [], [], []
            for m in month_list:
                rows = cov_by.get(oid, {}).get(m, {})
                vals, masks = [], []
                for feat in FEATURES:
                    if feat not in rows:
                        raise RuntimeError(f"missing covariate cell {oid}/{m}/{feat}")
                    cell = rows[feat]
                    masked = cell["masked"].lower() == "true"
                    reason = cell.get("mask_reason", "")
                    audit = None
                    if masked:
                        if not reason or str(cell.get("feature_value", "")).strip():
                            raise RuntimeError(f"masked covariate has value/missing reason {oid}/{m}/{feat}")
                        value = 0.0
                    else:
                        value = float(cell["feature_value"])
                        if not math.isfinite(value):
                            raise RuntimeError(f"nonfinite covariate {oid}/{m}/{feat}")
                        audit = cov_audit.get((oid, m, feat))
                        audit_matches = bool(audit) and (
                            audit.get("value_reconstruction_matches", "").lower() == "true" or
                            audit.get("npz_cell_matches_availability_ledger", "").lower() == "true")
                        if not audit or audit.get("masked_in_legacy_input", "false").lower() != "false" or not audit_matches:
                            raise RuntimeError(f"covariate source audit missing/mismatched {oid}/{m}/{feat}")
                        upper = stamp(audit["source_availability_upper_bound"])
                        if upper >= ot:
                            raise RuntimeError(f"covariate source bound not strictly before origin {oid}/{m}/{feat}")
                        source_check_rows.append({"origin_id": oid, "origin_month": om, "origin_timestamp_utc": ot.isoformat(),
                                                 "context_reference_month": m, "feature_name": feat,
                                                 "masked": False, "available_upper_bound_utc": upper.isoformat(),
                                                 "strictly_before_origin": True, "value_reconstruction_matches": True})
                    vals.append(value); masks.append(masked)
                    cv_payload.append((m, feat, cell.get("feature_value", ""), masked, reason,
                                       cell.get("feature_source_reference_month", ""), cell.get("feature_age_months", "")))
                    source_id = ""
                    series_id = feat
                    vintage_id = ""
                    available_at = ""
                    row_source_hash = ""
                    source_url = ""
                    if not masked:
                        source_url = audit.get("source_url", "")
                        source_id = source_url.split("/")[2].lower() if "://" in source_url else "unknown"
                        vintage_stamp = (audit.get("vintage_date", "").strip() or
                                         audit.get("original_vintage_day_upper_bound", "").strip() or
                                         audit.get("source_availability_upper_bound", "").strip())
                        source_ref = cell.get("feature_source_reference_month", "").strip() or m
                        vintage_id = f"{source_ref}:{vintage_stamp}"
                        available_at = audit["source_availability_upper_bound"]
                        row_source_hash = canonical_hash({"feature_value": format(value, ".17g"),
                                                          "source_url": source_url, "vintage_id": vintage_id,
                                                          "available_at": available_at,
                                                          "pit_input_sha256": source_sha["source_data/unified_pit_covariates.csv"],
                                                          "source_audit_sha256": source_sha["source_data/unified_covariate_audit.csv"]})
                    input_snapshot_rows.append({"origin_id": oid, "origin_month": om, "origin_timestamp_utc": ot.isoformat(),
                                                "context_reference_month": m, "feature_name": feat,
                                                "value": "" if masked else value, "masked": masked, "mask_reason": reason,
                                                "source_reference_month": cell.get("feature_source_reference_month", ""),
                                                "source_age_months": cell.get("feature_age_months", ""),
                                                "available_upper_bound_utc": "" if masked else audit["source_availability_upper_bound"],
                                                "source_id": source_id, "series_id": series_id, "vintage_id": vintage_id,
                                                "available_at": available_at, "source_sha256": row_source_hash,
                                                "source_url": source_url,
                                                "source_file_sha256": source_sha["source_data/unified_pit_covariates.csv"]})
                cv.append(vals); cm.append(masks)
            raw_payload = {"cpi": cpi_contexts[oid], "covariates": cv_payload,
                           "covariate_file_sha256": source_sha["source_data/unified_pit_covariates.csv"]}
            input_hash = canonical_hash(raw_payload)
            origin_input_hash.setdefault(oid, {})["B3 multivariate"] = input_hash
            b3_items.append({"origin_id": oid, "origin_month": om, "origin_timestamp": ot,
                             "months": month_list, "target": np.asarray(target_vals, np.float32),
                             "cov": np.asarray(cv, np.float32).T, "covmask": np.asarray(cm, np.bool_).T,
                             "input_hash": input_hash})
        write_csv(out / "b3_input_snapshots.csv", input_snapshot_rows)
        write_csv(out / "cpi_input_snapshots.csv", cpi_input_snapshot_rows)
        write_csv(out / "b3_source_availability_checks.csv", source_check_rows)
        if not b3_items:
            raise RuntimeError("no macro-ready historical origins are available for B3")
        b3_reference_outputs, b3_cache_check = load_frozen_b3_cache(
            DATA / "unified_b3_reference.csv",
            source_sha["source_data/unified_b3_reference.csv"], b3_items)
        write_csv(out / "b3_reference_key_check.csv", b3_cache_check)

        if args.preflight_only:
            write_csv(out / "rolling_cpi_value_access_log.csv", cpi_value_access_rows)
            report = {"protocol_id": state["protocol_id"], "status": "PIT_PREFLIGHT_PASSED_NO_PREDICTION",
                      "created_at_utc": now(), "planned_origin_horizon_keys": len(planned),
                      "unique_evaluation_origins": len(eval_origins), "cpi_only_eligible_origins": sum(cpi_head_eligible[x] for x in eval_origins),
                      "macro_provenance_ready_evaluation_origins": sum(ready[x] for x in eval_origins),
                      "macro_unresolved_evaluation_origins": sorted(x for x in eval_origins if not ready[x]),
                      "macro_eligible_historical_b3_input_origins": len(b3_items),
                      "b3_reference_origin_horizon_keys": len(b3_cache_check),
                      "new_expanding_origin_statistical_rows": len(stat_rows),
                      "cpi_asof_value_accesses": len(cpi_value_access_rows),
                      "score_target_values_materialized": 0,
                      "timesfm_checkpoint_loaded": False, "adapter_fits_run": 0,
                      "P1_accessed": False, "prospective_queue_accessed": False,
                      "source_sha256": source_sha}
            (out / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            state.update(status="PIT_PREFLIGHT_PASSED_NO_PREDICTION", completed_at_utc=now(),
                         planned_pairs=420, unique_test_origins=116, macro_ready_origins=sum(ready[x] for x in eval_origins),
                         scoring_targets_loaded=False, score_target_values_materialized=0,
                         P1_accessed=False, prospective_queue_accessed=False,
                         preflight_report_sha256=sha(out / "preflight_report.json"))
            (out / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            return

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA A800-SXM4-80GB":
            raise RuntimeError("expected CUDA_VISIBLE_DEVICES=0 on NVIDIA A800-SXM4-80GB")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        # The pinned reference cumsum has no strict deterministic CUDA kernel
        # in this PyTorch build. Keep deterministic checks for supported ops,
        # but let PyTorch warn and execute this frozen operation.
        torch.use_deterministic_algorithms(True, warn_only=True)
        from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
        device = "cuda:0"
        t_load = time.perf_counter()
        forecaster = TimesFM3Forecaster.from_pretrained(str(ROOT / "weights"), device=device,
                                                        per_core_batch_size=4, local_files_only=True)
        model_load_s = time.perf_counter() - t_load
        if tuple(forecaster.config.quantiles) != (.1, .2, .3, .4, .5, .6, .7, .8, .9):
            raise RuntimeError("unexpected TimesFM-3 quantile specification")
        if forecaster.config.quantiles[forecaster.config.median_quantile_index] != .5:
            raise RuntimeError("configured median quantile is not q50")
        backbone_params = sum(p.numel() for p in forecaster.model.parameters())
        forecast_rows = []
        gpu_resource = {}

        # Target-only q50 on every planned evaluation origin, using the same official
        # direct decode API, left padding, global mask, context and q50 index.
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        target_outputs = {}
        for start in range(0, len(target_items), 4):
            batch = target_items[start:start + 4]
            L = max(len(ctx) for _, ctx in batch)
            target = np.zeros((len(batch), 1, L), np.float32)
            mask = np.ones((len(batch), L), np.bool_)
            for i, (_, ctx) in enumerate(batch):
                n = len(ctx); sl = slice(L - n, L)
                target[i, 0, sl] = [v for _, v, _ in ctx]
                mask[i, sl] = False
            with torch.inference_mode():
                raw = forecaster.model.decode(target=torch.from_numpy(target).to(device), horizon=12,
                                              mask=torch.from_numpy(mask).to(device))
            raw = raw[:, 0, :12, :].detach().float().cpu().numpy()
            if raw.shape != (len(batch), 12, 9) or not np.isfinite(raw).all():
                raise RuntimeError("bad target-only output shape or values")
            raw = np.sort(raw, axis=-1)
            for (oid, _), q in zip(batch, raw):
                target_outputs[oid] = q
        target_elapsed = time.perf_counter() - t0
        gpu_resource["TimesFM-3 target-only"] = {"inference_wall_seconds": target_elapsed,
                                                   "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                                                   "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                                                   "origin_count": len(target_outputs), "parameter_count": backbone_params}

        # Re-run B3 from the same official checkpoint and the unified origin-level
        # CPI/covariate tensors. Compare against the previously sealed outputs as
        # an equivalence check, then use only the fresh predictions downstream.
        torch.cuda.reset_peak_memory_stats()
        b3_t0 = time.perf_counter()
        b3_outputs = {}
        b3_equivalence_rows = []
        b3_quantile_equivalence_rows = []
        for start in range(0, len(b3_items), 4):
            batch = b3_items[start:start + 4]
            L = max(len(x["target"]) for x in batch)
            target = np.zeros((len(batch), 1, L), np.float32)
            past_cov = np.zeros((len(batch), len(FEATURES), L), np.float32)
            past_mask = np.ones((len(batch), len(FEATURES), L), np.bool_)
            global_mask = np.ones((len(batch), L), np.bool_)
            for i, x in enumerate(batch):
                n = len(x["target"]); sl = slice(L - n, L)
                target[i, 0, sl] = x["target"]
                past_cov[i, :, sl] = x["cov"]
                past_mask[i, :, sl] = x["covmask"]
                global_mask[i, sl] = False
            with torch.inference_mode():
                raw = forecaster.model.decode(
                    target=torch.from_numpy(target).to(device), horizon=12,
                    past_only_covariates=torch.from_numpy(past_cov).to(device),
                    past_only_mask=torch.from_numpy(past_mask).to(device),
                    mask=torch.from_numpy(global_mask).to(device))
            raw = raw[:, 0, :12, :].detach().float().cpu().numpy()
            if raw.shape != (len(batch), 12, 9) or not np.isfinite(raw).all():
                raise RuntimeError("bad fresh B3 output shape or values")
            raw = np.sort(raw, axis=-1)
            for x, q in zip(batch, raw):
                oid = x["origin_id"]
                b3_outputs[oid] = q
                ref = b3_reference_outputs[oid]
                diff = np.abs(q - ref)
                maxdiff = float(diff.max())
                for hh in range(12):
                    for qi, qn in enumerate((10,20,30,40,50,60,70,80,90)):
                        b3_quantile_equivalence_rows.append({"origin_id":oid,"horizon_months":hh+1,
                                                             "quantile":qn,"fresh_prediction":float(q[hh,qi]),
                                                             "reference_prediction":float(ref[hh,qi]),
                                                             "difference":float(q[hh,qi]-ref[hh,qi])})
                b3_equivalence_rows.append({"origin_id": oid, "max_abs_quantile_difference_pp": maxdiff,
                                            "q50_abs_difference_pp": float(abs(q[:, 4] - ref[:, 4]).max()),
                                            "all_quantiles_finite": True, "reference_origin_present": True,
                                            "within_tolerance_1e-3": maxdiff <= 1e-3})
        b3_elapsed = time.perf_counter() - b3_t0
        if set(b3_outputs) != {x["origin_id"] for x in b3_items}:
            raise RuntimeError("fresh B3 output origin set differs from the eligible input set")
        write_csv(out / "b3_reference_equivalence.csv", b3_equivalence_rows)
        write_csv(out / "b3_quantile_equivalence.csv", b3_quantile_equivalence_rows)
        (out / "b3_reference_equivalence_summary.json").write_text(json.dumps({
            "origin_count": len(b3_equivalence_rows),
            "max_abs_quantile_difference_pp": max(r["max_abs_quantile_difference_pp"] for r in b3_equivalence_rows),
            "max_q50_difference_pp": max(r["q50_abs_difference_pp"] for r in b3_equivalence_rows),
            "origins_within_1e-3_pp": sum(r["within_tolerance_1e-3"] for r in b3_equivalence_rows),
            "origins_over_1e-3": [r["origin_id"] for r in b3_equivalence_rows if not r["within_tolerance_1e-3"]],
            "reference_comparison_is_diagnostic_only": True,
            "fresh_unified_predictions_used_for_all_downstream_models": True,
            "largest_differences": sorted(b3_equivalence_rows, key=lambda r: r["max_abs_quantile_difference_pp"], reverse=True)[:20]
        }, indent=2) + "\n", encoding="utf-8")
        gpu_resource["B3 multivariate"] = {"status": "FRESH_INFERENCE_INPUTS_MATCH_REFERENCE_SMALL_NUMERICAL_DIFFERENCES_RETAINED",
                                            "inference_wall_seconds": b3_elapsed,
                                            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                                            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                                            "origin_count": len(b3_outputs), "parameter_count": backbone_params,
                                            "reference_file_sha256": source_sha["source_data/unified_b3_reference.csv"],
                                            "max_abs_quantile_difference_pp": max(r["max_abs_quantile_difference_pp"] for r in b3_equivalence_rows),
                                            "origins_within_1e-3_pp": sum(r["within_tolerance_1e-3"] for r in b3_equivalence_rows),
                                            "reference_comparison_role": "diagnostic_only; all fresh B3 outputs retained"}
        del forecaster
        torch.cuda.empty_cache()

        # Export B3 full horizon paths for every permitted historical origin; no labels.
        b3_internal_rows = []
        for x in b3_items:
            oid = x["origin_id"]; q = b3_outputs[oid]
            for h in ALL_H:
                b3_internal_rows.append({"model": "B3 multivariate", "origin_id": oid,
                                         "origin_month": x["origin_month"],
                                         "origin_timestamp_utc": x["origin_timestamp"].isoformat(),
                                         "horizon_months": h, "target_month": add_month(x["origin_month"], h),
                                         "prediction": float(q[h - 1, 4]), "q10": float(q[h - 1, 0]),
                                         "q90": float(q[h - 1, 8]), "input_hash": x["input_hash"],
                                         "status": "FORECAST_COMPLETE"})
        write_csv(out / "b3_internal_horizon_forecasts.csv", b3_internal_rows)

        # Evaluation B3 and target-only rows, plus CPU fixed-candidate predictions.
        key_for_eval = {(r["origin_id"], r["horizon_months"]): r for r in planned}
        for oid, q in target_outputs.items():
            for h in HORIZONS:
                key = key_for_eval.get((oid, h))
                if not key:
                    continue
                forecast_rows.append({"model": "TimesFM-3 target-only", "seed": "", **key,
                                      "prediction": float(q[h - 1, 4]), "status": "FORECAST_COMPLETE",
                                      "input_hash": origin_input_hash[oid]["TimesFM-3 target-only"],
                                      "training_packet_hash": "", "selected_epochs": "", "parameter_count": backbone_params})
        for oid, q in b3_outputs.items():
            for h in HORIZONS:
                key = key_for_eval.get((oid, h))
                if not key:
                    continue
                forecast_rows.append({"model": "B3 multivariate", "seed": "", **key,
                                      "prediction": float(q[h - 1, 4]), "status": "FORECAST_COMPLETE",
                                      "input_hash": origin_input_hash[oid]["B3 multivariate"],
                                      "training_packet_hash": "", "selected_epochs": "", "parameter_count": backbone_params})
        stat_test = [r for r in stat_rows if (r["origin_id"], int(r["horizon_months"]), r["target_month"]) in planned_keys]
        forecast_rows.extend(stat_test)

        # Fixed candidate path on every CPI-input-eligible evaluation origin.
        cand_t0 = time.perf_counter()
        cand_rows = []
        for oid in sorted(eval_origins):
            r = panel[oid]
            anchor = torch.tensor([float(r["anchor_yoy_pct"])], dtype=torch.float32)
            monthly = torch.tensor([[float(r[f"mom_{j:02d}_pct"]) for j in range(12)]], dtype=torch.float32)
            with torch.inference_mode():
                pred = heads.fixed_candidate(anchor, monthly).cpu().numpy()[0]
            if not np.isfinite(pred).all():
                raise RuntimeError(f"fixed candidate is nonfinite at {oid}")
            for j, h in enumerate(HORIZONS):
                key = key_for_eval.get((oid, h))
                if key:
                    cand_rows.append({"model": "Fixed candidate", "seed": "", **key,
                                      "prediction": float(pred[j]), "status": "FORECAST_COMPLETE",
                                      "input_hash": origin_input_hash[oid]["Fixed candidate"],
                                      "training_packet_hash": "", "selected_epochs": "", "parameter_count": 0})
        candidate_elapsed = time.perf_counter() - cand_t0
        forecast_rows.extend(cand_rows)

        # Build rolling supervised packet candidates from label metadata. Numeric
        # labels are converted only for source-verified cells released < outer.
        ready_panel = {oid: {**panel[oid], "origin_timestamp": origins[oid]["origin_timestamp"]}
                       for oid in panel if ready.get(oid, False)}
        label_meta = label_audit
        train_data_by_outer = {}
        fit_rows, vector_rows, packet_hash_rows, validation_trace_rows = [], [], [], []
        packet_zip_path = out / "rolling_training_packets.zip"
        adapter_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for outer_id in sorted(eval_origins, key=lambda x: (origins[x]["origin_month"], x)):
            if not ready.get(outer_id, False):
                continue
            outer_time = datetime.fromisoformat(origins[outer_id]["origin_timestamp"])
            eligible = {}
            eligible_origin_ids = set()
            for (train_id, h), meta in label_meta.items():
                if train_id not in ready_panel or train_id not in origins or not cpi_head_eligible.get(train_id, False):
                    continue
                train_time = datetime.fromisoformat(origins[train_id]["origin_timestamp"])
                rel = stamp(meta["label_release_timestamp"]) if meta.get("label_release_timestamp", "") else None
                if train_time >= outer_time or rel is None or rel >= outer_time:
                    continue
                if meta.get("label_finite", "").lower() != "true" or meta.get("source_value_matches", "").lower() != "true" or meta.get("source_release_metadata_matches", "").lower() != "true":
                    continue
                target_month = add_month(origins[train_id]["origin_month"], h)
                if target_month != meta["target_reference_month"]:
                    raise RuntimeError(f"training target mapping mismatch: {train_id}/h{h}")
                # Only after the outer-origin release gate and source checks do
                # we materialize a historical first-release label from the CPI
                # source panel. No BACE target/label columns are read by runner.
                cpi_release = cpi[target_month]["available_at"]
                if stamp(meta["label_release_timestamp"]) != cpi_release:
                    raise RuntimeError(f"training label release differs from CPI source: {train_id}/h{h}")
                value = cpi_value_asof(target_month, outer_time, "rolling_training_label",
                                       outer_id, training_origin_id=train_id)
                eligible[(train_id, h)] = value
                eligible_origin_ids.add(train_id)
            eligible_rows = [ready_panel[oid] for oid in eligible_origin_ids]
            h3_ids = sorted({oid for oid, h in eligible if h == 3},
                            key=lambda oid: (datetime.fromisoformat(origins[oid]["origin_timestamp"]), oid))
            val_ids = h3_ids[-12:]
            if len(val_ids) != 12:
                raise RuntimeError(f"outer origin lacks 12 h3 validation rows: {outer_id}")
            train_ids, inner_train_eligible, validation_eligible, val_start, inner_split_audit = \
                make_clean_inner_split(eligible, label_meta, origins, val_ids, outer_time)
            final_ids = sorted(eligible_origin_ids,
                               key=lambda oid: (datetime.fromisoformat(origins[oid]["origin_timestamp"]), oid))
            train_rows = [ready_panel[x] for x in train_ids]
            val_rows = [ready_panel[x] for x in val_ids]
            final_rows = [ready_panel[x] for x in final_ids]
            tpack = make_packet(train_rows, inner_train_eligible, { (x, h): float(b3_outputs[x][h-1,4]) for x in train_ids for h in HORIZONS }, device)
            vpack = make_packet(val_rows, validation_eligible, { (x, h): float(b3_outputs[x][h-1,4]) for x in val_ids for h in HORIZONS }, device)
            fpack = make_packet(final_rows, eligible, { (x, h): float(b3_outputs[x][h-1,4]) for x in final_ids for h in HORIZONS }, device)
            # Ineligible cells are zero/masked. The numeric training packet hashes
            # include values only after the release gate has passed.
            pack_hash = canonical_hash({
                "outer_origin_id": outer_id, "outer_origin_timestamp_utc": outer_time.isoformat(),
                "train_ids": train_ids, "validation_ids": val_ids, "final_ids": final_ids,
                "nested_validation_policy": "all inner-training labels released strictly before validation start; train and h3 validation target-reference-month sets disjoint",
                "inner_split_audit": inner_split_audit,
                "train": [x.tolist() for x in tpack[4]], "validation": [x.tolist() for x in vpack[4]],
                "final": [x.tolist() for x in fpack[4]],
            })
            train_data_by_outer[outer_id] = (tpack, vpack, fpack, pack_hash, train_ids, val_ids, final_ids)
            packet_hash_rows.append({"outer_origin_id": outer_id, "training_packet_hash": pack_hash,
                                     "eligible_train_origins": len(train_ids), "inner_validation_h3_origins": len(val_ids),
                                     "final_fit_origins": len(final_ids), "eligible_label_cells": int(fpack[2].sum().item()),
                                     **{k: v for k, v in inner_split_audit.items() if k != "validation_target_months"},
                                     "validation_target_months": ";".join(inner_split_audit["validation_target_months"])})
            # Store the exact per-origin numerical packets, with invalid targets
            # physically zero and masked. They contain only labels released before
            # this outer origin, never its evaluation target or later labels.
            b = io.BytesIO()
            np.savez_compressed(b,
                train_anchor=tpack[4][0], train_monthly=tpack[4][1], train_b3=tpack[4][2], train_month=tpack[4][3], train_target=tpack[4][4], train_valid=tpack[4][5],
                validation_anchor=vpack[4][0], validation_monthly=vpack[4][1], validation_b3=vpack[4][2], validation_month=vpack[4][3], validation_target=vpack[4][4], validation_valid=vpack[4][5],
                final_anchor=fpack[4][0], final_monthly=fpack[4][1], final_b3=fpack[4][2], final_month=fpack[4][3], final_target=fpack[4][4], final_valid=fpack[4][5],
                train_origin_ids=np.asarray(train_ids, dtype="U32"), validation_origin_ids=np.asarray(val_ids, dtype="U32"), final_origin_ids=np.asarray(final_ids, dtype="U32"))
            if not packet_zip_path.exists():
                zf = zipfile.ZipFile(packet_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6)
            zf.writestr(f"packets/{outer_id}.npz", b.getvalue())

            adapter_rows = []
            for kind in ("BEAH", "Direct", "GateOnly"):
                for seed in SEEDS:
                    torch.cuda.reset_peak_memory_stats()
                    select_model = model_create(heads, kind, seed).to(device)
                    initial_sha = parameter_fingerprint(select_model)
                    tsel = time.perf_counter()
                    selection = training.select_epochs(select_model, (tpack[0], tpack[1], tpack[2]),
                                                       (vpack[0], vpack[1], vpack[2]))
                    selection_s = time.perf_counter() - tsel
                    selected_epochs = int(selection["selected_epoch"])
                    for point in selection["trace"]:
                        validation_trace_rows.append({"outer_origin_id": outer_id, "model": kind, "seed": seed,
                                                      "epoch": int(point["epoch"]), "validation_h3_mae_pp": float(point["h3_mae"])})
                    final_model = model_create(heads, kind, seed).to(device)
                    if parameter_fingerprint(final_model) != initial_sha:
                        raise RuntimeError(f"seed initialization mismatch {outer_id}/{kind}/{seed}")
                    opt = training.optimizer_for(final_model)
                    tfit = time.perf_counter()
                    for _ in range(selected_epochs):
                        training.step(final_model, opt, fpack[0], fpack[1], fpack[2])
                    fit_s = time.perf_counter() - tfit
                    final_model.eval()
                    fit_sha = parameter_fingerprint(final_model)
                    vec = parameter_vector(final_model)
                    expected_parameters = 4 if kind == "GateOnly" else 688
                    if vec.size != expected_parameters:
                        raise RuntimeError(f"parameter count mismatch for {kind}: {vec.size}")
                    vector_rows.append({"outer_origin_id": outer_id, "model": kind, "seed": seed,
                                        "parameters": vec, "parameter_count": int(vec.size),
                                        "parameter_sha256": fit_sha, "selected_epochs": selected_epochs})
                    # Inference only for the same planned test target-origin keys.
                    evalrow = panel[outer_id]
                    anchor = torch.tensor([float(evalrow["anchor_yoy_pct"])], device=device, dtype=torch.float32)
                    monthly = torch.tensor([[float(evalrow[f"mom_{j:02d}_pct"]) for j in range(12)]], device=device, dtype=torch.float32)
                    base = torch.tensor([[float(b3_outputs[outer_id][h-1,4]) for h in HORIZONS]], device=device, dtype=torch.float32)
                    month_num = torch.tensor([int(evalrow["anchor_month"])], device=device, dtype=torch.int64)
                    eval_hash = canonical_hash({"candidate_input_hash": origin_input_hash[outer_id]["Fixed candidate"],
                                                "b3_input_hash": origin_input_hash[outer_id]["B3 multivariate"],
                                                "training_packet_hash": pack_hash})
                    tpred = time.perf_counter()
                    with torch.inference_mode():
                        outputs = final_model(anchor, monthly, base, month_num)
                    pred_s = time.perf_counter() - tpred
                    if parameter_fingerprint(final_model) != fit_sha:
                        raise RuntimeError("final adapter parameters changed during inference")
                    ypred = outputs["yoy_pct"].detach().cpu().numpy()[0]
                    gates = outputs["gates"].detach().cpu().numpy()
                    if not np.isfinite(ypred).all():
                        raise RuntimeError(f"nonfinite rolling adapter forecast {outer_id}/{kind}/{seed}")
                    for j, h in enumerate(HORIZONS):
                        key = key_for_eval.get((outer_id, h))
                        if key:
                            forecast_rows.append({"model": {"BEAH": "BEAH rolling", "Direct": "Direct rolling",
                                                             "GateOnly": "Fixed candidate + gate"}[kind],
                                                  "seed": seed, **key, "prediction": float(ypred[j]),
                                                  "status": "FORECAST_COMPLETE", "input_hash": eval_hash,
                                                  "training_packet_hash": pack_hash, "selected_epochs": selected_epochs,
                                                  "parameter_count": int(vec.size), "parameter_sha256": fit_sha,
                                                  "gate_value": float(gates[j] if np.ndim(gates) else gates)})
                    adapter_rows.append({"outer_origin_id": outer_id, "outer_origin_timestamp_utc": outer_time.isoformat(),
                                         "model": kind, "seed": seed, "status": "FIT_AND_FORECAST_COMPLETE",
                                         "selection_train_origins": len(train_ids), "selection_train_label_cells": int(tpack[2].sum().item()),
                                         "inner_validation_h3_origins": len(val_ids), "final_fit_origins": len(final_ids),
                                         "final_fit_label_cells": int(fpack[2].sum().item()),
                                         "eligible_label_cells_h1": int(fpack[2][:,0].sum().item()),
                                         "eligible_label_cells_h3": int(fpack[2][:,1].sum().item()),
                                         "eligible_label_cells_h6": int(fpack[2][:,2].sum().item()),
                                         "eligible_label_cells_h12": int(fpack[2][:,3].sum().item()),
                                         "selected_epochs": selected_epochs, "selection_wall_seconds": selection_s,
                                         "selection_optimizer_steps": int(selection["optimizer_steps"]),
                                         "best_validation_h3_mae_pp": float(selection["best_h3_mae"]),
                                         "final_fit_wall_seconds": fit_s, "inference_wall_seconds": pred_s,
                                         "training_packet_hash": pack_hash, "input_hash": eval_hash,
                                         "initial_parameters_sha256": initial_sha, "final_parameters_sha256": fit_sha,
                                         "parameter_count": int(vec.size), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                                         "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                                         "status_detail": "fresh same-seed final fit; labels released strictly before outer origin"})
                    del select_model, final_model, opt
                    torch.cuda.empty_cache()
            fit_rows.extend(adapter_rows)
            if len(fit_rows) % 75 == 0:
                print(f"rolling heads complete: {len(fit_rows)} origin-model-seed fits; current={outer_id}", flush=True)
        if packet_zip_path.exists():
            zf.close()
        adapter_elapsed = time.perf_counter() - adapter_started
        if len(train_data_by_outer) != sum(1 for x in eval_origins if ready.get(x, False)):
            raise RuntimeError("rolling training packet coverage mismatch")
        if len(fit_rows) != len(train_data_by_outer) * 3 * len(SEEDS):
            raise RuntimeError(f"adapter fit count mismatch {len(fit_rows)}")
        write_csv(out / "adapter_fit_manifest.csv", fit_rows)
        write_csv(out / "adapter_validation_traces.csv", validation_trace_rows)
        write_csv(out / "rolling_training_packet_index.csv", packet_hash_rows)

        # Parameter vectors: exact final trainable values for each origin/model/seed.
        maxp = max(len(x["parameters"]) for x in vector_rows)
        matrix = np.zeros((len(vector_rows), maxp), dtype=np.float32)
        vector_index = []
        for i, row in enumerate(vector_rows):
            matrix[i, :row["parameter_count"]] = row["parameters"]
            vector_index.append({k: row[k] for k in ("outer_origin_id", "model", "seed", "parameter_count", "parameter_sha256", "selected_epochs")})
        np.savez_compressed(out / "rolling_adapter_parameter_vectors.npz", parameter_vectors=matrix,
                            vector_index_json=np.asarray(json.dumps(vector_index, separators=(",", ":")), dtype="U"))

        # Add statistics forecasts now that all prediction families are complete.
        write_csv(out / "rolling_cpi_value_access_log.csv", cpi_value_access_rows)
        write_csv(out / "model_forecasts_unscored.csv", forecast_rows)
        coverage_rows = []
        forecast_df = pd.DataFrame(forecast_rows)
        for model in sorted(set(forecast_df.model)):
            for h in HORIZONS:
                g = forecast_df[(forecast_df.model == model) & (forecast_df.horizon_months.astype(int) == h)]
                valid_origin_keys = set(g.origin_id)
                success = g[g.status == "FORECAST_COMPLETE"] if "status" in g else g
            coverage_rows.append({"model": model, "horizon_months": h, "planned_origins": 105,
                                      "prediction_rows": int(len(g)), "native_origin_count": len(valid_origin_keys),
                                      "successful_origin_count": int(success.origin_id.nunique()),
                                      "successful_origin_seed_pairs": int(success[["origin_id", "seed"]].drop_duplicates().shape[0]) if "seed" in success else int(success.origin_id.nunique()),
                                      "failed_or_missing_planned_origins": 105 - int(success.origin_id.nunique()),
                                      "seed_count": len(set(g.seed)) if "seed" in g else 0,
                                      "coverage_note": "seed rows are not independent origins"})
        write_csv(out / "pre_score_model_coverage.csv", coverage_rows)

        # Retain complete source inventory and actual-stage resource measurements.
        resource_metrics = {
            "determinism": {"deterministic_algorithms": True,
                            "warn_only": True,
                            "bitwise_cuda_determinism_guaranteed": False,
                            "known_exception": "frozen reference CUDA cumsum; warning retained in run log"},
            "cpu_statistics_methods": {"status": "PER_ORIGIN_EXPANDING_REFIT_MEASURED",
                                       "wall_seconds": stat_manifest.get("total_wall_seconds"),
                                       "sarima_fit_wall_seconds": stat_manifest.get("sarima_origin_fit_wall_seconds"),
                                       "process_peak_rss_kib": stat_manifest.get("process_peak_rss_kib"),
                                       "source_run_created_at_utc": stat_manifest.get("created_at_utc"),
                                       "source_runtime_metadata_contains_measurements": True,
                                       "statsmodels_version": stat_manifest.get("statsmodels_version")},
            "timesfm3_checkpoint_load_wall_seconds": model_load_s,
            "gpu_forecast_stages": gpu_resource,
            "fixed_candidate": {
                "prediction_wall_seconds": candidate_elapsed,
                "origin_count": len({str(r["origin_id"]) for r in cand_rows}),
                "forecast_record_count": len(cand_rows),
                "forecast_records_by_horizon": {
                    str(h): sum(int(r["horizon_months"]) == h for r in cand_rows) for h in HORIZONS
                },
                "parameter_count": 0, "device": "CPU torch float32",
            },
            "rolling_adapters": {"fit_and_prediction_wall_seconds": adapter_elapsed,
                                 "fit_invocations": len(fit_rows), "seeds": list(SEEDS),
                                 "parameter_count_by_model": {"BEAH": 688, "Direct": 688, "GateOnly": 4},
                                 "max_per_fit_peak_allocated_bytes": max(int(r["gpu_peak_allocated_bytes"]) for r in fit_rows),
                                 "max_per_fit_peak_reserved_bytes": max(int(r["gpu_peak_reserved_bytes"]) for r in fit_rows),
                                 "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
            "hardware": {"device": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "cpu": platform.processor(), "cpu_threads": torch.get_num_threads()},
            "measurement_note": "Times are actual stage wall-clock durations for TimesFM-3, candidate prediction, and rolling heads. The frozen statistics run did not retain runtime or memory; no value is inferred.",
        }
        (out / "resource_metrics.json").write_text(json.dumps(resource_metrics, indent=2, default=str) + "\n", encoding="utf-8")
        (out / "source_hashes.json").write_text(json.dumps({**source_sha,
            "weights/model.safetensors": sha(weight_path),
            "protocol_v1.md": sha(EXP / "protocol_v1.md"), "protocol_v1.json": sha(EXP / "protocol_v1.json"),
            "source_bundle_manifest_sha256": sha(DATA / "source_bundle_manifest.json"),
            "execution_amendment_b3_replay_v1_sha256": sha(EXP / "execution_amendment_b3_replay_v1.md"),
            "runner_sha256": sha(Path(__file__)),
            "statistics_reference_manifest_sha256": sha(EXP / "statistical_reference_v1/manifest.json")},
            indent=2) + "\n", encoding="utf-8")
        completed = now()
        state.update(status="FORECASTS_SEALED_NO_TARGET_SCORING", completed_at_utc=completed,
                     elapsed_seconds=time.perf_counter() - t_run, planned_pairs=420,
                     unique_test_origins=116, planned_targets_per_horizon=105,
                     macro_ready_origins=sum(ready[x] for x in eval_origins),
                     all_macro_eligible_training_origins=len(b3_outputs),
                     rolling_adapter_outer_origins=len(train_data_by_outer), adapter_fit_invocations=len(fit_rows),
                     unscored_forecast_rows=len(forecast_rows), scoring_targets_loaded=False,
                     score_target_values_materialized=0,
                     asof_cpi_value_accesses=len(cpi_value_access_rows),
                     rolling_training_label_accesses=sum(r["purpose"] == "rolling_training_label" for r in cpi_value_access_rows),
                     historical_training_labels_used=True,
                     B3_predictions_recomputed_under_unified_protocol=True,
                     P1_accessed=False, prospective_queue_accessed=False,
                     target_scoring_authorized=False, source_sha256=source_sha,
                     official_checkpoint_sha256=sha(weight_path), official_source_commit=SOURCE_COMMIT,
                     scoring_status="NOT_RUN_IN_THIS_PROCESS")
        (out / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "sealed_manifest.json"}
        (out / "sealed_manifest.json").write_text(json.dumps({"created_at_utc": now(), "protocol_id": state["protocol_id"],
                                                              "target_scoring_authorized": False, "files": files}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": state["status"], "run_dir": str(out), "elapsed_seconds": state["elapsed_seconds"],
                          "unscored_forecast_rows": state["unscored_forecast_rows"],
                          "macro_ready_origins": state["macro_ready_origins"],
                          "adapter_fit_invocations": state["adapter_fit_invocations"],
                          "scoring_targets_loaded": False}, indent=2), flush=True)
    except BaseException as exc:
        partial_access_log = locals().get("cpi_value_access_rows", [])
        if partial_access_log:
            write_csv(out / "rolling_cpi_value_access_log_partial.csv", partial_access_log)
        state.update(status="FAILED_NO_TARGET_SCORING", failed_at_utc=now(),
                     error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                     elapsed_seconds=time.perf_counter() - t_run, scoring_targets_loaded=False,
                     score_target_values_materialized=0,
                     asof_cpi_value_accesses=len(partial_access_log),
                     rolling_training_label_accesses=sum(r["purpose"] == "rolling_training_label" for r in partial_access_log),
                     historical_training_labels_used=True,
                     P1_accessed=False, prospective_queue_accessed=False)
        (out / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
