#!/usr/bin/env python3
"""Create a target-free 10-method scoring matrix with only three heads replaced."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
EXP = PROJECT / "reproducibility/matched_origin_2018_2026_unified_v1"
DEFAULT_OLD_RUN = EXP / "results/complete_run_v1/runs/run_20261030_unified_full_v2"
DEFAULT_CLEAN_RUN = EXP / "results/nested_validation_clean_v1/run_20261030T190812Z"
REPLACED = {"BEAH rolling", "Direct rolling", "Fixed candidate + gate"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError("empty combined matrix")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def verify_run(run_dir: Path):
    manifest_path = run_dir / "sealed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = []
    for rel, expected in manifest.get("files", {}).items():
        p = run_dir / rel
        if not p.is_file() or sha(p) != expected:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"input run seal mismatch: {bad[:5]}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-run", type=Path, default=DEFAULT_OLD_RUN)
    ap.add_argument("--clean-run", type=Path, default=DEFAULT_CLEAN_RUN)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    old = args.old_run.resolve()
    clean = args.clean_run.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite scoring input directory: {out}")
    old_manifest = verify_run(old)
    clean_manifest = verify_run(clean)
    old_rows = read_csv(old / "model_forecasts_unscored.csv")
    new_rows = read_csv(clean / "corrected_head_forecasts_unscored.csv")
    planned_rows = read_csv(old / "planned_origin_horizon_keys.csv")
    coverage_rows = read_csv(old / "pre_score_model_coverage.csv")
    planned = {(r["origin_id"], int(r["horizon_months"])): r for r in planned_rows}
    if len(planned) != 420:
        raise RuntimeError("planned forecast key count is not 420")

    old_head = [r for r in old_rows if r["model"] in REPLACED]
    if len(old_head) != len(new_rows):
        raise RuntimeError(f"old/new head forecast row count differs: {len(old_head)} vs {len(new_rows)}")
    old_keys = {(r["model"], r["seed"], r["origin_id"], int(r["horizon_months"])) for r in old_head}
    new_keys = {(r["model"], r["seed"], r["origin_id"], int(r["horizon_months"])) for r in new_rows}
    if old_keys != new_keys:
        raise RuntimeError(f"old and corrected head prediction keys differ: old={len(old_keys)} new={len(new_keys)}")

    untouched_old = [r for r in old_rows if r["model"] not in REPLACED]
    untouched_new = [r for r in old_rows if r["model"] not in REPLACED]
    # This equality is intentionally checked before serialization: every
    # statistical, fixed-candidate, target-only and B3 record is retained as-is.
    if untouched_old != untouched_new:
        raise RuntimeError("a non-head forecast row changed during assembly")

    seen = set()
    for r in new_rows:
        key = (r["model"], r["seed"], r["origin_id"], int(r["horizon_months"]))
        if key in seen:
            raise RuntimeError(f"duplicate corrected head forecast key {key}")
        seen.add(key)
        if key[0] not in REPLACED or key[2:4] not in planned:
            raise RuntimeError(f"corrected forecast outside planned cohort {key}")
        if not math.isfinite(float(r["prediction"])):
            raise RuntimeError(f"non-finite corrected forecast {key}")
        base = planned[key[2:4]]
        for col in ("origin_month", "origin_timestamp_utc", "target_month"):
            if r[col] != base[col]:
                raise RuntimeError(f"corrected forecast origin mapping mismatch {key}/{col}")

    combined = untouched_old + new_rows
    combined_keys = [(r["model"], r["seed"], r["origin_id"], int(r["horizon_months"])) for r in combined]
    if len(combined_keys) != len(set(combined_keys)):
        raise RuntimeError("combined method/origin/horizon/seed keys are not unique")
    methods = sorted({r["model"] for r in combined})
    if len(methods) != 10:
        raise RuntimeError(f"expected 10 methods after assembly, found {methods}")
    # The archived ledger's representative coverage is defined at h=12 and is
    # checked by the reference scorer against all four horizons.
    coverage = {r["model"]: int(r["native_origin_count"]) for r in coverage_rows
                if int(r["horizon_months"]) == 12}
    for model in methods:
        for h in (1, 3, 6, 12):
            origins = {r["origin_id"] for r in combined if r["model"] == model and int(r["horizon_months"]) == h}
            expected = coverage.get(model)
            if expected is None or len(origins) != expected:
                raise RuntimeError(f"native coverage mismatch for {model}/h{h}: {len(origins)} vs {expected}")
    if len(combined) != 8856:
        raise RuntimeError(f"combined full matrix row count changed: {len(combined)}")

    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "model_forecasts_unscored.csv", combined)
    for name in ("planned_origin_horizon_keys.csv", "pre_score_model_coverage.csv"):
        (out / name).write_bytes((old / name).read_bytes())
    input_info = {
        "status": "CLEAN_NESTED_VALIDATION_MATRIX_SEALED_NO_TARGETS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_run_id": old.name,
        "corrected_head_run_id": clean.name,
        "original_run_manifest_sha256": sha(old / "sealed_manifest.json"),
        "corrected_head_manifest_sha256": sha(clean / "sealed_manifest.json"),
        "corrected_head_forecast_sha256": sha(clean / "corrected_head_forecasts_unscored.csv"),
        "replaced_methods": sorted(REPLACED),
        "unchanged_methods": sorted({r["model"] for r in untouched_old}),
        "forecast_rows": len(combined),
        "planned_keys": len(planned),
        "all_method_counts": {m: sum(r["model"] == m for r in combined) for m in methods},
        "target_values_loaded": False,
        "P1_accessed": False,
    }
    (out / "assembly_record.json").write_text(json.dumps(input_info, indent=2) + "\n", encoding="utf-8")
    file_hashes = {name: sha(out / name) for name in
                   ("model_forecasts_unscored.csv", "planned_origin_horizon_keys.csv",
                    "pre_score_model_coverage.csv", "assembly_record.json")}
    sealed = {"protocol_id": "CN_UNIFIED_EXPANDING_WINDOW_2018_2026_V1",
              "status": input_info["status"], "files": file_hashes,
              "scoring_targets_loaded": False, "P1_accessed": False}
    (out / "sealed_manifest.json").write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": input_info["status"], "out_dir": str(out),
                      "forecast_rows": len(combined), "method_count": len(methods),
                      "unchanged_methods_row_count": len(untouched_old),
                      "corrected_head_rows": len(new_rows),
                      "P1_accessed": False, "target_values_loaded": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
