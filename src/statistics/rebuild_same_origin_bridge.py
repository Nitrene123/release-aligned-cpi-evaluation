#!/usr/bin/env python3
"""Rebuild the target-free 2018--2024 month-key bridge from two frozen ZIP inputs.

This script performs a derived-loss join only. It does not read CPI target values,
forecast values, model weights, or P1 materials, and it does not fit or infer a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import sys
import zipfile
from calendar import monthrange
from pathlib import Path

ROLLING_ZIP_SHA256 = "c81b6e6e89bddaa8ee922a712dc58a176abfbdbc7e55a357aa21c45a8598fd54"
ARCHIVE_ZIP_SHA256 = "88c33bf4e87ab8c82e02ea118030819ab3bd9b49937b1dd376f972b4824cbd21"
ROLLING_MEMBER = "manuscript/ars_full_v2_4/reproducibility/rolling_origin_cpi_only_2018_2024_v1/code/runs/run_20261029T164057Z/forecast_losses.csv"
ARCHIVE_MEMBER = "run_20261029T121606Z/origin_level_paired_losses_vs_B3.csv"
OUTPUT_FIELDS = ["family", "method_version", "target_month", "origin_month", "horizon_months", "absolute_error_pp"]
HORIZONS = (1, 3, 6, 12)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_month(month: str, horizon: int) -> str:
    year, mon = map(int, month.split("-"))
    index = year * 12 + (mon - 1) + horizon
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def read_member(path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        try:
            raw = zf.read(member)
        except KeyError as exc:
            raise ValueError(f"required member missing from {path.name}: {member}") from exc
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="strict"))))


def expected_keys(horizon: int) -> set[tuple[str, str, int]]:
    keys = set()
    for year in (2023, 2024):
        for month in range(1, 13):
            target = f"{year:04d}-{month:02d}"
            # target - horizon, calculated without external date libraries
            index = year * 12 + month - 1 - horizon
            origin = f"{index // 12:04d}-{index % 12 + 1:02d}"
            keys.add((target, origin, horizon))
    return keys


def finite_float(value: str, label: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric {label}: {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"non-finite {label}: {value!r}")
    return x


def main() -> int:
    here = Path(__file__).resolve()
    content_root = here.parents[2]
    review_dir = content_root / "review_materials"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-zip", type=Path, default=review_dir / "Rolling_Origin_CPI_Only_2018_2024_Reproduction_v1.1.zip")
    parser.add_argument("--archived-scores-zip", type=Path, default=review_dir / "CN_2022_2024_archived_development_scores_v1.zip")
    parser.add_argument("--output", required=True, type=Path, help="new output path; existing files are never overwritten")
    parser.add_argument("--compare", type=Path, help="optional frozen S7 CSV to compare with the rebuilt output")
    args = parser.parse_args()

    for path, expected in ((args.rolling_zip, ROLLING_ZIP_SHA256), (args.archived_scores_zip, ARCHIVE_ZIP_SHA256)):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"input archive hash mismatch for {path.name}: {actual}")

    rolling_rows = read_member(args.rolling_zip, ROLLING_MEMBER)
    archive_rows = read_member(args.archived_scores_zip, ARCHIVE_MEMBER)
    methods: dict[tuple[str, str], dict[tuple[str, str, int], float]] = {}

    for row in rolling_rows:
        if row.get("forecast_status") != "FORECAST_COMPLETE":
            continue
        h = int(row["horizon_months"])
        target = row["target_month"]
        if h not in HORIZONS or not ("2023-01" <= target <= "2024-12"):
            continue
        origin = row["origin_month"]
        if add_month(origin, h) != target:
            raise ValueError(f"target/origin/horizon mismatch: {row}")
        key = (target, origin, h)
        model = row["model"]
        index = ("statistical", model)
        bucket = methods.setdefault(index, {})
        if key in bucket:
            raise ValueError(f"duplicate statistical key: {index}, {key}")
        bucket[key] = finite_float(row["absolute_error_pp"], "absolute_error_pp")

    for row in archive_rows:
        stage = row["archive_stage"]
        model = row["model_version"]
        origin = row["origin_month"]
        h = int(row["horizon_months"])
        if h not in HORIZONS:
            continue
        target = add_month(origin, h)
        if not ("2023-01" <= target <= "2024-12"):
            continue
        key = (target, origin, h)
        index = (stage, model)
        bucket = methods.setdefault(index, {})
        if key in bucket:
            raise ValueError(f"duplicate archived key: {index}, {key}")
        bucket[key] = finite_float(row["origin_mean_absolute_error_pp"], "origin_mean_absolute_error_pp")

    expected_method_names = {
        ("statistical", "AR(1)-rolling-expanding"),
        ("statistical", "No-change"),
        ("statistical", "SARIMA(1,0,0)x(1,0,0,12)-rolling-expanding"),
        ("statistical", "Seasonal-naive"),
        ("stage2", "B3"), ("stage2", "BEAH"),
        ("stage2", "Fixed_BaseEffect_recent3"), ("stage2", "Matched_MLP"),
        ("stage3", "BACE_aux"), ("stage3", "BACE_yoy"), ("stage3", "BEAH_reference"),
        ("stage3", "Direct_aux"), ("stage3", "Direct_yoy"),
        ("stage4", "GateOnly_fixed_base"), ("stage4", "Gated_YoY_MLP"), ("stage4", "MLP_matched"),
    }
    if set(methods) != expected_method_names:
        raise ValueError(f"method inventory mismatch: found {sorted(methods)}")
    for h in HORIZONS:
        expected = expected_keys(h)
        for method, observations in methods.items():
            actual = {key for key in observations if key[2] == h}
            if actual != expected:
                raise ValueError(f"key-set mismatch for {method}, h={h}: expected {len(expected)}, found {len(actual)}")

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for (family, method), observations in methods.items():
        for (target, origin, h), loss in observations.items():
            if (target, origin, h) in expected_keys(h):
                out_rows.append({
                    "family": family,
                    "method_version": method,
                    "target_month": target,
                    "origin_month": origin,
                    "horizon_months": str(h),
                    "absolute_error_pp": format(loss, ".12g"),
                })
    family_order = {"statistical": 0, "stage2": 1, "stage3": 2, "stage4": 3}
    out_rows.sort(key=lambda r: (family_order[r["family"]], r["method_version"], int(r["horizon_months"]), r["target_month"]))
    with args.output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Rebuilt {len(out_rows)} target-free loss rows across {len(methods)} methods and four horizons.")
    print(f"Output: {args.output}")
    print(f"SHA-256: {sha256(args.output)}")
    if args.compare:
        original = list(csv.DictReader(args.compare.open("r", newline="", encoding="utf-8-sig")))
        compare = list(csv.DictReader(args.output.open("r", newline="", encoding="utf-8-sig")))
        def normalized(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, int], float]:
            result = {}
            for r in rows:
                key = (r["family"], r["method_version"], r["target_month"], r["origin_month"], int(r["horizon_months"]))
                if key in result:
                    raise ValueError(f"duplicate comparison key: {key}")
                result[key] = finite_float(r["absolute_error_pp"], "comparison loss")
            return result
        a, b = normalized(original), normalized(compare)
        if set(a) != set(b):
            raise ValueError(f"comparison keys differ: original={len(a)}, rebuilt={len(b)}")
        maximum = max((abs(a[k] - b[k]) for k in a), default=0.0)
        if maximum > 1e-10:
            raise ValueError(f"comparison value mismatch: max_abs_diff={maximum}")
        print(f"Compared with frozen S7: {len(a)} keys match; max absolute difference={maximum:.3g}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)