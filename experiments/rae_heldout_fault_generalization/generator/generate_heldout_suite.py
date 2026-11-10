#!/usr/bin/env python3
"""Independent deterministic generator for the held-out RAE fault suite.

This generator does not import the RAE checker or reuse the earlier injection
benchmark's case-generation code. Ground-truth labels are written before the
separate runner imports the frozen checker.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parents[1]
CASES = HERE / "cases"
TRUTH = HERE / "results" / "heldout_truth.json"
SEED = 20261111
MODELS = ("BASE", "MULTI")
HORIZONS = (1, 3, 6, 12)
FEATURES = ("cpi_yoy", "fx_return")
ORIGIN_ID = "O0001"
ORIGIN_UTC = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)


def month_index(label: str) -> int:
    y, m = map(int, label.split("-"))
    return y * 12 + m - 1


def month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def shift_month(label: str, n: int) -> str:
    return month_label(month_index(label) + n)


def iso(dt: datetime, offset_minutes: int | None = None) -> str:
    if offset_minutes is None:
        return dt.isoformat().replace("+00:00", "Z")
    return dt.astimezone(timezone(timedelta(minutes=offset_minutes))).isoformat()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_input_hash(rows: list[dict[str, str]]) -> str:
    # Independent reimplementation of the documented canonical row contract.
    payload = json.dumps(
        sorted(rows, key=lambda r: (r["feature"], r["observation_month"],
                                    r.get("source_observation_month", ""))),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]], *, variant: str = "canonical") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    if variant == "reordered_columns":
        fields = fields[::2] + fields[1::2]
    line_ending = "\r\n" if variant in {"bom_crlf", "quoted_crlf"} else "\n"
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator=line_ending,
                            quoting=csv.QUOTE_ALL if variant == "quoted_crlf" else csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(rows)
    data = buf.getvalue()
    encoding = "utf-8-sig" if variant == "bom_crlf" else "utf-8"
    path.write_text(data, encoding=encoding, newline="")


def base_case(case_id: str, rng: random.Random, format_variant: str = "canonical") -> tuple[Path, dict[str, str]]:
    case_dir = CASES / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    origin_offset = rng.choice((-660, -420, -300, 0, 330, 480, 540, 600, 660))
    origin_stamp = iso(ORIGIN_UTC, origin_offset)
    origin_month = "2024-01"
    origins = [{"model": m, "origin_id": ORIGIN_ID, "origin_month": origin_month,
                "origin_timestamp": origin_stamp} for m in MODELS]

    inputs: list[dict[str, str]] = []
    vintages: list[dict[str, str]] = []
    input_months = ("2023-11", "2023-12")
    for model in MODELS:
        for f_idx, feature in enumerate(FEATURES):
            for m_idx, observation_month in enumerate(input_months):
                value = f"{(f_idx + 1) * 0.25 + m_idx * 0.125:.6f}"
                available_dt = ORIGIN_UTC - timedelta(days=2 + f_idx, minutes=m_idx * 7)
                available_at = iso(available_dt, rng.choice((-600, -300, 0, 330, 480, 600)))
                source_hash = digest(f"{feature}|{observation_month}|v1|{value}")
                row = {
                    "model": model, "origin_id": ORIGIN_ID, "feature": feature,
                    "observation_month": observation_month,
                    "source_observation_month": observation_month,
                    "value": value, "masked": "false", "mask_reason": "",
                    "source_id": "SYNTH", "series_id": feature, "vintage_id": "v1",
                    "available_at": available_at, "source_sha256": source_hash,
                }
                inputs.append(row)
                # The vintage registry is source-level and therefore contains
                # one row per series/month, shared by both model configurations.
                if model == MODELS[0]:
                    vintages.append({
                        "source_id": "SYNTH", "series_id": feature,
                        "observation_month": observation_month, "vintage_id": "v1",
                        "available_at": available_at, "value": value,
                        "source_sha256": source_hash,
                    })

    targets = []
    forecasts = []
    target_release_by_month: dict[str, datetime] = {}
    for h in HORIZONS:
        target_month = shift_month(origin_month, h)
        release_dt = datetime(month_index(shift_month(target_month, 1)) // 12,
                              month_index(shift_month(target_month, 1)) % 12 + 1,
                              10, 12, 0, tzinfo=timezone.utc)
        targets.append({"target_month": target_month, "first_release_at": iso(release_dt)})
        target_release_by_month[target_month] = release_dt
    for model in MODELS:
        model_inputs = [r for r in inputs if r["model"] == model and r["origin_id"] == ORIGIN_ID]
        input_hash = canonical_input_hash(model_inputs)
        for h in HORIZONS:
            forecasts.append({"model": model, "origin_id": ORIGIN_ID,
                              "horizon_months": str(h),
                              "target_month": shift_month(origin_month, h),
                              "prediction": f"{0.2 + h / 100:.6f}",
                              "input_hash": input_hash})

    config = {
        "origins_file": "origins.csv", "inputs_file": "inputs.csv",
        "vintages_file": "vintages.csv", "targets_file": "targets.csv",
        "forecasts_file": "forecasts.csv", "expected_models": list(MODELS),
        "expected_horizons": list(HORIZONS),
        "require_identical_model_origin_sets": True,
        "required_features_by_model": {m: list(FEATURES) for m in MODELS},
        "scope_note": "synthetic held-out schema generalization test; no target values",
        "vintage_registry_scope": "complete synthetic registry for this bundle",
    }
    (case_dir / "bundle.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if format_variant == "row_permutation":
        for rows in (origins, inputs, vintages, targets, forecasts):
            rng.shuffle(rows)
    for name, rows in (("origins.csv", origins), ("inputs.csv", inputs),
                       ("vintages.csv", vintages), ("targets.csv", targets),
                       ("forecasts.csv", forecasts)):
        write_csv(case_dir / name, rows, variant=format_variant)
    return case_dir, config


def load_rows(case_dir: Path, filename: str) -> tuple[list[str], list[dict[str, str]]]:
    with (case_dir / filename).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def save_rows(case_dir: Path, filename: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    write_csv(case_dir / filename, rows)


def refresh_forecast_hashes(case_dir: Path) -> None:
    _, inputs = load_rows(case_dir, "inputs.csv")
    _, forecasts = load_rows(case_dir, "forecasts.csv")
    for forecast in forecasts:
        rows = [r for r in inputs if r["model"] == forecast["model"] and
                r["origin_id"] == forecast["origin_id"]]
        forecast["input_hash"] = canonical_input_hash(rows)
    fields = list(forecasts[0])
    save_rows(case_dir, "forecasts.csv", fields, forecasts)


def mutate_release_equal(case_dir: Path, rng: random.Random) -> None:
    _, inputs = load_rows(case_dir, "inputs.csv")
    _, origins = load_rows(case_dir, "origins.csv")
    target = next(r for r in inputs if r["model"] == "BASE")
    origin = next(r for r in origins if r["model"] == target["model"])
    # Same instant, represented with a different offset; strict admissibility rejects equality.
    dt = datetime.fromisoformat(origin["origin_timestamp"].replace("Z", "+00:00"))
    target["available_at"] = iso(dt.astimezone(timezone.utc), 330)
    save_rows(case_dir, "inputs.csv", list(inputs[0]), inputs)


def mutate_release_after(case_dir: Path, rng: random.Random) -> None:
    _, inputs = load_rows(case_dir, "inputs.csv")
    _, origins = load_rows(case_dir, "origins.csv")
    target = next(r for r in inputs if r["model"] == "BASE")
    origin = next(r for r in origins if r["model"] == target["model"])
    dt = datetime.fromisoformat(origin["origin_timestamp"].replace("Z", "+00:00"))
    target["available_at"] = iso(dt.astimezone(timezone.utc) + timedelta(microseconds=1), -300)
    save_rows(case_dir, "inputs.csv", list(inputs[0]), inputs)


def mutate_naive_timestamp(case_dir: Path, rng: random.Random) -> None:
    _, inputs = load_rows(case_dir, "inputs.csv")
    target = next(r for r in inputs if r["model"] == "BASE")
    target["available_at"] = "2024-01-15T11:59:59"
    save_rows(case_dir, "inputs.csv", list(inputs[0]), inputs)


def mutate_duplicate_vintage(case_dir: Path, rng: random.Random) -> None:
    fields, vintages = load_rows(case_dir, "vintages.csv")
    vintages.append(dict(vintages[0]))
    save_rows(case_dir, "vintages.csv", fields, vintages)


def mutate_stale_vintage(case_dir: Path, rng: random.Random) -> None:
    fields, vintages = load_rows(case_dir, "vintages.csv")
    chosen = vintages[0]
    later = dict(chosen)
    later["vintage_id"] = "v2"
    later["available_at"] = "2024-01-15T10:00:00Z"
    later["value"] = f"{float(chosen['value']) + 0.01:.6f}"
    later["source_sha256"] = digest("v2|" + later["series_id"] + "|" + later["observation_month"] + "|" + later["value"])
    vintages.append(later)
    save_rows(case_dir, "vintages.csv", fields, vintages)


def mutate_direct_dependency(case_dir: Path, rng: random.Random) -> None:
    fields, inputs = load_rows(case_dir, "inputs.csv")
    _, vintages = load_rows(case_dir, "vintages.csv")
    row = next(r for r in inputs if r["model"] == "BASE")
    future_source_month = shift_month(row["observation_month"], 1)
    row["source_observation_month"] = future_source_month
    matching = dict(next(v for v in vintages if v["series_id"] == row["series_id"] and
                         v["observation_month"] == row["observation_month"]))
    matching["observation_month"] = future_source_month
    matching["vintage_id"] = "dep1"
    matching["available_at"] = "2024-01-10T12:00:00Z"
    vintages.append(matching)
    save_rows(case_dir, "inputs.csv", fields, inputs)
    save_rows(case_dir, "vintages.csv", list(vintages[0]), vintages)


def mutate_transitive_dependency(case_dir: Path, rng: random.Random) -> None:
    # The direct source month remains admissible. Independent truth records a future
    # upstream parent in a provenance edge file the frozen RAE schema does not read.
    rows = [
        {"child_node": "derived_index@2023-12", "parent_node": "raw_index@2023-12"},
        {"child_node": "raw_index@2023-12", "parent_node": "revision@2024-01"},
    ]
    write_csv(case_dir / "dependency_edges.csv", rows)


def mutate_missing_required(case_dir: Path, rng: random.Random) -> None:
    fields, inputs = load_rows(case_dir, "inputs.csv")
    inputs = [r for r in inputs if not (r["model"] == "MULTI" and r["feature"] == "fx_return")]
    save_rows(case_dir, "inputs.csv", fields, inputs)


def mutate_model_origin_mismatch(case_dir: Path, rng: random.Random) -> None:
    fields, origins = load_rows(case_dir, "origins.csv")
    origins = [r for r in origins if r["model"] != "MULTI"]
    save_rows(case_dir, "origins.csv", fields, origins)


def mutate_target_mapping(case_dir: Path, rng: random.Random) -> None:
    fields, forecasts = load_rows(case_dir, "forecasts.csv")
    target = next(r for r in forecasts if r["model"] == "BASE" and int(r["horizon_months"]) == 3)
    target["target_month"] = shift_month(target["target_month"], 1)
    save_rows(case_dir, "forecasts.csv", fields, forecasts)


def mutate_duplicate_forecast(case_dir: Path, rng: random.Random) -> None:
    fields, forecasts = load_rows(case_dir, "forecasts.csv")
    forecasts.append(dict(forecasts[0]))
    save_rows(case_dir, "forecasts.csv", fields, forecasts)


MUTATORS: dict[str, Callable[[Path, random.Random], None]] = {
    "release_equal_instant_timezone_offset": mutate_release_equal,
    "release_after_cutoff_microsecond": mutate_release_after,
    "timezone_missing_offset": mutate_naive_timestamp,
    "duplicate_vintage_identity": mutate_duplicate_vintage,
    "stale_vintage_selection": mutate_stale_vintage,
    "direct_dependency_future_source": mutate_direct_dependency,
    "transitive_dependency_future_parent": mutate_transitive_dependency,
    "required_feature_missing": mutate_missing_required,
    "model_origin_eligibility_mismatch": mutate_model_origin_mismatch,
    "target_horizon_mapping_error": mutate_target_mapping,
    "duplicate_forecast_key": mutate_duplicate_forecast,
}

FAULT_FAMILY = {
    "release_equal_instant_timezone_offset": "release_time",
    "release_after_cutoff_microsecond": "release_time",
    "timezone_missing_offset": "release_time",
    "duplicate_vintage_identity": "vintage",
    "stale_vintage_selection": "vintage",
    "direct_dependency_future_source": "dependency",
    "transitive_dependency_future_parent": "dependency",
    "required_feature_missing": "eligibility",
    "model_origin_eligibility_mismatch": "eligibility",
    "target_horizon_mapping_error": "mapping",
    "duplicate_forecast_key": "mapping",
}


def build_suite() -> dict:
    if CASES.exists():
        shutil.rmtree(CASES)
    CASES.mkdir(parents=True)
    rng = random.Random(SEED)
    truth_cases = []
    case_no = 1

    # Clean held-out controls: varied serialization and timezone encodings.
    clean_variants = (["canonical"] * 10 + ["bom_crlf"] * 10 +
                      ["quoted_crlf"] * 10 + ["reordered_columns"] * 10 +
                      ["row_permutation"] * 10)
    for variant in clean_variants:
        cid = f"H{case_no:04d}"
        folder, _ = base_case(cid, rng, variant)
        truth_cases.append({"case_id": cid, "expected_clean": True,
                            "case_type": "clean_control",
                            "faults": [], "fault_families": [],
                            "format_variant": variant,
                            "independent_truth_source": "generator control definition"})
        case_no += 1

    families = {
        "release_time": ["release_equal_instant_timezone_offset", "release_after_cutoff_microsecond",
                         "timezone_missing_offset"],
        "vintage": ["duplicate_vintage_identity", "stale_vintage_selection"],
        "dependency": ["direct_dependency_future_source", "transitive_dependency_future_parent"],
        "eligibility": ["required_feature_missing", "model_origin_eligibility_mismatch"],
        "mapping": ["target_horizon_mapping_error", "duplicate_forecast_key"],
    }
    for family, recipes in families.items():
        for j in range(24):
            cid = f"H{case_no:04d}"
            folder, _ = base_case(cid, rng, "canonical")
            recipe = recipes[j % len(recipes)]
            MUTATORS[recipe](folder, rng)
            refresh_forecast_hashes(folder)
            truth_cases.append({"case_id": cid, "expected_clean": False,
                                "case_type": "single_fault",
                                "faults": [recipe], "fault_families": [family],
                                "format_variant": "canonical",
                                "independent_truth_source": "held-out mutation recipe"})
            case_no += 1

    # Composite faults use pairings not present as the single-fault-only template.
    composite_pairs = [
        ("release_after_cutoff_microsecond", "target_horizon_mapping_error"),
        ("duplicate_vintage_identity", "required_feature_missing"),
        ("direct_dependency_future_source", "duplicate_forecast_key"),
        ("stale_vintage_selection", "model_origin_eligibility_mismatch"),
        ("transitive_dependency_future_parent", "release_equal_instant_timezone_offset"),
    ]
    for pair_index, pair in enumerate(composite_pairs):
        for _ in range(6):
            cid = f"H{case_no:04d}"
            folder, _ = base_case(cid, rng, rng.choice(("bom_crlf", "quoted_crlf", "reordered_columns")))
            for recipe in pair:
                MUTATORS[recipe](folder, rng)
            refresh_forecast_hashes(folder)
            fault_families = sorted({FAULT_FAMILY[x] for x in pair})
            truth_cases.append({"case_id": cid, "expected_clean": False,
                                "case_type": "compound_fault",
                                "faults": list(pair), "fault_families": fault_families,
                                "format_variant": "mixed_format_with_composite_fault",
                                "independent_truth_source": "held-out compound recipe"})
            case_no += 1

    out = {"suite": "RAE held-out fault generalization",
           "seed": SEED, "cases_total": len(truth_cases),
           "clean_controls": sum(x["expected_clean"] for x in truth_cases),
           "fault_cases": sum(not x["expected_clean"] for x in truth_cases),
           "cases": truth_cases,
           "truth_generation_note": "Truth labels derive from the independent generator recipe and are saved before any RAE run."}
    TRUTH.parent.mkdir(parents=True, exist_ok=True)
    TRUTH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    result = build_suite()
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
