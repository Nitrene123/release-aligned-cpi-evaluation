#!/usr/bin/env python3
"""Seeded property-based/metamorphic tests for the frozen RAE contract.

Uses only the Python standard library because Hypothesis is absent from the
reproducible environment. The generated inputs and invariants are explicit;
this harness does not claim Hypothesis shrinking or exhaustive exploration.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GEN_PATH = HERE / "generator" / "generate_heldout_suite.py"
RAE_PATH = HERE / "frozen_rae" / "pit_audit.py"
RESULT = HERE / "results" / "property_test_results.json"
SEED = 8349917
PER_GROUP = 125


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    gen = load_module("independent_generator_for_properties", GEN_PATH)
    rae = load_module("frozen_rae_for_properties", RAE_PATH)
    freeze = json.loads((HERE / "freeze_receipt.json").read_text(encoding="utf-8"))
    if sha(RAE_PATH) != freeze["frozen_sha256"] or not freeze.get("byte_identical"):
        raise RuntimeError("frozen RAE snapshot does not match the pre-test freeze receipt")
    counts = {"timezone_equivalence": {"trials": 0, "passed": 0, "failed": 0},
              "release_cutoff_boundary": {"trials": 0, "passed": 0, "failed": 0,
                                          "expected_pass": 0, "expected_reject": 0,
                                          "observed_pass_for_expected_pass": 0,
                                          "observed_reject_for_expected_reject": 0},
              "csv_format_and_row_permutation": {"trials": 0, "passed": 0, "failed": 0},
              "target_mapping_metamorphic": {"trials": 0, "passed": 0, "failed": 0,
                                              "valid_bundle_passed": 0,
                                              "mutated_bundle_rejected": 0}}
    failures = []
    audit_calls = 0
    trial_root = Path(tempfile.mkdtemp(prefix="rae_pbt_", dir=HERE / "results"))
    try:
        case_index = 0

        def build(variant: str = "canonical") -> Path:
            nonlocal case_index
            case_index += 1
            base = trial_root / f"trial_{case_index:05d}"
            gen.CASES = base
            case_dir, _ = gen.base_case(f"P{case_index:05d}", rng, variant)
            return case_dir

        def audit(case_dir: Path, suffix: str) -> str:
            nonlocal audit_calls
            audit_calls += 1
            output = trial_root / f"audit_{suffix}_{case_index:05d}.json"
            try:
                return rae.run_audit(case_dir / "bundle.json", output, write_outputs=False)["status"]
            except Exception as exc:
                return "EXCEPTION:" + type(exc).__name__

        # Metamorphic property: representing the same instant in different
        # explicit offsets must not change the eligibility result.
        for i in range(PER_GROUP):
            case_dir = build()
            _, origins = gen.load_rows(case_dir, "origins.csv")
            _, inputs = gen.load_rows(case_dir, "inputs.csv")
            _, vintages = gen.load_rows(case_dir, "vintages.csv")
            for row in origins:
                dt = parse(row["origin_timestamp"])
                offset = rng.choice((-660, -480, -300, 0, 330, 480, 540, 600, 660))
                row["origin_timestamp"] = gen.iso(dt, offset)
            for row in inputs:
                dt = parse(row["available_at"])
                offset = rng.choice((-660, -480, -300, 0, 330, 480, 540, 600, 660))
                row["available_at"] = gen.iso(dt, offset)
            for row in vintages:
                dt = parse(row["available_at"])
                offset = rng.choice((-660, -480, -300, 0, 330, 480, 540, 600, 660))
                row["available_at"] = gen.iso(dt, offset)
            gen.save_rows(case_dir, "origins.csv", list(origins[0]), origins)
            gen.save_rows(case_dir, "inputs.csv", list(inputs[0]), inputs)
            gen.save_rows(case_dir, "vintages.csv", list(vintages[0]), vintages)
            gen.refresh_forecast_hashes(case_dir)
            status = audit(case_dir, "tz_equiv")
            bucket = counts["timezone_equivalence"]
            bucket["trials"] += 1
            bucket["passed" if status == "PASS" else "failed"] += 1
            if status != "PASS" and len(failures) < 20:
                failures.append({"property": "timezone_equivalence", "case": case_index,
                                 "status": status})
            shutil.rmtree(case_dir.parent)

        # Strict cutoff property: only instants strictly before the origin pass.
        for i in range(PER_GROUP):
            delta = rng.choice((-1000, -100, -10, -1, 0, 1, 10, 100, 1000))
            case_dir = build()
            _, origins = gen.load_rows(case_dir, "origins.csv")
            _, inputs = gen.load_rows(case_dir, "inputs.csv")
            _, vintages = gen.load_rows(case_dir, "vintages.csv")
            origin = parse(origins[0]["origin_timestamp"])
            boundary = origin.astimezone(timezone.utc) + timedelta(microseconds=delta)
            selected = next(r for r in inputs if r["model"] == "BASE")
            selected["available_at"] = gen.iso(boundary, rng.choice((-420, 0, 330, 480)))
            matching = next(v for v in vintages if v["source_id"] == selected["source_id"] and
                            v["series_id"] == selected["series_id"] and
                            v["observation_month"] == selected["observation_month"] and
                            v["vintage_id"] == selected["vintage_id"])
            matching["available_at"] = selected["available_at"]
            gen.save_rows(case_dir, "inputs.csv", list(inputs[0]), inputs)
            gen.save_rows(case_dir, "vintages.csv", list(vintages[0]), vintages)
            gen.refresh_forecast_hashes(case_dir)
            status = audit(case_dir, "boundary")
            expected_pass = delta < 0
            ok = (status == "PASS") if expected_pass else (status != "PASS")
            bucket = counts["release_cutoff_boundary"]
            bucket["trials"] += 1
            bucket["passed" if ok else "failed"] += 1
            bucket["expected_pass" if expected_pass else "expected_reject"] += 1
            if expected_pass and status == "PASS":
                bucket["observed_pass_for_expected_pass"] += 1
            if not expected_pass and status != "PASS":
                bucket["observed_reject_for_expected_reject"] += 1
            if not ok and len(failures) < 20:
                failures.append({"property": "release_cutoff_boundary", "case": case_index,
                                 "delta_microseconds": delta, "expected_pass": expected_pass,
                                 "status": status})
            shutil.rmtree(case_dir.parent)

        # Parser invariance: BOM/CRLF, quoting, reordered columns and row order
        # should preserve the result for valid bundles.
        variants = ("bom_crlf", "quoted_crlf", "reordered_columns", "row_permutation")
        for i in range(PER_GROUP):
            variant = rng.choice(variants)
            case_dir = build(variant)
            status = audit(case_dir, "format")
            bucket = counts["csv_format_and_row_permutation"]
            bucket["trials"] += 1
            bucket["passed" if status == "PASS" else "failed"] += 1
            if status != "PASS" and len(failures) < 20:
                failures.append({"property": "csv_format_and_row_permutation", "case": case_index,
                                 "variant": variant, "status": status})
            shutil.rmtree(case_dir.parent)

        # Metamorphic mapping property: a correct horizon mapping passes; a
        # one-month target shift with otherwise unchanged rows is rejected.
        for i in range(PER_GROUP):
            case_dir = build()
            valid_status = audit(case_dir, "mapping_valid")
            fields, forecasts = gen.load_rows(case_dir, "forecasts.csv")
            target = next(r for r in forecasts if r["model"] == "BASE" and
                          int(r["horizon_months"]) == 3)
            y, m = map(int, target["target_month"].split("-"))
            m += 1
            if m > 12:
                y += 1; m = 1
            target["target_month"] = f"{y:04d}-{m:02d}"
            gen.save_rows(case_dir, "forecasts.csv", fields, forecasts)
            invalid_status = audit(case_dir, "mapping_mutated")
            ok = valid_status == "PASS" and invalid_status != "PASS"
            bucket = counts["target_mapping_metamorphic"]
            bucket["trials"] += 1
            bucket["passed" if ok else "failed"] += 1
            bucket["valid_bundle_passed"] += int(valid_status == "PASS")
            bucket["mutated_bundle_rejected"] += int(invalid_status != "PASS")
            if not ok and len(failures) < 20:
                failures.append({"property": "target_mapping_metamorphic", "case": case_index,
                                 "valid_status": valid_status, "mutated_status": invalid_status})
            shutil.rmtree(case_dir.parent)
    finally:
        shutil.rmtree(trial_root, ignore_errors=True)

    totals = {"trials": sum(v["trials"] for v in counts.values()),
              "passed": sum(v["passed"] for v in counts.values()),
              "failed": sum(v["failed"] for v in counts.values())}
    report = {
        "test_type": "seeded property-based and metamorphic contract tests",
        "generator": "Python standard-library random.Random; deterministic seed; no Hypothesis dependency",
        "seed": SEED,
        "per_property_trials": PER_GROUP,
        "total_trials": totals["trials"],
        "checker_executions": audit_calls,
        "passed": totals["passed"],
        "failed": totals["failed"],
        "properties": counts,
        "failure_examples": failures,
        "frozen_rae_sha256": sha(RAE_PATH),
        "limits": ["Seeded finite generation is not exhaustive.",
                   "No automatic counterexample shrinking is provided.",
                   "Results describe the stated schema and generated domain only."],
    }
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(totals["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
