#!/usr/bin/env python3
"""Score the independently generated held-out suite with frozen RAE only."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TRUTH_PATH = HERE / "results" / "heldout_truth.json"
RAE_PATH = HERE / "frozen_rae" / "pit_audit.py"
RESULTS = HERE / "results"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_rae():
    spec = importlib.util.spec_from_file_location("frozen_rae_under_test", RAE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen RAE module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def classify_errors(errors: list[str]) -> set[str]:
    text = "\n".join(errors).lower()
    out: set[str] = set()
    patterns = {
        "release_time": ("available_at", "released by origin", "available before origin",
                         "not available before origin", "timestamp must include", "invalid timestamp"),
        "vintage": ("vintage", "ambiguous latest"),
        "dependency": ("source observation month",),
        "eligibility": ("required input", "missing required", "origin model set mismatch",
                        "unknown model-origin", "origin sets differ", "coverage mismatch"),
        "mapping": ("target-origin-horizon", "target absent", "duplicate forecast key",
                    "unregistered horizon", "target month", "duplicate/empty target"),
    }
    for family, needles in patterns.items():
        if any(needle in text for needle in needles):
            out.add(family)
    return out


def main() -> int:
    truth = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
    auditor = load_frozen_rae()
    freeze = json.loads((HERE / "freeze_receipt.json").read_text(encoding="utf-8"))
    frozen_hash = sha(RAE_PATH)
    if frozen_hash != freeze["frozen_sha256"] or not freeze.get("byte_identical"):
        raise RuntimeError("frozen RAE snapshot does not match the pre-test freeze receipt")
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_dir = RESULTS / "audit_reports"
    output_dir.mkdir(exist_ok=True)
    all_rows = []
    exceptions = []
    durations = []
    family_truth = Counter()
    family_found = Counter()
    family_tp = Counter()
    family_fp = Counter()
    clean_n = fault_n = 0
    flagged_clean = flagged_faults = 0
    detected_fault_cases = 0
    localized_cases = 0
    total_fault_components = 0
    localized_components = 0
    by_type = defaultdict(lambda: {"cases": 0, "detected": 0,
                                   "all_expected_families_localized": 0})
    t0 = time.perf_counter()

    for label in truth["cases"]:
        case_id = label["case_id"]
        bundle = HERE / "cases" / case_id / "bundle.json"
        output_path = output_dir / f"{case_id}.json"
        started = time.perf_counter()
        try:
            report = auditor.run_audit(bundle, output_path, write_outputs=False)
            status = report.get("status", "ERROR")
            errors = list(report.get("errors", []))
        except Exception as exc:  # retain crashes as audit failures, never drop them
            status = "EXCEPTION"
            errors = [f"runner exception: {type(exc).__name__}: {exc}"]
            exceptions.append({"case_id": case_id, "exception": errors[0]})
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        detected = status != "PASS"
        found_families = classify_errors(errors)
        expected_families = set(label["fault_families"])
        type_row = by_type[label.get("case_type", "unknown")]
        type_row["cases"] += 1
        type_row["detected"] += int(detected)
        type_row["all_expected_families_localized"] += int(
            bool(expected_families) and expected_families.issubset(found_families))
        all_rows.append({
            "case_id": case_id,
            "expected_clean": label["expected_clean"],
            "case_type": label.get("case_type", "unknown"),
            "faults": ";".join(label["faults"]),
            "expected_families": ";".join(sorted(expected_families)),
            "status": status,
            "detected": detected,
            "localized_families": ";".join(sorted(found_families)),
            "expected_family_localization": expected_families.issubset(found_families),
            "error_count": len(errors),
            "elapsed_ms": round(elapsed * 1000, 4),
            "errors": " | ".join(errors),
        })
        if label["expected_clean"]:
            clean_n += 1
            flagged_clean += int(detected)
            for family in found_families:
                family_fp[family] += 1
        else:
            fault_n += 1
            flagged_faults += int(detected)
            detected_fault_cases += int(detected)
            localized_cases += int(bool(expected_families) and expected_families.issubset(found_families))
            components = len(label["faults"])
            total_fault_components += components
            component_localized = 0
            for family in expected_families:
                family_truth[family] += sum(1 for f in label["faults"]
                                            if _family_for_fault(f, label) == family)
                if family in found_families:
                    family_found[family] += 1
                    component_localized += sum(1 for f in label["faults"]
                                               if _family_for_fault(f, label) == family)
            localized_components += min(components, component_localized)

    total_s = time.perf_counter() - t0
    with (RESULTS / "heldout_case_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    tp = flagged_faults
    fp = flagged_clean
    fn = fault_n - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    clean_fpr = fp / clean_n if clean_n else 0.0
    family_rows = []
    for family in sorted(set(family_truth) | set(family_fp)):
        n = family_truth[family]
        found = family_found[family]
        family_rows.append({"fault_family": family, "fault_components": n,
                            "localized_components": found,
                            "localization_recall": found / n if n else None,
                            "false_positive_cases": family_fp[family]})
    with (RESULTS / "heldout_family_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)

    pbt = json.loads((RESULTS / "property_test_results.json").read_text(encoding="utf-8"))
    summary = {
        "experiment": "heldout_unknown_fault_generalization",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_rae_sha256": frozen_hash,
        "truth_manifest_sha256": sha(TRUTH_PATH),
        "heldout_cases": truth["cases_total"],
        "clean_controls": clean_n,
        "fault_cases": fault_n,
        "fault_case_detection": {"tp": tp, "fp": fp, "fn": fn,
                                  "precision": precision, "recall": recall, "f1": f1,
                                  "clean_control_false_positive_rate": clean_fpr},
        "family_localization": family_rows,
        "cases_by_type": {
            name: {**values,
                   "detection_rate": values["detected"] / values["cases"] if values["cases"] else None,
                   "all_family_localization_rate": (
                       values["all_expected_families_localized"] / values["cases"]
                       if values["cases"] else None)}
            for name, values in sorted(by_type.items())},
        "fault_cases_all_expected_families_localized": localized_cases,
        "composite_and_single_fault_components_localized": localized_components,
        "fault_components_total": total_fault_components,
        "property_based_tests": pbt,
        "exceptions": exceptions,
        "timing": {"total_seconds": total_s,
                   "mean_case_ms": (sum(durations) / len(durations) * 1000 if durations else 0),
                   "p95_case_ms": _percentile(durations, .95) * 1000 if durations else 0},
        "interpretation": "The held-out synthetic suite tests generalization beyond the earlier targeted fingerprint-mismatch injections. It does not validate real-world detection rates. The transitive dependency-edge family is outside the frozen RAE input contract and is reported as a limitation if not detected.",
    }
    (RESULTS / "heldout_generalization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _family_for_fault(fault: str, label: dict) -> str:
    mapping = {
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
    return mapping[fault]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


if __name__ == "__main__":
    raise SystemExit(main())
