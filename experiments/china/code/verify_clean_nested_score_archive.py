#!/usr/bin/env python3
"""Target-free integrity and arithmetic audit for the corrected rolling-head score."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "reproducibility" / "matched_origin_2018_2026_unified_v1" / "results"
BASE_RUN = RESULTS / "complete_run_v1" / "runs" / "run_20261030_unified_full_v2"
NESTED = RESULTS / "nested_validation_clean_v1"
OLD_PREFLIGHT = NESTED / "preflight_20261030T190741Z"
CLEAN_RUN = NESTED / "run_20261030T190812Z"
ASSEMBLY = NESTED / "score_input_20261031_v1"
SCORED = NESTED / "score_v1_clean_20261031"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    mismatches = []
    for name, expected in files.items():
        candidate = path.parent / name
        actual = sha256(candidate) if candidate.is_file() else None
        if actual != expected:
            mismatches.append(name)
    if mismatches:
        raise AssertionError(f"manifest mismatch at {path}: {mismatches}")
    return {"path": str(path.relative_to(ROOT)), "verified": len(files), "mismatches": 0}


def main() -> None:
    base_manifest = verify_manifest(BASE_RUN / "sealed_manifest.json")
    clean_manifest = verify_manifest(CLEAN_RUN / "sealed_manifest.json")
    assembly_manifest = verify_manifest(ASSEMBLY / "sealed_manifest.json")
    score_manifest = verify_manifest(SCORED / "score_manifest.json")

    clean_state = json.loads((CLEAN_RUN / "run_state.json").read_text(encoding="utf-8"))
    score_state = json.loads((SCORED / "score_run_metadata.json").read_text(encoding="utf-8"))
    if clean_state.get("P1_accessed") is not False or score_state.get("p1_accessed") is not False:
        raise AssertionError("P1 isolation metadata is not false")
    if clean_state.get("fit_invocations") != 1650:
        raise AssertionError("corrected fit invocation count differs from 1,650")

    # Confirm the reported old overlap directly from the target-free, packet-derived audit.
    old_audit = rows(OLD_PREFLIGHT / "inner_validation_target_audit.csv")
    clean_audit = rows(CLEAN_RUN / "inner_validation_target_audit.csv")
    if len(old_audit) != 110 or len(clean_audit) != 110:
        raise AssertionError("expected 110 outer-origin audit rows")
    validation_month_instances = sum(len(r["validation_target_months"].split(";")) for r in old_audit)
    overlap_month_instances = sum(
        len(r["overlapping_training_target_months_before_filter"].split(";"))
        if r["overlapping_training_target_months_before_filter"] else 0
        for r in old_audit
    )
    overlap_cells = sum(int(r["overlapping_training_label_cells"]) for r in old_audit)
    exact_overlap_values = sum(int(r["exact_value_matches_on_overlapping_cells"]) for r in old_audit)
    remaining_overlap_cells = sum(int(r["train_validation_overlap_after_filter"]) for r in clean_audit)
    remaining_late_cells = sum(int(r["dropped_release_at_or_after_validation_start"]) for r in clean_audit)
    if (validation_month_instances, overlap_month_instances, overlap_cells, exact_overlap_values) != (1320, 900, 1182, 1182):
        raise AssertionError("pre-correction overlap audit differs from reported counts")
    if remaining_overlap_cells != 0:
        raise AssertionError("post-correction training/validation target overlap is nonzero")

    corrected = rows(CLEAN_RUN / "corrected_head_forecasts_unscored.csv")
    if len(corrected) != 5940:
        raise AssertionError("corrected head forecast row count differs from 5,940")
    forecast_keys = [(r["model"], r["seed"], r["origin_id"], r["target_month"], r["horizon_months"]) for r in corrected]
    if len(set(forecast_keys)) != len(forecast_keys):
        raise AssertionError("duplicate corrected head forecast keys")
    if not all(math.isfinite(float(r["prediction"])) for r in corrected):
        raise AssertionError("non-finite corrected prediction")

    old_forecasts = rows(BASE_RUN / "model_forecasts_unscored.csv")
    assembled = rows(ASSEMBLY / "model_forecasts_unscored.csv")
    unchanged = {"AR(1)", "B3 multivariate", "Fixed candidate", "No-change", "SARIMA", "Seasonal-naive", "TimesFM-3 target-only"}
    key = lambda r: (r["model"], r["seed"], r["origin_id"], r["target_month"], r["horizon_months"])
    old_by_key = {key(r): r for r in old_forecasts if r["model"] in unchanged}
    assembled_by_key = {key(r): r for r in assembled if r["model"] in unchanged}
    if old_by_key != assembled_by_key:
        raise AssertionError("one or more non-head forecast rows changed during assembly")
    if len(assembled) != 8856:
        raise AssertionError("assembled score-input row count differs from 8,856")

    losses = rows(SCORED / "origin_method_losses.csv")
    summary = rows(SCORED / "score_summary_by_method_horizon.csv")
    pairwise = rows(SCORED / "pairwise_method_comparisons.csv")
    target_audit = rows(SCORED / "target_access_audit.csv")
    if (len(losses), len(summary), len(pairwise), len(target_audit)) != (4104, 40, 180, 105):
        raise AssertionError("scored output row counts differ from the frozen protocol")
    if any(r["raw_target_value_included"].lower() != "false" for r in target_audit):
        raise AssertionError("raw target values are marked as present in the target audit")

    # Recompute each MAE and exact all-ten common-origin count from the public derived losses.
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    origins: dict[tuple[str, int], set[str]] = defaultdict(set)
    loss_lookup: dict[tuple[str, int, str], float] = {}
    for r in losses:
        k = (r["model"], int(r["horizon_months"]))
        value = float(r["mean_seed_absolute_error_pp"])
        grouped[k].append(value)
        origins[k].add(r["origin_id"])
        loss_lookup[(r["model"], int(r["horizon_months"]), r["origin_id"])] = value
    summary_by_key = {(r["model"], int(r["horizon_months"])): r for r in summary}
    for k, values in grouped.items():
        rebuilt = sum(values) / len(values)
        saved = float(summary_by_key[k]["native_MAE_pp"])
        if abs(rebuilt - saved) > 1e-9:
            raise AssertionError(f"MAE mismatch at {k}: {rebuilt} vs {saved}")
    common_counts = {}
    for h in (1, 3, 6, 12):
        common = set.intersection(*(origins[(model, h)] for model, _ in grouped if _ == h))
        common_counts[str(h)] = len(common)
    if common_counts != {"1": 99, "3": 99, "6": 99, "12": 99}:
        raise AssertionError(f"all-method common-origin counts differ: {common_counts}")

    # Rebuild every paired contrast and lower/tie/higher count from origin losses.
    rebuilt_pairwise = 0
    for saved in pairwise:
        h = int(saved["horizon_months"])
        a, b = saved["method_a"], saved["method_b"]
        common = origins[(a, h)] & origins[(b, h)]
        diffs = [loss_lookup[(a, h, oid)] - loss_lookup[(b, h, oid)] for oid in sorted(common)]
        if int(saved["pairwise_common_origins"]) != len(common):
            raise AssertionError(f"paired origin count mismatch at {(h, a, b)}")
        if abs(sum(diffs) / len(diffs) - float(saved["mean_absolute_loss_difference_a_minus_b_pp"])) > 1e-9:
            raise AssertionError(f"paired mean difference mismatch at {(h, a, b)}")
        a_wins = sum(d < -1e-12 for d in diffs)
        ties = sum(abs(d) <= 1e-12 for d in diffs)
        b_wins = sum(d > 1e-12 for d in diffs)
        if (a_wins, ties, b_wins) != (
            int(saved["a_lower_loss_origins"]), int(saved["ties"]), int(saved["b_lower_loss_origins"])
        ):
            raise AssertionError(f"paired lower/tie/higher counts mismatch at {(h, a, b)}")
        rebuilt_pairwise += 1
    if rebuilt_pairwise != 180:
        raise AssertionError(f"expected to rebuild 180 paired contrasts, got {rebuilt_pairwise}")

    result = {
        "status": "PASS_TARGET_FREE_ARCHIVE_AND_SCORE_AUDIT",
        "target_values_read_by_this_audit": False,
        "p1_accessed": False,
        "manifests": [base_manifest, clean_manifest, assembly_manifest, score_manifest],
        "pre_correction_inner_validation": {
            "outer_origins": len(old_audit),
            "validation_target_month_instances": validation_month_instances,
            "overlapped_target_month_instances": overlap_month_instances,
            "exact_reused_training_label_cells": overlap_cells,
            "exact_numeric_matches": exact_overlap_values,
            "overlap_rate": overlap_month_instances / validation_month_instances,
        },
        "corrected_refit": {
            "outer_origins": len(clean_audit),
            "fit_invocations": clean_state["fit_invocations"],
            "forecast_rows": len(corrected),
            "post_filter_target_month_overlap_cells": remaining_overlap_cells,
            "inner_training_cells_dropped_for_release_cutoff": remaining_late_cells,
            "non_head_forecast_rows_preserved_exactly": len(old_by_key),
            "assembled_forecast_rows": len(assembled),
        },
        "score_audit": {
            "origin_method_losses": len(losses),
            "method_horizon_maes": len(summary),
            "pairwise_rows": len(pairwise),
            "pairwise_contrasts_rebuilt": rebuilt_pairwise,
            "target_audit_rows": len(target_audit),
            "all_ten_common_origins_by_horizon": common_counts,
            "mae_recomputed_from_origin_losses": True,
        },
    }
    out = NESTED / "post_score_audit_20261031.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
