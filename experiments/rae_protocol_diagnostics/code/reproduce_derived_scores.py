#!/usr/bin/env python3
"""Rebuild both target-free diagnostic score tables in an isolated directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    manifest = json.loads((HERE / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        if sha(HERE / name) != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    project = out / "project"
    exp = project / "revision_experiments_v2/rae_protocol_ablation_20261109"
    exp.mkdir(parents=True)
    for name in ["PROTOCOL.md", "PROTOCOL_PARTIAL_VINTAGE.md", "SCORING_PARTIAL_VINTAGE.md", "PROTOCOL_COMBINED_PARTIAL.md",
                 "analyze_archived_diagnostic.py", "verify_archived_diagnostic.py", "build_brent_vintage_map.py",
                 "score_brent_vintage.py", "verify_brent_vintage.py", "build_combined_map.py",
                 "score_combined_partial.py", "verify_combined_partial.py"]:
        shutil.copy2(HERE / name, exp / name)
    data = HERE / "inputs"
    mapping = {
        "s11_paired_losses.csv": "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/derived_scores_v2/paired_origin_losses.csv",
        "china_ten_model_losses.csv": "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a/score_v2_clean_20261031/origin_method_losses.csv",
        "china_ets_theta_losses.csv": "revision_experiments_v2/ets_theta_rolling_20261109/output/CN/score/new_origin_losses.csv",
        "timestral_losses.csv": "revision_experiments_v2/timestral_cpi_baseline_20261109/historical_scores/origin_losses.csv",
        "vintages.csv": "manuscript/ars_full_v2_4/reproducibility/matched_origin_2018_2024_v1/work/pit_audit_complete_rebuild_20261030/run_20261030T044200Z_pit_audit_v2/vintages.csv",
        "s11_timing_forecasts.csv": "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/server_outputs_v1/forecast_predictions.csv",
    }
    s11 = "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/inputs/"
    for name in ["b3_input_snapshots.csv", "cpi_input_snapshots.csv", "archived_b3_q50.csv", "scored_origin_horizon_keys.csv", "mask_recovery_map.csv"]:
        mapping[name] = s11 + name
    for name, rel in mapping.items():
        target = project / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / name, target)
    shutil.copytree(HERE / "vintage_forecasts", exp / "vintage_forecasts")
    shutil.copytree(HERE / "combined_forecasts", exp / "combined_forecasts")
    for name in ["analyze_archived_diagnostic.py", "verify_archived_diagnostic.py", "build_brent_vintage_map.py", "score_brent_vintage.py", "verify_brent_vintage.py", "build_combined_map.py", "score_combined_partial.py", "verify_combined_partial.py"]:
        subprocess.run([sys.executable, str(exp / name)], check=True)
    comparisons = {
        "results": ["model_scenario_metrics.csv", "b3_rank_changes.csv", "paired_origin_changes.csv", "paired_block_intervals.csv", "coverage.csv"],
        "vintage_scores": ["summary.csv", "paired_losses.csv", "paired_block_intervals.csv"],
        "vintage_map": ["brent_vintage_replacement_map.csv"],
        "combined_map": ["mask_recovery_map.csv"],
        "combined_scores": ["scenario_summary.csv", "paired_origin_losses.csv", "paired_block_intervals.csv"],
    }
    checked = 0
    for folder, names in comparisons.items():
        for name in names:
            expected = pd.read_csv(HERE / folder / name)
            actual = pd.read_csv(exp / folder / name)
            if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
                raise ValueError(f"Shape mismatch in {folder}/{name}")
            for col in expected.columns:
                if pd.api.types.is_numeric_dtype(expected[col]):
                    if not np.allclose(expected[col].to_numpy(float), actual[col].to_numpy(float), atol=1e-10, rtol=0, equal_nan=True):
                        raise ValueError(f"Numeric mismatch in {folder}/{name}: {col}")
                elif not expected[col].fillna("").astype(str).equals(actual[col].fillna("").astype(str)):
                    raise ValueError(f"Text mismatch in {folder}/{name}: {col}")
            checked += 1
    report = {"status": "PASS", "tables_compared": checked, "isolated_scoring_target_ledger_read": False,
              "historical_cpi_context_snapshots_copied_and_hashed": True, "p1_accessed": False,
              "note": "Scores rebuilt from archived predictions and derived residuals; model inference was not rerun."}
    (out / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
