#!/usr/bin/env python3
"""Score frozen strict/backfill predictions without exporting target values."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qstats(values):
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75)),
        "min": float(np.min(x)), "max": float(np.max(x)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--forecast-dir", type=Path, required=True)
    ap.add_argument("--target-panel", type=Path, required=True,
                    help="Local first-release CPI panel; target values are read but never copied to outputs")
    ap.add_argument("--archived-losses", type=Path, required=True,
                    help="Target-free B3 origin losses from the frozen score archive")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    inputs = args.input_dir
    fmeta = json.loads((args.forecast_dir / "inference_metadata.json").read_text(encoding="utf-8"))
    if fmeta.get("status") != "STRICT_REPLAY_WITHIN_PREDECLARED_Q50_TOLERANCE":
        raise RuntimeError("Forecast archive does not pass strict replay")
    forecast_path = args.forecast_dir / "forecast_predictions.csv"
    if sha256(forecast_path) != fmeta["forecast_predictions_sha256"]:
        raise RuntimeError("Forecast file hash does not match inference metadata")
    target_hash = sha256(args.target_panel)
    forecasts = pd.read_csv(forecast_path, dtype={"origin_month": str})
    keys = pd.read_csv(inputs / "scored_origin_horizon_keys.csv", dtype={"target_month": str})
    target_panel = pd.read_csv(args.target_panel, dtype={"reference_month": str})
    archived = pd.read_csv(args.archived_losses, dtype={"target_month": str})
    recovery = pd.read_csv(inputs / "mask_recovery_map.csv", dtype={"origin_month": str})

    if forecasts.duplicated(["variant", "origin_id", "horizon_months"]).any():
        raise ValueError("Duplicate forecast key")
    expected = {(str(r.origin_id), int(r.horizon_months)) for r in keys.itertuples(index=False)}
    if any(len(g) != 99 for _, g in keys.groupby("horizon_months")):
        raise ValueError("Expected 99 scored pairs per horizon")
    if archived.duplicated(["origin_id", "horizon_months"]).any():
        raise ValueError("Duplicate archived loss key")
    target_map = target_panel.set_index("reference_month").first_release_yoy_pct.to_dict()
    if target_panel.reference_month.duplicated().any():
        raise ValueError("Duplicate first-release target month")
    ref = pd.read_csv(inputs / "archived_b3_q50.csv", dtype={"target_month": str})
    ref["horizon_months"] = ref.horizon_months.astype(int)
    ref_map = {(str(r.origin_id), int(r.horizon_months)): float(r.archived_q50)
               for r in ref.itertuples(index=False)}
    archived["horizon_months"] = archived.horizon_months.astype(int)
    archived["seed_mean_prediction_minus_target_pp"] = pd.to_numeric(
        archived.seed_mean_prediction_minus_target_pp, errors="raise"
    )
    archived_map = {(str(r.origin_id), int(r.horizon_months)): r
                    for r in archived.itertuples(index=False)}
    q50 = forecasts[forecasts.horizon_months.isin([1, 3, 6, 12])].copy()
    if set(q50.variant) != {"strict_replay", "post_origin_backfill"}:
        raise ValueError("Expected strict and post-origin forecast variants")
    q50["horizon_months"] = q50.horizon_months.astype(int)
    by_variant = {}
    for variant in ("strict_replay", "post_origin_backfill"):
        part = q50[q50.variant.eq(variant)]
        by_variant[variant] = {
            (str(r.origin_id), int(r.horizon_months)): float(r.point_forecast_q50)
            for r in part.itertuples(index=False)
        }
    strict_keys = set(by_variant["strict_replay"])
    backfill_keys = set(by_variant["post_origin_backfill"])
    if strict_keys != backfill_keys or not expected.issubset(strict_keys):
        raise ValueError("Strict, counterfactual, and scored forecast keys do not align")

    recovery_origins = set(recovery.origin_id.astype(str))
    rows = []
    target_panel_matches = 0
    target_panel_fallbacks = []
    for r in keys.itertuples(index=False):
        oid, h, tm = str(r.origin_id), int(r.horizon_months), str(r.target_month)
        # Check that the frozen key maps the horizon to its intended target month.
        y, mo = map(int, str(r.origin_month).split("-")[:2])
        z = y * 12 + mo - 1 + h
        expected_tm = f"{z // 12:04d}-{z % 12 + 1:02d}"
        if tm != expected_tm:
            raise ValueError(f"Origin/horizon target mapping mismatch at {oid}/h{h}")
        archived_row = archived_map[(oid, h)]
        # The archived first-release panel has one blank final row. Recover that
        # target transiently from the already-scored B3 q50 and signed residual.
        # The value is never written to disk.
        implied_target = ref_map[(oid, h)] - float(archived_row.seed_mean_prediction_minus_target_pp)
        panel_value = target_map.get(tm)
        if panel_value is None or pd.isna(panel_value):
            target = implied_target
            target_panel_fallbacks.append(tm)
        else:
            target = float(panel_value)
            if abs(target - implied_target) > 1e-7:
                raise ValueError(f"First-release panel and frozen signed residual disagree at {tm}")
            target_panel_matches += 1
        s = by_variant["strict_replay"][(oid, h)]
        c = by_variant["post_origin_backfill"][(oid, h)]
        se, ce = abs(s - target), abs(c - target)
        rows.append({
            "origin_id": oid, "origin_month": str(r.origin_month), "origin_year": str(r.origin_month)[:4],
            "target_month": tm, "horizon_months": h,
            "strict_q50": s, "post_origin_q50": c,
            "strict_absolute_error_pp": se, "post_origin_absolute_error_pp": ce,
            "paired_absolute_error_change_pp": ce - se,
            "recovery_applied_to_origin": oid in recovery_origins,
        })

    paired = pd.DataFrame(rows)
    archived_loss_map = {(str(r.origin_id), int(r.horizon_months)): float(r.mean_seed_absolute_error_pp)
                         for r in archived.itertuples(index=False)}
    replay_differences = [
        abs(r.strict_absolute_error_pp - archived_loss_map[(r.origin_id, int(r.horizon_months))])
        for r in paired.itertuples(index=False)
    ]
    max_loss_rebuild = max(replay_differences)
    if max_loss_rebuild > 1e-7:
        raise RuntimeError(f"Strict replay did not rebuild archived B3 losses: max={max_loss_rebuild}")

    summary_rows, annual_rows, loyo_rows, forecast_change_rows, concentration_rows = [], [], [], [], []
    for h, g in paired.groupby("horizon_months", sort=True):
        d = g.paired_absolute_error_change_pp.to_numpy(dtype=float)
        strict_mae = float(g.strict_absolute_error_pp.mean())
        backfill_mae = float(g.post_origin_absolute_error_pp.mean())
        ties = int(np.isclose(d, 0.0, atol=1e-12).sum())
        improved = int((d < -1e-12).sum())
        worsened = int((d > 1e-12).sum())
        absd = np.abs(d)
        top3 = float(np.sort(absd)[-3:].sum() / absd.sum()) if absd.sum() else 0.0
        dist = qstats(d)
        summary_rows.append({
            "horizon_months": int(h), "n_paired_origins": int(len(g)),
            "strict_replay_mae_pp": strict_mae,
            "post_origin_backfill_mae_pp": backfill_mae,
            "mae_change_pp": backfill_mae - strict_mae,
            "relative_mae_change_percent": 100.0 * (backfill_mae - strict_mae) / strict_mae,
            "improved_origins": improved, "worsened_origins": worsened, "tied_origins": ties,
            "delta_mean_pp": dist["mean"], "delta_median_pp": dist["median"],
            "delta_q25_pp": dist["q25"], "delta_q75_pp": dist["q75"],
            "delta_min_pp": dist["min"], "delta_max_pp": dist["max"],
            "top_three_share_of_absolute_changes": top3,
            "forecast_changed_origins": int((np.abs(g.post_origin_q50 - g.strict_q50) > 1e-12).sum()),
            "origins_with_verified_recovery_input": int(g.recovery_applied_to_origin.sum()),
        })
        for year, gy in g.groupby("origin_year", sort=True):
            annual_rows.append({
                "horizon_months": int(h), "origin_year": year, "n": int(len(gy)),
                "mean_paired_absolute_error_change_pp": float(gy.paired_absolute_error_change_pp.mean()),
                "median_change_pp": float(gy.paired_absolute_error_change_pp.median()),
                "improved_origins": int((gy.paired_absolute_error_change_pp < -1e-12).sum()),
                "worsened_origins": int((gy.paired_absolute_error_change_pp > 1e-12).sum()),
                "tied_origins": int(np.isclose(gy.paired_absolute_error_change_pp, 0.0, atol=1e-12).sum()),
            })
        for year in sorted(g.origin_year.unique()):
            remainder = g[g.origin_year.ne(year)]
            loyo_rows.append({
                "horizon_months": int(h), "omitted_origin_year": year,
                "remaining_n": int(len(remainder)),
                "mean_paired_absolute_error_change_pp": float(remainder.paired_absolute_error_change_pp.mean()),
            })
        changed = g[np.abs(g.post_origin_q50 - g.strict_q50) > 1e-12]
        forecast_change_rows.append({
            "horizon_months": int(h), "n_scored": int(len(g)), "n_forecasts_changed": int(len(changed)),
            "mean_absolute_q50_forecast_change_pp": float(np.abs(changed.post_origin_q50 - changed.strict_q50).mean()) if len(changed) else 0.0,
            "max_absolute_q50_forecast_change_pp": float(np.abs(changed.post_origin_q50 - changed.strict_q50).max()) if len(changed) else 0.0,
        })
        top = g.assign(abs_delta=np.abs(d)).sort_values("abs_delta", ascending=False).head(5)
        for rank, row in enumerate(top.itertuples(index=False), 1):
            concentration_rows.append({
                "horizon_months": int(h), "rank_abs_loss_change": rank, "origin_id": row.origin_id,
                "origin_month": row.origin_month, "paired_absolute_error_change_pp": row.paired_absolute_error_change_pp,
                "absolute_change_share": float(abs(row.paired_absolute_error_change_pp) / absd.sum()) if absd.sum() else 0.0,
                "recovery_applied_to_origin": bool(row.recovery_applied_to_origin),
            })

    # Keep forecast errors/predictions and derived paired changes; never write target values.
    out_files = {
        "paired_origin_losses.csv": paired,
        "mae_by_horizon.csv": pd.DataFrame(summary_rows),
        "paired_change_by_origin_year.csv": pd.DataFrame(annual_rows),
        "paired_change_leave_one_year_out.csv": pd.DataFrame(loyo_rows),
        "forecast_change_by_horizon.csv": pd.DataFrame(forecast_change_rows),
        "largest_origin_changes.csv": pd.DataFrame(concentration_rows),
    }
    for name, frame in out_files.items():
        frame.to_csv(args.output_dir / name, index=False, encoding="utf-8", lineterminator="\n")
    meta = {
        "status": "RETROSPECTIVE_DESCRIPTIVE_SCORE_COMPLETE",
        "cohort": "China CPI unified 2018-2026 rolling archive; B3 only",
        "forecast_archive_sha256": sha256(forecast_path),
        "target_panel_sha256": target_hash,
        "archived_b3_losses_sha256": sha256(args.archived_losses),
        "first_release_target_panel_matches": target_panel_matches,
        "target_fallback_months_from_previously_scored_signed_residual": sorted(set(target_panel_fallbacks)),
        "paired_origin_count_by_horizon": {str(h): int(n) for h, n in keys.groupby("horizon_months").size().items()},
        "strict_replay_max_abs_loss_difference_from_original_b3_pp": max_loss_rebuild,
        "archived_loss_rounding_tolerance_pp": 1e-7,
        "target_values_written_to_outputs": False,
        "p_values_or_bootstrap_intervals": "not calculated; post-score descriptive analysis only",
        "all_outcomes_and_prior_results_known_before_this_analysis": True,
        "not_independent_confirmation": True,
    }
    (args.output_dir / "scoring_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    hashes = {p.name: sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}
    (args.output_dir / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": meta, "mae_by_horizon": summary_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
