#!/usr/bin/env python3
"""Score the sealed CN 2018-01..2026-09 unified rolling forecasts.

This is a retrospective expanding-window score pass. It never edits the sealed
forecast directory and does not write target values to score outputs.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, shutil, tempfile
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

HORIZONS = (1, 3, 6, 12)
START = "2018-01"
END = "2026-09"
METHODS = (
    "No-change", "Seasonal-naive", "AR(1)", "SARIMA",
    "TimesFM-3 target-only", "B3 multivariate", "BEAH rolling",
    "Direct rolling", "Fixed candidate", "Fixed candidate + gate",
)
SEED_METHODS = {"BEAH rolling", "Direct rolling", "Fixed candidate + gate"}
EXPECTED_SEEDS = {"417", "42", "123", "2027", "3407"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def month_ord(m: str) -> int:
    y, mo = map(int, m.split("-"))
    if not 1 <= mo <= 12:
        raise ValueError(f"bad month {m}")
    return y * 12 + mo - 1


def parse_utc(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware UTC: {s}")
    return d.astimezone(ZoneInfo("UTC"))


def parse_cn_release(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return d.astimezone(ZoneInfo("UTC"))


def canonical_target(v: Decimal) -> str:
    return str(v.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def fnum(x, label):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"invalid number {label}: {x!r}")
    if not math.isfinite(v):
        raise ValueError(f"non-finite number {label}: {x!r}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-dir", required=True, type=Path)
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--target-extension", required=True, type=Path,
                    help="Official first-release targets for 2025-01..2026-09")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="New, separate output directory; must not already exist")
    args = ap.parse_args()
    run_dir, source_dir, ext_path, out_dir = map(Path, (args.forecast_dir, args.source_dir,
                                                        args.target_extension, args.out_dir))
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite score output: {out_dir}")
    manifest_path = run_dir / "sealed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != "CN_UNIFIED_EXPANDING_WINDOW_2018_2026_V1":
        raise ValueError("unexpected sealed forecast protocol")
    # Recheck every member of the sealed forecast manifest before reading predictions.
    checked, mismatches = 0, []
    for rel, expected_hash in manifest.get("files", {}).items():
        p = run_dir / rel
        checked += 1
        if not p.is_file():
            mismatches.append((rel, "MISSING"))
        elif sha256(p) != expected_hash:
            mismatches.append((rel, "SHA256_MISMATCH"))
    if mismatches:
        raise ValueError(f"sealed manifest integrity failure: {mismatches[:5]}")
    if not (run_dir / "model_forecasts_unscored.csv").is_file():
        raise FileNotFoundError("sealed forecast CSV is missing")

    # Construct the scoring target map. Historical target values are from the
    # unified first-release CPI panel; 2026 values are sourced from official NBS
    # release pages and replace float32-derived panel copies.
    panel = read_csv(source_dir / "unified_cpi_panel.csv")
    panel_by_month = {r["reference_month"]: r for r in panel}
    if len(panel_by_month) != len(panel):
        raise ValueError("duplicate reference month in unified CPI panel")
    ext_rows = read_csv(ext_path)
    ext_by_month = {r["reference_month"]: r for r in ext_rows}
    wanted_extension = {f"{y}-{m:02d}" for y, first, last in ((2025, 1, 12), (2026, 1, 9))
                        for m in range(first, last + 1)}
    if set(ext_by_month) != wanted_extension:
        raise ValueError(f"official target extension must be exactly 2025-01..2026-09: {sorted(ext_by_month)}")
    targets = {}
    target_meta = {}
    for year in range(2018, 2025):
        for mo in range(1, 13):
            m = f"{year}-{mo:02d}"
            row = panel_by_month.get(m)
            if not row or not row.get("first_release_yoy_pct", "").strip():
                raise ValueError(f"missing numeric first-release target in panel: {m}")
            v = Decimal(row["first_release_yoy_pct"])
            v = v.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            release = parse_cn_release(row["available_at_conservative_local"])
            url = row.get("value_source_url") or row.get("release_url") or row.get("index_url") or ""
            if not url:
                raise ValueError(f"target source URL missing for {m}")
            targets[m] = float(v)
            target_meta[m] = {"month":m, "value":float(v), "release_utc":release,
                              "source_url":url, "source_vintage":row.get("source_vintage", ""),
                              "value_sha256":hashlib.sha256(canonical_target(v).encode()).hexdigest(),
                              "canonical_value":canonical_target(v)}
    for m, row in ext_by_month.items():
        v = Decimal(row["first_release_yoy_pct"]).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        release = parse_utc(row["available_at_utc"])
        url = row["source_url"].strip()
        if not url.startswith("https://www.stats.gov.cn/"):
            raise ValueError(f"non-official target source URL for {m}: {url}")
        targets[m] = float(v)
        target_meta[m] = {"month":m, "value":float(v), "release_utc":release,
                          "source_url":url, "source_vintage":"NBS first-release release page",
                          "value_sha256":hashlib.sha256(canonical_target(v).encode()).hexdigest(),
                          "canonical_value":canonical_target(v)}
    expected_months = []
    for i in range(month_ord(START), month_ord(END) + 1):
        y, mo0 = divmod(i, 12)
        expected_months.append(f"{y:04d}-{mo0+1:02d}")
    if len(expected_months) != 105 or set(targets).intersection(expected_months) != set(expected_months):
        raise ValueError("target window is incomplete or not 105 monthly outcomes")

    planned = read_csv(run_dir / "planned_origin_horizon_keys.csv")
    planned_map = {}
    for r in planned:
        key = (r["origin_id"], int(r["horizon_months"]))
        if key in planned_map:
            raise ValueError(f"duplicate planned origin-horizon key: {key}")
        planned_map[key] = r
        if month_ord(r["target_month"]) != month_ord(r["origin_month"]) + key[1]:
            raise ValueError(f"bad origin-target horizon mapping: {key}")
    # Exactly one origin key for each target month and horizon.
    planned_target_keys = {(r["target_month"], int(r["horizon_months"])) for r in planned}
    expected_target_keys = {(m, h) for m in expected_months for h in HORIZONS}
    if len(planned) != 420 or planned_target_keys != expected_target_keys:
        raise ValueError("planned origin/horizon keys do not match 105 targets × four horizons")

    forecast_path = run_dir / "model_forecasts_unscored.csv"
    forecasts = read_csv(forecast_path)
    seen = set()
    pred_index = defaultdict(dict)
    forecast_meta = {}
    for r in forecasts:
        model, seed, origin, h = r["model"], r["seed"], r["origin_id"], int(r["horizon_months"])
        if model not in METHODS or h not in HORIZONS:
            raise ValueError(f"unexpected method/horizon: {model}/{h}")
        k = (model, seed, origin, h)
        if k in seen:
            raise ValueError(f"duplicate forecast row: {k}")
        seen.add(k)
        if r["status"] != "FORECAST_COMPLETE":
            raise ValueError(f"incomplete forecast row {k}: {r['status']}")
        pkey = (origin, h)
        if pkey not in planned_map:
            raise ValueError(f"forecast not in planned key set: {k}")
        p = planned_map[pkey]
        if r["origin_month"] != p["origin_month"] or r["target_month"] != p["target_month"]:
            raise ValueError(f"forecast mapping mismatch: {k}")
        if r["origin_timestamp_utc"] != p["origin_timestamp_utc"]:
            raise ValueError(f"origin timestamp mismatch: {k}")
        value = fnum(r["prediction"], k)
        if r["target_month"] not in targets:
            raise ValueError(f"target not available in score target map: {r['target_month']}")
        if target_meta[r["target_month"]]["release_utc"] <= parse_utc(r["origin_timestamp_utc"]):
            raise ValueError(f"target was not released after historical forecast origin: {k}")
        pred_index[(model, h)][(origin, seed)] = (value, r)
        forecast_meta[k] = r

    # Verify the 10 method coverage/seed structure and the archived pre-score
    # ledger. That ledger records one representative horizon (h=12); coverage
    # is checked here against the actual forecast key set at every horizon.
    pre_score_rows = read_csv(run_dir / "pre_score_model_coverage.csv")
    for model in METHODS:
        coverage_rows = [x for x in pre_score_rows
                         if x["model"] == model and int(x["horizon_months"]) == 12]
        if len(coverage_rows) != 1:
            raise ValueError(f"missing/duplicate representative pre-score coverage row {model}")
        reported = int(coverage_rows[0]["native_origin_count"])
        reported_seeds = int(coverage_rows[0]["seed_count"])
        expected_seeds = len(EXPECTED_SEEDS) if model in SEED_METHODS else 1
        if reported_seeds != expected_seeds:
            raise ValueError(f"pre-score seed count mismatch for {model}")
        for h in HORIZONS:
            records = pred_index[(model, h)]
            by_origin = defaultdict(set)
            for (origin, seed), _ in records.items():
                by_origin[origin].add(seed)
            expected_seeds = EXPECTED_SEEDS if model in SEED_METHODS else {""}
            if any(s != expected_seeds for s in by_origin.values()):
                raise ValueError(f"wrong seed set for {model}, h={h}")
            if len(by_origin) != reported:
                raise ValueError(f"pre-score coverage mismatch for {model}, h={h}: {len(by_origin)} vs {reported}")
            if set(by_origin) - {k[0] for k in planned_map if k[1] == h}:
                raise ValueError(f"unplanned origins in {model}, h={h}")

    # Build target-derived origin losses. Seeded heads use mean(seed-specific
    # absolute errors) within origin; seeds are never counted as time samples.
    losses_by_model_h = defaultdict(dict)
    origin_loss_rows = []
    for model in METHODS:
        for h in HORIZONS:
            for origin in sorted({o for (o, _s) in pred_index[(model, h)]}):
                p = planned_map[(origin, h)]
                target = targets[p["target_month"]]
                seed_items = []
                for seed in sorted(EXPECTED_SEEDS) if model in SEED_METHODS else [""]:
                    pred, _row = pred_index[(model, h)][(origin, seed)]
                    err = pred - target
                    seed_items.append((seed, pred, err))
                mae_loss = sum(abs(e) for _, _, e in seed_items) / len(seed_items)
                avg_pred = sum(pr for _, pr, _ in seed_items) / len(seed_items)
                signed = avg_pred - target
                sq_loss = sum(e*e for _, _, e in seed_items) / len(seed_items)
                losses_by_model_h[(model, h)][origin] = mae_loss
                origin_loss_rows.append({
                    "model":model,"origin_id":origin,"origin_month":p["origin_month"],
                    "origin_timestamp_utc":p["origin_timestamp_utc"],"target_month":p["target_month"],
                    "horizon_months":h,"seed_count":len(seed_items),
                    "mean_seed_absolute_error_pp":f"{mae_loss:.12f}",
                    "mean_seed_squared_error_pp2":f"{sq_loss:.12f}",
                    "seed_mean_prediction_minus_target_pp":f"{signed:.12f}",
                    "target_value_sha256":target_meta[p["target_month"]]["value_sha256"]
                })

    # All-method common-origin cohort per horizon, plus native coverage.
    common_by_h = {}
    for h in HORIZONS:
        sets = [set(losses_by_model_h[(m,h)]) for m in METHODS]
        common = set.intersection(*sets)
        common_by_h[h] = common
    summary_rows = []
    for h in HORIZONS:
        b3_native = losses_by_model_h[("B3 multivariate",h)]
        common = common_by_h[h]
        b3_common_mae = sum(b3_native[o] for o in common) / len(common) if common else float("nan")
        for model in METHODS:
            native = losses_by_model_h[(model,h)]
            mae = sum(native.values()) / len(native)
            rmse = math.sqrt(sum(
                (sum((pred_index[(model,h)][(o,s)][0]-targets[planned_map[(o,h)]["target_month"]])**2
                     for s in (sorted(EXPECTED_SEEDS) if model in SEED_METHODS else [""])) /
                 (5 if model in SEED_METHODS else 1))
                for o in native)/len(native))
            bias = sum(
                sum(pred_index[(model,h)][(o,s)][0]-targets[planned_map[(o,h)]["target_month"]]
                    for s in (sorted(EXPECTED_SEEDS) if model in SEED_METHODS else [""])) /
                (5 if model in SEED_METHODS else 1)
                for o in native)/len(native)
            common_mae = sum(native[o] for o in common)/len(common) if common else float("nan")
            b3_comparison_origins = set(native) & set(b3_native)
            pair_n = len(b3_comparison_origins)
            method_vs_b3 = (sum(native[o]-b3_native[o] for o in b3_comparison_origins)/pair_n) if pair_n else float("nan")
            rel = ((method_vs_b3 / (sum(b3_native[o] for o in b3_comparison_origins)/pair_n))*100
                   if pair_n and sum(b3_native[o] for o in b3_comparison_origins) != 0 else float("nan"))
            summary_rows.append({
                "model":model,"horizon_months":h,"planned_origins":105,
                "native_origins":len(native),"native_coverage_pct":f"{100*len(native)/105:.3f}",
                "native_MAE_pp":f"{mae:.9f}","native_RMSE_pp":f"{rmse:.9f}",
                "native_mean_bias_pp":f"{bias:.9f}",
                "all10_common_origins":len(common),"all10_common_MAE_pp":f"{common_mae:.9f}",
                "all10_common_B3_MAE_pp":f"{b3_common_mae:.9f}",
                "all10_common_relative_change_vs_B3_pct":
                    f"{((common_mae/b3_common_mae-1)*100):.4f}" if b3_common_mae else "",
                "pairwise_common_with_B3_origins":pair_n,
                "pairwise_mean_loss_difference_vs_B3_pp":f"{method_vs_b3:.9f}",
                "pairwise_relative_MAE_change_vs_B3_pct":f"{rel:.4f}"
            })

    pair_rows=[]
    for h in HORIZONS:
        for i, a in enumerate(METHODS):
            for b in METHODS[i+1:]:
                la, lb = losses_by_model_h[(a,h)], losses_by_model_h[(b,h)]
                common = sorted(set(la)&set(lb))
                diffs = [la[o]-lb[o] for o in common]
                ties = sum(abs(d)<=1e-12 for d in diffs)
                wins_a = sum(d < -1e-12 for d in diffs)
                wins_b = sum(d > 1e-12 for d in diffs)
                pair_rows.append({
                    "horizon_months":h,"method_a":a,"method_b":b,"pairwise_common_origins":len(common),
                    "mean_absolute_loss_difference_a_minus_b_pp":f"{sum(diffs)/len(diffs):.9f}" if diffs else "",
                    "a_lower_loss_origins":wins_a,"ties":ties,"b_lower_loss_origins":wins_b,
                    "interpretation":"descriptive paired comparison; no inferential claim"
                })

    target_audit=[]
    for m in expected_months:
        t=target_meta[m]
        target_audit.append({
            "target_month":m,"first_release_timestamp_utc":t["release_utc"].isoformat(),
            "source_vintage":t["source_vintage"],"official_source_url":t["source_url"],
            "canonical_display_value_sha256":t["value_sha256"],
            "raw_target_value_included":False
        })

    # Write separately from the immutable forecast archive; no raw target levels.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".score_unified_tmp_", dir=out_dir.parent))
    try:
        write_csv(tmp/"origin_method_losses.csv", list(origin_loss_rows[0].keys()), origin_loss_rows)
        write_csv(tmp/"score_summary_by_method_horizon.csv", list(summary_rows[0].keys()), summary_rows)
        write_csv(tmp/"pairwise_method_comparisons.csv", list(pair_rows[0].keys()), pair_rows)
        write_csv(tmp/"target_access_audit.csv", list(target_audit[0].keys()), target_audit)
        run_meta={
            "status":"RETROSPECTIVE_SCORE_COMPLETE",
            "protocol_id":"CN_UNIFIED_EXPANDING_WINDOW_2018_2026_V1",
            "score_window":{"first_target_month":START,"last_target_month":END,"target_months":105,
                            "horizons_months":list(HORIZONS),"planned_target_origin_horizon_pairs":420},
            "method_count":len(METHODS),"methods":list(METHODS),
            "scoring_rule":"For seeded heads, absolute error is computed per seed, averaged within each origin-horizon, then averaged over origins; seeds are not independent time samples.",
            "coverage_rule":"Native MAE uses each method's successful eligible origins. Fixed-denominator coverage is n/105. Pairwise summaries use common origins. The all-10 cohort is reported separately.",
            "classification":"Retrospective expanding-window backtest; not blind, not prospective, not independent confirmation. All target releases were known at score time; historical information cutoffs were enforced in forecast construction.",
            "p1_accessed":False,"prospective_queue_accessed":False,
            "sealed_forecast_manifest_sha256":sha256(manifest_path),
            "sealed_forecast_csv_sha256":sha256(forecast_path),
            "source_cpi_panel_sha256":sha256(source_dir/"unified_cpi_panel.csv"),
            "score_target_extension_sha256":sha256(ext_path),
            "score_targets_accessed_after_forecasts_sealed":True,
            "score_targets_count":len(target_audit),
            "target_values_written_to_score_outputs":False,
            "raw_target_values_canonicalized_to_official_one_decimal_display_resolution":True,
            "run_scoring_authorization":"User requested all methods under a unified configuration from the start of the evaluation window through 2026; this score is retrospective and is not represented as confirmatory.",
            "sealed_manifest_file_count_verified":checked,
            "sealed_manifest_mismatches":0,
        }
        (tmp/"score_run_metadata.json").write_text(json.dumps(run_meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        out_hashes={p.name:sha256(p) for p in sorted(tmp.iterdir()) if p.is_file()}
        score_manifest={"status":"RETROSPECTIVE_SCORE_COMPLETE","files":out_hashes,
                        "manifest_self_hash_excluded":True,
                        "forecast_run_directory_unchanged":True,
                        "score_script_sha256":sha256(Path(__file__).resolve()),
                        "target_values_in_outputs":False}
        (tmp/"score_manifest.json").write_text(json.dumps(score_manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        os.rename(tmp,out_dir)
    except Exception:
        # Preserve the partial temporary directory for diagnosis; never touch forecasts.
        raise
    print(json.dumps({"status":"RETROSPECTIVE_SCORE_COMPLETE","out_dir":str(out_dir),
                      "output_files":sorted(p.name for p in out_dir.iterdir()),
                      "native_summary_rows":len(summary_rows),"pairwise_rows":len(pair_rows),
                      "origin_loss_rows":len(origin_loss_rows),"all10_common_n_by_h":{str(h):len(common_by_h[h]) for h in HORIZONS},
                      "forecast_manifest_files_verified":checked},indent=2))

if __name__ == "__main__":
    main()
