from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
S11 = ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1"
BASE = ROOT / "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a/score_v2_clean_20261031/origin_method_losses.csv"
OUT = HERE / "combined_scores"
FILES = {
    "base": BASE,
    "paired": S11 / "derived_scores_v2/paired_origin_losses.csv",
    "timing_forecasts": S11 / "server_outputs_v1/forecast_predictions.csv",
    "vintage_forecasts": HERE / "vintage_forecasts/forecast_predictions.csv",
    "combined_forecasts": HERE / "combined_forecasts/forecast_predictions.csv",
    "combined_metadata": HERE / "combined_forecasts/inference_metadata.json",
    "combined_map": HERE / "combined_map/mask_recovery_map.csv",
    "vintage_scores": HERE / "vintage_scores/paired_losses.csv",
    "reference_metrics": HERE / "results/model_scenario_metrics.csv",
}
KEYS = ["origin_id", "origin_month", "target_month", "horizon_months"]
FKEYS = ["origin_id", "origin_month", "origin_timestamp_utc", "horizon_months"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = json.loads(FILES["combined_metadata"].read_text(encoding="utf-8"))
    if sha(FILES["combined_forecasts"]) != meta["forecast_predictions_sha256"] or not meta["strict_replay_within_tolerance"]:
        raise ValueError("Combined forecast seal or strict replay failed")
    if meta["P1_accessed"] or meta["scored_target_outcomes_read"] or meta["modified_cells"] != 118:
        raise ValueError("Combined inference scope mismatch")
    if meta["input_file_sha256"]["mask_recovery_map.csv"] != sha(FILES["combined_map"]):
        raise ValueError("Combined map hash mismatch")
    forecasts = {}
    for label, filename, variant in [("timing", "timing_forecasts", "post_origin_backfill"),
                                     ("vintage", "vintage_forecasts", "latest_archived_brent"),
                                     ("combined", "combined_forecasts", "partial_combined")]:
        f = pd.read_csv(FILES[filename], dtype={"origin_month": str})
        if f.duplicated(["variant", "origin_id", "horizon_months"]).any():
            raise ValueError(f"Duplicate prediction keys: {label}")
        a = f[f.variant.eq("strict_replay")][FKEYS + ["point_forecast_q50"]]
        b = f[f.variant.eq(variant)][FKEYS + ["point_forecast_q50"]]
        q = a.merge(b, on=FKEYS, how="outer", validate="one_to_one", suffixes=("_strict", "_changed"), indicator=True)
        if len(q) != 1884 or not q._merge.eq("both").all():
            raise ValueError(f"Forecast key mismatch: {label}")
        q["origin_timestamp_utc"] = pd.to_datetime(q.origin_timestamp_utc, utc=True)
        q[f"{label}_forecast_change_pp"] = q.point_forecast_q50_changed - q.point_forecast_q50_strict
        forecasts[label] = q[FKEYS + ["point_forecast_q50_strict", f"{label}_forecast_change_pp"]].rename(columns={"point_forecast_q50_strict": f"{label}_strict_q50"})
    base = pd.read_csv(FILES["base"], dtype={"origin_month": str, "target_month": str})
    base = base[base.model.eq("B3 multivariate")].copy()
    pair = pd.read_csv(FILES["paired"], dtype={"origin_month": str, "target_month": str})
    data = pair[KEYS + ["strict_q50", "strict_absolute_error_pp", "post_origin_absolute_error_pp"]].merge(base, on=KEYS, validate="one_to_one")
    data["origin_timestamp_utc"] = pd.to_datetime(data.origin_timestamp_utc, utc=True)
    for label in ("timing", "vintage", "combined"):
        data = data.merge(forecasts[label], on=FKEYS, validate="one_to_one")
        if np.max(np.abs(data[f"{label}_strict_q50"] - data.strict_q50)) > 1e-6:
            raise ValueError(f"Strict q50 parity failed: {label}")
    if len(data) != 396 or not data.groupby("horizon_months").size().eq(99).all():
        raise ValueError("Common-key coverage mismatch")
    signed = data.seed_mean_prediction_minus_target_pp.to_numpy(float)
    if np.max(np.abs(np.abs(signed) - data.strict_absolute_error_pp.to_numpy(float))) > 1e-7:
        raise ValueError("Archived strict loss mismatch")
    data["strict_loss_pp"] = np.abs(signed)
    for label in ("timing", "vintage", "combined"):
        data[f"{label}_loss_pp"] = np.abs(signed + data[f"{label}_forecast_change_pp"].to_numpy(float))
    if np.max(np.abs(data.timing_loss_pp - data.post_origin_absolute_error_pp)) > 1e-7:
        raise ValueError("Timing loss does not match the S11 score")
    vint = pd.read_csv(FILES["vintage_scores"])
    check = data.merge(vint[KEYS + ["vintage_loss_pp"]], on=KEYS, validate="one_to_one", suffixes=("", "_prior"))
    if np.max(np.abs(check.vintage_loss_pp - check.vintage_loss_pp_prior)) > 1e-9:
        raise ValueError("Vintage loss does not match the previous score")
    ref = pd.read_csv(FILES["reference_metrics"])
    ref = ref[ref.scenario.eq("strict")]
    contrasts = {"timing_minus_strict": ("timing", "strict"), "vintage_minus_strict": ("vintage", "strict"),
                 "combined_minus_strict": ("combined", "strict"), "combined_minus_timing": ("combined", "timing"),
                 "combined_minus_vintage": ("combined", "vintage")}
    for name, (a, b) in contrasts.items():
        data[name] = data[f"{a}_loss_pp"] - data[f"{b}_loss_pp"]
    data["interaction"] = data.combined_loss_pp - data.timing_loss_pp - data.vintage_loss_pp + data.strict_loss_pp
    summaries, intervals = [], []
    rng = np.random.default_rng(417)
    for h in (1, 3, 6, 12):
        q = data[data.horizon_months.eq(h)].sort_values("target_month")
        other = ref[ref.horizon_months.eq(h) & ~ref.model.eq("B3 multivariate")]
        if len(q) != 99 or len(other) != 12 or q.target_month.nunique() != 99:
            raise ValueError("Invalid matched sample")
        for scenario in ("strict", "timing", "vintage", "combined"):
            loss = q[f"{scenario}_loss_pp"].to_numpy(float)
            mae = float(np.mean(loss))
            summaries.append({"horizon_months": h, "scenario": scenario, "n": 99, "mae_pp": mae,
                              "rmse_pp": float(np.sqrt(np.mean(loss**2))), "rank_among_13": 1 + int((other.mae_pp < mae).sum())})
        calendar = pd.period_range(q.target_month.min(), q.target_month.max(), freq="M")
        for name in [*contrasts, "interaction"]:
            delta = q[name]
            values = pd.Series(delta.to_numpy(float), index=pd.PeriodIndex(q.target_month, freq="M")).reindex(calendar).to_numpy(float)
            for block in (3, 6, 12):
                starts = rng.integers(0, len(values), size=(30000, int(np.ceil(len(values) / block))))
                idx = (starts[..., None] + np.arange(block)) % len(values)
                draws = values[idx.reshape(30000, -1)[:, :len(values)]]
                counts = np.isfinite(draws).sum(axis=1)
                if (counts == 0).any():
                    raise ValueError("Empty bootstrap draw")
                low, high = np.quantile(np.nansum(draws, axis=1) / counts, [0.025, 0.975])
                intervals.append({"horizon_months": h, "contrast": name, "block_months": block, "n": 99,
                                  "mean_difference_pp": float(delta.mean()), "interval_low_pp": float(low),
                                  "interval_high_pp": float(high)})
    OUT.mkdir()
    columns = KEYS + ["strict_loss_pp", "timing_loss_pp", "vintage_loss_pp", "combined_loss_pp"] + list(contrasts) + ["interaction"]
    data[columns].to_csv(OUT / "paired_origin_losses.csv", index=False, float_format="%.12g", lineterminator="\n")
    pd.DataFrame(summaries).to_csv(OUT / "scenario_summary.csv", index=False, float_format="%.12g", lineterminator="\n")
    pd.DataFrame(intervals).to_csv(OUT / "paired_block_intervals.csv", index=False, float_format="%.12g", lineterminator="\n")
    manifest = {"status": "PARTIAL_TWO_FACTOR_DIAGNOSTIC_SCORED", "target_values_read": False, "target_values_exported": False,
                "p1_accessed": False, "common_origins_per_horizon": 99, "source_sha256": {k: sha(v) for k, v in FILES.items()},
                "protocol_sha256": sha(HERE / "PROTOCOL_COMBINED_PARTIAL.md"), "script_sha256": sha(Path(__file__)),
                "outputs_sha256": {n: sha(OUT / n) for n in ["paired_origin_losses.csv", "scenario_summary.csv", "paired_block_intervals.csv"]}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
