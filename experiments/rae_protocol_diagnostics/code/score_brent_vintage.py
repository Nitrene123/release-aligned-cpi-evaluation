from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "vintage_scores"
BASE = ROOT / "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a/score_v2_clean_20261031/origin_method_losses.csv"
PAIRED = ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/derived_scores_v2/paired_origin_losses.csv"
FORECAST = HERE / "vintage_forecasts/forecast_predictions.csv"
META = HERE / "vintage_forecasts/inference_metadata.json"
MAP = HERE / "vintage_map/brent_vintage_replacement_map.csv"
METRICS = HERE / "results/model_scenario_metrics.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = json.loads(META.read_text(encoding="utf-8"))
    if sha(FORECAST) != meta["forecast_predictions_sha256"] or not meta["strict_replay_within_tolerance"]:
        raise ValueError("Forecast seal or strict replay check failed")
    if meta["P1_accessed"] or meta["scored_target_outcomes_read"]:
        raise ValueError("Unexpected inference input scope")
    f = pd.read_csv(FORECAST, dtype={"origin_month": str})
    if f.duplicated(["variant", "origin_id", "horizon_months"]).any():
        raise ValueError("Duplicate prediction key")
    strict = f[f.variant.eq("strict_replay")].copy()
    revised = f[f.variant.eq("latest_archived_brent")].copy()
    fkeys = ["origin_id", "origin_month", "origin_timestamp_utc", "horizon_months"]
    x = strict[fkeys + ["point_forecast_q50"]].merge(revised[fkeys + ["point_forecast_q50"]], on=fkeys, how="outer", validate="one_to_one", suffixes=("_strict", "_revised"), indicator=True)
    if not x._merge.eq("both").all() or len(x) != 1884:
        raise ValueError("Forecast variant key mismatch")
    x["forecast_change_pp"] = x.point_forecast_q50_revised - x.point_forecast_q50_strict
    affected = set(pd.read_csv(MAP).origin_id.astype(str))
    if (x.loc[~x.origin_id.isin(affected), "forecast_change_pp"].abs() > 1e-6).any():
        raise ValueError("Forecast changed outside mapped origin set")
    base = pd.read_csv(BASE, dtype={"origin_month": str, "target_month": str})
    base = base[base.model.eq("B3 multivariate")].copy()
    paired = pd.read_csv(PAIRED, dtype={"origin_month": str, "target_month": str})
    keys = ["origin_id", "origin_month", "target_month", "horizon_months"]
    if paired.duplicated(keys).any() or base.duplicated(keys).any():
        raise ValueError("Duplicate scoring key")
    b = paired[keys + ["strict_q50", "strict_absolute_error_pp"]].merge(base, on=keys, validate="one_to_one")
    x["origin_timestamp_utc"] = pd.to_datetime(x.origin_timestamp_utc, utc=True)
    b["origin_timestamp_utc"] = pd.to_datetime(b.origin_timestamp_utc, utc=True)
    b = b.merge(x, on=["origin_id", "origin_month", "origin_timestamp_utc", "horizon_months"], validate="one_to_one")
    if len(b) != 396 or not b.groupby("horizon_months").size().eq(99).all():
        raise ValueError("Scored key set differs from the 99-origin design")
    if np.max(np.abs(b.strict_q50 - b.point_forecast_q50_strict)) > 1e-6:
        raise ValueError("Strict forecast parity failed")
    signed = b.seed_mean_prediction_minus_target_pp.to_numpy(float)
    if np.max(np.abs(np.abs(signed) - b.mean_seed_absolute_error_pp.to_numpy(float))) > 1e-7:
        raise ValueError("Signed and absolute archived losses disagree")
    if np.max(np.abs(np.abs(signed) - b.strict_absolute_error_pp.to_numpy(float))) > 1e-7:
        raise ValueError("S11 loss and archived signed residual disagree")
    b["strict_loss_pp"] = np.abs(signed)
    b["vintage_loss_pp"] = np.abs(signed + b.forecast_change_pp.to_numpy(float))
    b["loss_change_pp"] = b.vintage_loss_pp - b.strict_loss_pp
    ref = pd.read_csv(METRICS)
    ref = ref[ref.scenario.eq("strict")].copy()
    rng = np.random.default_rng(417)
    summaries, intervals = [], []
    for h in (1, 3, 6, 12):
        q = b[b.horizon_months.eq(h)].sort_values("target_month")
        other = ref[(ref.horizon_months.eq(h)) & ~ref.model.eq("B3 multivariate")]
        if len(q) != 99 or len(other) != 12 or q.target_month.nunique() != 99:
            raise ValueError("Invalid common sample or comparison method count")
        strict_mae = float(q.strict_loss_pp.mean())
        revised_mae = float(q.vintage_loss_pp.mean())
        strict_rmse = float(np.sqrt(np.mean(q.strict_loss_pp.to_numpy() ** 2)))
        revised_rmse = float(np.sqrt(np.mean(q.vintage_loss_pp.to_numpy() ** 2)))
        delta = q.loss_change_pp
        summaries.append({"horizon_months": h, "n": len(q), "strict_mae_pp": strict_mae,
                          "vintage_mae_pp": revised_mae, "strict_rmse_pp": strict_rmse,
                          "vintage_rmse_pp": revised_rmse, "mean_loss_change_pp": float(delta.mean()),
                          "median_loss_change_pp": float(delta.median()),
                          "strict_rank": 1 + int((other.mae_pp < strict_mae).sum()),
                          "vintage_rank": 1 + int((other.mae_pp < revised_mae).sum()),
                          "improved": int((delta < -1e-12).sum()), "worsened": int((delta > 1e-12).sum()),
                          "ties": int((delta.abs() <= 1e-12).sum()),
                          "changed_forecasts": int((q.forecast_change_pp.abs() > 1e-12).sum())})
        calendar = pd.period_range(q.target_month.min(), q.target_month.max(), freq="M")
        series = pd.Series(delta.to_numpy(float), index=pd.PeriodIndex(q.target_month, freq="M")).reindex(calendar).to_numpy(float)
        for block in (3, 6, 12):
            starts = rng.integers(0, len(series), size=(30000, int(np.ceil(len(series) / block))))
            idx = (starts[..., None] + np.arange(block)) % len(series)
            draws = series[idx.reshape(30000, -1)[:, :len(series)]]
            counts = np.isfinite(draws).sum(axis=1)
            if (counts == 0).any():
                raise ValueError("Empty block-bootstrap draw")
            low, high = np.quantile(np.nansum(draws, axis=1) / counts, [0.025, 0.975])
            intervals.append({"horizon_months": h, "block_months": block, "n": 99,
                              "mean_difference_pp": float(delta.mean()), "interval_low_pp": float(low),
                              "interval_high_pp": float(high)})
    OUT.mkdir()
    b[keys + ["forecast_change_pp", "strict_loss_pp", "vintage_loss_pp", "loss_change_pp"]].to_csv(OUT / "paired_losses.csv", index=False, float_format="%.12g", lineterminator="\n")
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False, float_format="%.12g", lineterminator="\n")
    pd.DataFrame(intervals).to_csv(OUT / "paired_block_intervals.csv", index=False, float_format="%.12g", lineterminator="\n")
    sources = {"forecasts": FORECAST, "inference_metadata": META, "replacement_map": MAP, "baseline_losses": BASE, "s11_paired_losses": PAIRED, "reference_metrics": METRICS}
    result = {"status": "PARTIAL_BRENT_VINTAGE_DIAGNOSTIC_SCORED", "target_values_read": False,
              "target_values_exported": False, "p1_accessed": False, "n_common_per_horizon": 99,
              "source_sha256": {k: sha(v) for k, v in sources.items()},
              "scoring_protocol_sha256": sha(HERE / "SCORING_PARTIAL_VINTAGE.md"), "scoring_code_sha256": sha(Path(__file__)),
              "outputs_sha256": {n: sha(OUT / n) for n in ["paired_losses.csv", "summary.csv", "paired_block_intervals.csv"]}}
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
