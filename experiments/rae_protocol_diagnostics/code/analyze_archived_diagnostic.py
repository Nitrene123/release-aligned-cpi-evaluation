from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
INPUTS = {
    "paired": ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/derived_scores_v2/paired_origin_losses.csv",
    "base": ROOT / "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a/score_v2_clean_20261031/origin_method_losses.csv",
    "ets_theta": ROOT / "revision_experiments_v2/ets_theta_rolling_20261109/output/CN/score/new_origin_losses.csv",
    "timestral": ROOT / "revision_experiments_v2/timestral_cpi_baseline_20261109/historical_scores/origin_losses.csv",
    "recovery": ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/inputs/mask_recovery_map.csv",
}
KEYS = ["origin_id", "origin_month", "target_month", "horizon_months"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, dtype={"origin_id": str, "origin_month": str, "target_month": str})


def main():
    if OUT.exists():
        raise FileExistsError(f"Output already exists: {OUT}")
    base, extra, ts, paired, recovery = (read(INPUTS[k]) for k in ("base", "ets_theta", "timestral", "paired", "recovery"))
    base = base.rename(columns={"mean_seed_absolute_error_pp": "absolute", "mean_seed_squared_error_pp2": "squared"})
    base = base[["model", *KEYS, "origin_timestamp_utc", "absolute", "squared"]]
    extra = extra.rename(columns={"absolute_error_pp": "absolute"})
    ts = ts[ts.country.eq("CN")].rename(columns={"absolute_error_pp": "absolute"})
    for x in (extra, ts):
        x["squared"] = x.absolute.astype(float) ** 2
    all_losses = pd.concat([base, extra[base.columns], ts[base.columns]], ignore_index=True)
    if all_losses.duplicated(["model", *KEYS]).any() or paired.duplicated(KEYS).any():
        raise ValueError("Duplicate model or paired-loss keys")
    methods = sorted(all_losses.model.unique())
    if len(methods) != 13:
        raise ValueError(f"Expected 13 methods, found {methods}")
    recovery_counts = recovery.groupby("feature_name").size().to_dict()
    if len(recovery) != 104:
        raise ValueError(f"Unexpected recovered-cell count: {len(recovery)}")
    output_rows, rank_rows, interval_rows, paired_rows, coverage = [], [], [], [], []
    rng = np.random.default_rng(417)
    maximum_strict_difference = 0.0
    for h in (1, 3, 6, 12):
        part = all_losses[all_losses.horizon_months.eq(h)].copy()
        pr = paired[paired.horizon_months.eq(h)].copy()
        key_sets = [set(map(tuple, part.loc[part.model.eq(m), KEYS].to_numpy())) for m in methods]
        common = set.intersection(*key_sets, set(map(tuple, pr[KEYS].to_numpy())))
        if len(common) != 99:
            raise ValueError(f"Unexpected common count for h={h}: {len(common)}")
        keep = pd.MultiIndex.from_frame(part[KEYS]).isin(pd.MultiIndex.from_tuples(sorted(common), names=KEYS))
        part = part.loc[keep].copy()
        pr = pr[pd.MultiIndex.from_frame(pr[KEYS]).isin(pd.MultiIndex.from_tuples(sorted(common), names=KEYS))].copy()
        clock_counts = part.groupby(KEYS, dropna=False).origin_timestamp_utc.nunique()
        if not clock_counts.eq(1).all():
            raise ValueError(f"Origin timestamp mismatch for h={h}")
        b3 = part[part.model.eq("B3 multivariate")].merge(pr, on=KEYS, validate="one_to_one")
        maxdiff = float(np.max(np.abs(b3.absolute.astype(float) - b3.strict_absolute_error_pp.astype(float))))
        maximum_strict_difference = max(maximum_strict_difference, maxdiff)
        if maxdiff > 1e-7:
            raise ValueError(f"Strict B3 loss mismatch h={h}: {maxdiff}")
        if not np.allclose(pr.post_origin_absolute_error_pp - pr.strict_absolute_error_pp,
                           pr.paired_absolute_error_change_pp, atol=1e-12, rtol=0):
            raise ValueError("Paired loss change mismatch")
        for scenario in ("strict", "later_value_partial"):
            for m in methods:
                q = part[part.model.eq(m)]
                if scenario == "later_value_partial" and m == "B3 multivariate":
                    absolute = pr.post_origin_absolute_error_pp.to_numpy(float)
                    squared = absolute ** 2
                else:
                    absolute = q.absolute.to_numpy(float)
                    squared = q.squared.to_numpy(float)
                if len(absolute) != 99 or not (np.isfinite(absolute).all() and np.isfinite(squared).all()):
                    raise ValueError(f"Invalid loss data: {scenario}, {m}, {h}")
                output_rows.append({"scenario": scenario, "horizon_months": h, "model": m, "n": 99,
                                    "mae_pp": float(np.mean(absolute)), "rmse_pp": float(np.sqrt(np.mean(squared)))})
        pr = pr.sort_values("target_month")
        for _, row in pr.iterrows():
            paired_rows.append({k: row[k] for k in KEYS} | {
                "strict_absolute_error_pp": float(row.strict_absolute_error_pp),
                "later_value_absolute_error_pp": float(row.post_origin_absolute_error_pp),
                "loss_change_pp": float(row.paired_absolute_error_change_pp),
            })
        dates = pd.period_range(pr.target_month.min(), pr.target_month.max(), freq="M")
        series = pd.Series(pr.paired_absolute_error_change_pp.to_numpy(float), index=pd.PeriodIndex(pr.target_month, freq="M"))
        if not series.index.is_unique:
            raise ValueError("Duplicate target month within horizon")
        calendar = series.reindex(dates).to_numpy(float)
        for block in (3, 6, 12):
            starts = rng.integers(0, len(calendar), size=(30000, int(np.ceil(len(calendar) / block))))
            idx = (starts[..., None] + np.arange(block)) % len(calendar)
            sampled = calendar[idx.reshape(30000, -1)[:, :len(calendar)]]
            count = np.isfinite(sampled).sum(axis=1)
            if (count == 0).any():
                raise ValueError("Empty bootstrap draw")
            means = np.nansum(sampled, axis=1) / count
            lo, hi = np.quantile(means, [0.025, 0.975])
            interval_rows.append({"horizon_months": h, "block_months": block, "n": 99,
                                  "calendar_months": len(calendar), "mean_difference_pp": float(series.mean()),
                                  "median_difference_pp": float(series.median()), "interval_low_pp": float(lo),
                                  "interval_high_pp": float(hi), "improved": int((series < -1e-12).sum()),
                                  "worsened": int((series > 1e-12).sum()), "ties": int((series.abs() <= 1e-12).sum())})
        for m in methods:
            coverage.append({"horizon_months": h, "model": m, "native_available": int((all_losses.model.eq(m) & all_losses.horizon_months.eq(h)).sum()), "common": len(common)})
    results = pd.DataFrame(output_rows)
    results["rank_by_mae"] = results.groupby(["scenario", "horizon_months"]).mae_pp.rank(method="min").astype(int)
    for h in (1, 3, 6, 12):
        b = results[results.horizon_months.eq(h) & results.model.eq("B3 multivariate")].set_index("scenario")
        rank_rows.append({"horizon_months": h, "strict_mae_pp": b.loc["strict", "mae_pp"],
                          "partial_later_mae_pp": b.loc["later_value_partial", "mae_pp"],
                          "strict_rank": int(b.loc["strict", "rank_by_mae"]),
                          "partial_later_rank": int(b.loc["later_value_partial", "rank_by_mae"])})
    OUT.mkdir()
    tables = {"model_scenario_metrics.csv": results, "b3_rank_changes.csv": pd.DataFrame(rank_rows),
              "paired_origin_changes.csv": pd.DataFrame(paired_rows), "paired_block_intervals.csv": pd.DataFrame(interval_rows),
              "coverage.csv": pd.DataFrame(coverage)}
    for name, frame in tables.items():
        frame.to_csv(OUT / name, index=False, lineterminator="\n", float_format="%.12g")
    manifest = {"status": "ARCHIVED_PARTIAL_INFORMATION_DIAGNOSTIC_COMPLETE", "n_methods": 13,
                "common_origins_per_horizon": 99, "max_strict_b3_loss_difference_pp": maximum_strict_difference,
                "recovered_cells_by_feature": recovery_counts, "full_P0_P3_experiment": False,
                "source_sha256": {k: sha(v) for k, v in INPUTS.items()},
                "protocol_sha256": sha(HERE / "PROTOCOL.md"), "script_sha256": sha(Path(__file__)),
                "outputs_sha256": {name: sha(OUT / name) for name in tables}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "max_strict_loss_difference_pp": maximum_strict_difference,
                      "recovery_by_feature": recovery_counts, "b3_ranks": rank_rows}, indent=2))


if __name__ == "__main__":
    main()
