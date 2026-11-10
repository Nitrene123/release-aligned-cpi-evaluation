from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
AUDIT = HERE / "audit"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    paths = {
        "paired": ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/derived_scores_v2/paired_origin_losses.csv",
        "base": ROOT / "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a/score_v2_clean_20261031/origin_method_losses.csv",
        "ets_theta": ROOT / "revision_experiments_v2/ets_theta_rolling_20261109/output/CN/score/new_origin_losses.csv",
        "timestral": ROOT / "revision_experiments_v2/timestral_cpi_baseline_20261109/historical_scores/origin_losses.csv",
        "recovery": ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/inputs/mask_recovery_map.csv",
    }
    assert all(sha(paths[k]) == h for k, h in manifest["source_sha256"].items())
    assert sha(HERE / "PROTOCOL.md") == manifest["protocol_sha256"]
    assert sha(HERE / "analyze_archived_diagnostic.py") == manifest["script_sha256"]
    assert all(sha(OUT / k) == h for k, h in manifest["outputs_sha256"].items())
    raw = pd.read_csv(paths["paired"])
    changes = pd.read_csv(OUT / "paired_origin_changes.csv")
    metrics = pd.read_csv(OUT / "model_scenario_metrics.csv")
    intervals = pd.read_csv(OUT / "paired_block_intervals.csv")
    keys = ["origin_id", "origin_month", "target_month", "horizon_months"]
    joined = changes.merge(raw, on=keys, validate="one_to_one", suffixes=("_out", "_source"))
    assert len(joined) == 396
    assert np.allclose(joined.loss_change_pp, joined.paired_absolute_error_change_pp, atol=1e-11, rtol=0)
    rng = np.random.default_rng(417)
    max_metric_error = 0.0
    max_interval_error = 0.0
    for h in (1, 3, 6, 12):
        q = joined[joined.horizon_months.eq(h)].sort_values("target_month")
        assert len(q) == 99 and q.target_month.nunique() == 99
        for scenario, col in [("strict", "strict_absolute_error_pp_source"), ("later_value_partial", "post_origin_absolute_error_pp")]:
            row = metrics[(metrics.scenario == scenario) & metrics.horizon_months.eq(h) & metrics.model.eq("B3 multivariate")].iloc[0]
            vals = q[col].to_numpy(float)
            max_metric_error = max(max_metric_error, abs(row.mae_pp - float(vals.mean())), abs(row.rmse_pp - math.sqrt(float(np.mean(vals ** 2)))))
            group = metrics[(metrics.scenario == scenario) & metrics.horizon_months.eq(h)]
            assert len(group) == 13 and int(row.rank_by_mae) == 1 + int((group.mae_pp < row.mae_pp).sum())
        ix = pd.period_range(q.target_month.min(), q.target_month.max(), freq="M")
        x = pd.Series(q.paired_absolute_error_change_pp.to_numpy(float), index=pd.PeriodIndex(q.target_month, freq="M")).reindex(ix).to_numpy(float)
        for block in (3, 6, 12):
            starts = rng.integers(len(x), size=(30000, math.ceil(len(x) / block)))
            picks = (starts[:, :, None] + np.arange(block)) % len(x)
            resampled = x[picks.reshape(30000, -1)[:, :len(x)]]
            means = np.nansum(resampled, axis=1) / np.isfinite(resampled).sum(axis=1)
            lo, hi = np.quantile(means, [.025, .975])
            row = intervals[(intervals.horizon_months == h) & (intervals.block_months == block)].iloc[0]
            max_interval_error = max(max_interval_error, abs(float(row.interval_low_pp)-lo), abs(float(row.interval_high_pp)-hi))
    # Strict replay is compared against the original score archive with a 1e-7 pp serialization tolerance.
    assert max_metric_error < 1e-7 and max_interval_error < 1e-10
    AUDIT.mkdir()
    report = {"status": "PASS", "paired_keys": 396, "model_scenario_cells": len(metrics),
              "intervals_recomputed": len(intervals), "max_metric_difference_pp": max_metric_error,
              "max_interval_difference_pp": max_interval_error, "input_and_output_hashes": "PASS",
              "verifier_sha256": sha(Path(__file__))}
    (AUDIT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
