#!/usr/bin/env python3
"""Post-score U.S. ETS/Theta versus B3 circular block intervals, target-free."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from us_paired_block_uncertainty import BLOCKS, DRAWS, HORIZONS, SEED, TOL, add_month, draw_indices, read_csv, sha256

METHODS = ("ETS(A,A,N)", "Theta")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    supp = Path(__file__).resolve().parents[1]
    ap.add_argument("--original-score-dir", type=Path, default=supp / "us_cpi_rolling_score_v1/score")
    ap.add_argument("--new-score-dir", type=Path, default=supp / "ets_theta_rolling/US/score")
    ap.add_argument("--plan", type=Path, default=supp / "us_paired_extra_baselines_analysis_plan.md")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.out_dir}")
    files = {
        "original_losses": args.original_score_dir / "origin_method_losses.csv",
        "original_metadata": args.original_score_dir / "scoring_metadata.json",
        "additional_losses": args.new_score_dir / "new_origin_losses.csv",
        "additional_pairwise": args.new_score_dir / "new_vs_existing_pairwise.csv",
        "additional_metadata": args.new_score_dir / "score_manifest.json",
        "analysis_plan": args.plan,
    }
    hashes = {k: sha256(v) for k, v in files.items()}
    old_meta = json.loads(files["original_metadata"].read_text(encoding="utf8"))
    old_hash = next(x["sha256"] for x in old_meta["output_files"] if x["name"] == "origin_method_losses.csv")
    new_meta = json.loads(files["additional_metadata"].read_text(encoding="utf8"))
    if old_hash != hashes["original_losses"] or new_meta["old_origin_losses_sha256"] != old_hash:
        raise ValueError("Original B3 losses do not match the two score manifests")
    for key in ("additional_losses", "additional_pairwise"):
        if new_meta["outputs"].get(files[key].name) != hashes[key]:
            raise ValueError(f"Additional score hash mismatch: {key}")
    originals = read_csv(files["original_losses"])
    extra = read_csv(files["additional_losses"])
    if len(extra) != 2 * 4 * 84 or any(r["country"] != "US" or r["model"] not in METHODS for r in extra):
        raise ValueError("Unexpected additional model loss inventory")
    baseline = {(int(r["horizon_months"]), r["target_month"]): r for r in originals if r["model"] == "B3 multivariate"}
    additions = {(r["model"], int(r["horizon_months"]), r["target_month"]): r for r in extra}
    if len(baseline) != 336 or len(additions) != 672:
        raise ValueError("Duplicate or missing matched loss keys")
    pairs = read_csv(files["additional_pairwise"])
    calendar = [add_month("2018-01", n) for n in range(84)]
    rng = np.random.default_rng(SEED)
    out = []
    for h in HORIZONS:
        vecs = {}
        for m in METHODS:
            diffs = []
            for target in calendar:
                new, old = additions[(m, h, target)], baseline[(h, target)]
                fields = ("origin_id", "origin_month", "origin_timestamp_utc", "target_month", "horizon_months")
                if any(new[k] != old[k] for k in fields):
                    raise ValueError(f"Origin/target mismatch: {(m, h, target)}")
                diffs.append(float(new["absolute_error_pp"]) - float(old["origin_mean_absolute_error_pp"]))
            d = np.array(diffs, dtype=float)
            if not np.isfinite(d).all():
                raise ValueError("Nonfinite paired difference")
            comparison = [x for x in pairs if x["model"] == m and x["reference"] == "B3 multivariate" and int(x["horizon_months"]) == h]
            if len(comparison) != 1 or int(comparison[0]["n_pairs"]) != 84 or abs(float(d.mean()) - float(comparison[0]["mean_loss_difference_pp"])) > 1e-9:
                raise ValueError(f"Pairwise mean mismatch: {(m, h)}")
            v = comparison[0]
            counts = (int(np.sum(d < -TOL)), int(np.sum(d > TOL)), int(np.sum(np.abs(d) <= TOL)))
            if counts != (int(v["model_lower_count"]), int(v["reference_lower_count"]), int(v["tie_count"])):
                raise ValueError(f"Pairwise counts mismatch: {(m, h)}")
            vecs[m] = d
        for block in BLOCKS:
            idx = draw_indices(84, block, DRAWS, rng)
            for m in METHODS:
                d = vecs[m]
                means = d[idx].mean(axis=1)
                lo, hi = np.quantile(means, [.025, .975], method="linear")
                out.append({"method_a": m, "method_b": "B3 multivariate", "horizon_months": h,
                            "paired_origins": 84, "mean_loss_difference_pp": float(d.mean()),
                            "a_lower": int(np.sum(d < -TOL)), "b_lower": int(np.sum(d > TOL)),
                            "ties_1e-12": int(np.sum(np.abs(d) <= TOL)), "block_length_months": block,
                            "bootstrap_draws": DRAWS, "seed": SEED,
                            "percentile_2_5_pp": float(lo), "percentile_97_5_pp": float(hi)})
    if len(out) != 24 or hashes != {k: sha256(v) for k, v in files.items()}:
        raise AssertionError("Unexpected output count or changed input")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    result = args.out_dir / "us_ets_theta_block_intervals.csv"
    with result.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    metadata = {
        "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "exploratory, specified after original and additional point scores",
        "input_sha256": hashes, "analysis_script_sha256": sha256(Path(__file__)),
        "shared_bootstrap_script_sha256": sha256(Path(__file__).with_name("us_paired_block_uncertainty.py")),
        "output_csv_sha256": sha256(result), "python": platform.python_version(), "numpy": np.__version__,
        "models": list(METHODS), "reference": "B3 multivariate", "horizons": list(HORIZONS),
        "paired_origins_per_horizon": 84, "blocks_months": list(BLOCKS), "draws": DRAWS, "seed": SEED,
        "interval": "2.5th/97.5th linear percentile; pointwise descriptive; no multiplicity adjustment",
        "rows": len(out), "target_values_loaded": False, "forecast_models_fitted": False, "p1_accessed": False,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PASS", "rows": len(out), "output_csv_sha256": metadata["output_csv_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
