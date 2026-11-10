#!/usr/bin/env python3
"""Exploratory paired block intervals using the frozen U.S. loss archive.

No target values, prediction arrays, model weights, or prospective queue are read.
The output directory must not already exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASELINE = "B3 multivariate"
METHODS = (
    "No-change", "Seasonal-naive", "AR(1)", "SARIMA", "TimesFM-3 target-only",
    BASELINE, "BEAH rolling", "Direct rolling", "Fixed candidate", "Fixed candidate + gate",
)
SEEDED = {"BEAH rolling", "Direct rolling", "Fixed candidate + gate"}
HORIZONS = (1, 3, 6, 12)
BLOCKS = (3, 6, 12)
DRAWS = 30_000
SEED = 417
TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def add_month(month: str, n: int) -> str:
    year, mon = map(int, month.split("-"))
    if not 1 <= mon <= 12:
        raise ValueError(f"Invalid month: {month}")
    x = year * 12 + mon - 1 + n
    return f"{x // 12:04d}-{x % 12 + 1:02d}"


def draw_indices(n: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    ap.add_argument("--score-dir", type=Path, default=base / "us_cpi_rolling_score_v1/score")
    ap.add_argument("--plan", type=Path, default=base / "us_paired_uncertainty_analysis_plan.md")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.out_dir}")

    files = {
        "losses": args.score_dir / "origin_method_losses.csv",
        "common_mae": args.score_dir / "mae_by_method_horizon_common10.csv",
        "pairwise": args.score_dir / "pairwise_loss_contrasts.csv",
        "score_metadata": args.score_dir / "scoring_metadata.json",
        "analysis_plan": args.plan,
    }
    digests = {name: sha256(path) for name, path in files.items()}
    score_meta = json.loads(files["score_metadata"].read_text(encoding="utf8"))
    registered = {entry["name"]: entry["sha256"] for entry in score_meta["output_files"]}
    for name in ("losses", "common_mae", "pairwise"):
        filename = files[name].name
        if registered.get(filename) != digests[name]:
            raise ValueError(f"Input differs from original score metadata: {filename}")

    calendar = [add_month("2018-01", n) for n in range(84)]
    if calendar[-1] != "2024-12":
        raise AssertionError("Unexpected monthly calendar")
    losses = read_csv(files["losses"])
    if len(losses) != len(METHODS) * len(HORIZONS) * len(calendar):
        raise ValueError("Unexpected number of origin-level loss rows")
    keyed = {}
    for r in losses:
        model = r["model"]
        h = int(r["horizon_months"])
        target = r["target_month"]
        key = (model, h, target)
        if key in keyed or model not in METHODS or h not in HORIZONS or target not in calendar:
            raise ValueError(f"Invalid or duplicate model/horizon/target key: {key}")
        if add_month(r["origin_month"], h) != target:
            raise ValueError(f"Origin/target mismatch: {key}")
        if int(r["n_seed_predictions"]) != (5 if model in SEEDED else 1):
            raise ValueError(f"Incorrect seed aggregation count: {key}")
        loss = float(r["origin_mean_absolute_error_pp"])
        if not math.isfinite(loss) or loss < 0:
            raise ValueError(f"Invalid absolute loss: {key}")
        keyed[key] = r
    common_mae = {(r["model"], int(r["horizon_months"])): r for r in read_csv(files["common_mae"])}
    pairwise = read_csv(files["pairwise"])
    contrasts = [(m, BASELINE) for m in METHODS if m != BASELINE] + [("BEAH rolling", "Direct rolling")]
    rng = np.random.default_rng(SEED)
    out = []
    for h in HORIZONS:
        for target in calendar:
            anchor = keyed[(BASELINE, h, target)]
            for method in METHODS:
                r = keyed[(method, h, target)]
                if any(r[field] != anchor[field] for field in ("origin_id", "origin_month", "origin_timestamp_utc", "target_month")):
                    raise ValueError(f"Mismatch in matched origins for {(method, h, target)}")
        vectors = {}
        for a, b in contrasts:
            d = np.array([float(keyed[(a, h, t)]["origin_mean_absolute_error_pp"]) -
                          float(keyed[(b, h, t)]["origin_mean_absolute_error_pp"]) for t in calendar])
            vectors[(a, b)] = d
            for method in (a, b):
                measured = np.mean([float(keyed[(method, h, t)]["origin_mean_absolute_error_pp"]) for t in calendar])
                summary = common_mae[(method, h)]
                if int(summary["n_common_origins"]) != 84 or abs(measured - float(summary["mae_pp"])) > 1e-9:
                    raise ValueError(f"MAE does not match sealed summary: {(method, h)}")
            matches = [r for r in pairwise if int(r["horizon_months"]) == h and
                       ((r["method_a"], r["method_b"]) in {(a, b), (b, a)})]
            if len(matches) != 1 or int(matches[0]["n_paired_origins"]) != 84:
                raise ValueError(f"Pairwise summary is absent or ambiguous: {(a, b, h)}")
            row = matches[0]
            sign = 1 if row["method_a"] == a else -1
            if abs(float(np.mean(d)) - sign * float(row["mean_loss_difference_a_minus_b_pp"])) > 1e-9:
                raise ValueError(f"Pairwise mean does not match sealed summary: {(a, b, h)}")
            wins, ties, losses_b = int(np.sum(d < -TOL)), int(np.sum(np.abs(d) <= TOL)), int(np.sum(d > TOL))
            expected_wins = int(row["origins_a_lower_loss"] if sign == 1 else row["origins_b_lower_loss"])
            expected_losses = int(row["origins_b_lower_loss"] if sign == 1 else row["origins_a_lower_loss"])
            if (wins, losses_b, ties) != (expected_wins, expected_losses, int(row["ties_1e-12"])):
                raise ValueError(f"Pairwise direction counts do not match sealed summary: {(a, b, h)}")
        for block in BLOCKS:
            idx = draw_indices(len(calendar), block, DRAWS, rng)
            for (a, b), d in vectors.items():
                means = d[idx].mean(axis=1)
                # Verify vectorized indexing against an independent scalar calculation.
                assert all(abs(means[k] - sum(float(d[j]) for j in idx[k]) / len(calendar)) < 1e-12 for k in range(10))
                lo, hi = np.quantile(means, [0.025, 0.975], method="linear")
                out.append({
                    "method_a": a, "method_b": b, "horizon_months": h,
                    "paired_origins": len(calendar), "mean_loss_difference_pp": float(d.mean()),
                    "a_lower": int(np.sum(d < -TOL)), "b_lower": int(np.sum(d > TOL)),
                    "ties_1e-12": int(np.sum(np.abs(d) <= TOL)),
                    "block_length_months": block, "bootstrap_draws": DRAWS, "seed": SEED,
                    "percentile_2_5_pp": float(lo), "percentile_97_5_pp": float(hi),
                })
    if len(out) != 120:
        raise AssertionError("Unexpected number of contrast/block summaries")
    if {name: sha256(path) for name, path in files.items()} != digests:
        raise ValueError("An input changed during analysis")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.out_dir / "us_paired_block_intervals.csv"
    with result_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    meta = {
        "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "U.S. CPI-U first-release 2018-01 to 2024-12",
        "analysis_status": "exploratory, specified after original scoring",
        "input_sha256": digests, "analysis_script_sha256": sha256(Path(__file__)),
        "output_csv_sha256": sha256(result_path), "python": platform.python_version(), "numpy": np.__version__,
        "horizons": list(HORIZONS), "paired_origins_each_horizon": len(calendar),
        "contrasts": [list(x) for x in contrasts], "difference": "method_a loss minus method_b loss",
        "bootstrap": {"type": "circular moving block", "blocks_months": list(BLOCKS),
                      "draws": DRAWS, "seed": SEED, "percentiles": [2.5, 97.5],
                      "quantile_method": "linear", "same_draws_for_contrasts": True},
        "rows": len(out), "seed_rule": "five seed absolute losses averaged within origin by original scorer",
        "interpretation": "descriptive, pointwise, post-score; no multiplicity adjustment or p-values",
        "target_values_loaded": False, "p1_accessed": False, "forecast_models_fitted": False,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": meta["status"], "rows": len(out), "input_loss_sha256": digests["losses"],
                      "output_csv_sha256": meta["output_csv_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
