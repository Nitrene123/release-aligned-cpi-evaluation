#!/usr/bin/env python3
"""Post-score descriptive analysis of U.S. BEAH-vs-B3 origin losses.

This script reads only the derived origin_method_losses.csv. It does not load
the raw target file, forecast rows, or model weights, and it does not rerun
scoring. Negative BEAH-minus-B3 differences favor BEAH.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


HORIZONS = (1, 3, 6, 12)
YEARS = tuple(range(2018, 2025))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear quantile using index (n-1)*q (NumPy default method='linear')."""
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sample")
    position = (len(sorted_values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = position - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("empty paired sample")
    ordered = sorted(values)
    return {
        "n_origins": len(values),
        "mean_difference_pp": sum(values) / len(values),
        "median_difference_pp": quantile(ordered, 0.5),
        "q25_difference_pp": quantile(ordered, 0.25),
        "q75_difference_pp": quantile(ordered, 0.75),
        "minimum_difference_pp": ordered[0],
        "maximum_difference_pp": ordered[-1],
        "beah_lower_loss_origins": sum(v < 0 for v in values),
        "b3_lower_loss_origins": sum(v > 0 for v in values),
        "ties": sum(v == 0 for v in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", type=Path, required=True,
                        help="path to the already derived origin_method_losses.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")

    rows = read_rows(args.losses)
    selected: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        model = row["model"]
        if model not in {"B3 multivariate", "BEAH rolling"}:
            continue
        key = (row["target_month"], int(row["horizon_months"]))
        if key in selected.setdefault(key, {}):
            raise ValueError(f"duplicate model loss for {key}/{model}")
        selected[key][model] = float(row["origin_mean_absolute_error_pp"])

    expected = {(f"{year}-{month:02d}", h) for year in YEARS for month in range(1, 13) for h in HORIZONS}
    if set(selected) != expected:
        raise ValueError(f"paired key set differs from 84 targets x 4 horizons: {len(selected)}/{len(expected)}")
    if any(set(models) != {"B3 multivariate", "BEAH rolling"} for models in selected.values()):
        raise ValueError("B3 and BEAH do not have identical origin coverage")
    paired = {(month, h): values["BEAH rolling"] - values["B3 multivariate"]
              for (month, h), values in selected.items()}

    scopes = {
        "all_84_targets": set(),
        "exclude_2020": {2020},
        "exclude_2022": {2022},
        "exclude_2020_and_2022": {2020, 2022},
    }
    summary_rows = []
    yearly_rows = []
    for h in HORIZONS:
        for scope, excluded_years in scopes.items():
            values = [d for (month, horizon), d in paired.items()
                      if horizon == h and int(month[:4]) not in excluded_years]
            summary_rows.append({"comparison": "BEAH rolling - B3 multivariate",
                                 "horizon_months": h, "sample": scope,
                                 **summarize(values)})
        for year in YEARS:
            values = [d for (month, horizon), d in paired.items()
                      if horizon == h and int(month[:4]) == year]
            yearly_rows.append({"comparison": "BEAH rolling - B3 multivariate",
                                "horizon_months": h, "target_year": year,
                                **summarize(values)})

    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / "us_beah_b3_leave_year_out_2018_2024.csv"
    yearly_path = args.output_dir / "us_beah_b3_by_target_year_2018_2024.csv"
    summary_fields = list(summary_rows[0])
    yearly_fields = list(yearly_rows[0])
    for path, output_rows, fields in ((summary_path, summary_rows, summary_fields),
                                      (yearly_path, yearly_rows, yearly_fields)):
        with path.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    full_h3 = paired
    sum_all = sum(v for (month, h), v in full_h3.items() if h == 3)
    sum_2022 = sum(v for (month, h), v in full_h3.items() if h == 3 and month.startswith("2022-"))
    metadata = {
        "analysis_id": "US_BE​​AH_B3_POST_SCORE_DESCRIPTIVES_2018_2024_V1".replace("\u200b", ""),
        "status": "POST_SCORE_EXPLORATORY_DERIVED_LOSS_ANALYSIS",
        "input_file": args.losses.name,
        "input_sha256": sha256(args.losses),
        "input_rows_read": len(rows),
        "selected_methods": ["BEAH rolling", "B3 multivariate"],
        "paired_origin_horizon_keys": len(paired),
        "horizons_months": list(HORIZONS),
        "target_values_read": False,
        "forecast_rows_read": False,
        "scoring_rerun": False,
        "p_values_computed": False,
        "confidence_or_bootstrap_intervals_computed": False,
        "pairwise_difference_definition": "BEAH rolling origin-mean absolute error minus B3 multivariate origin-mean absolute error; negative favors BEAH",
        "quantile_method": "linear interpolation at position (n-1)*q",
        "year_exclusions": "Post-score descriptive sensitivity suggested by the reviewer; target years 2020 and 2022 analyzed separately and jointly",
        "h3_2022_share_of_net_difference_sum": sum_2022 / sum_all,
        "limitations": [
            "The leave-year-out and annual summaries are exploratory and were not prespecified.",
            "They do not establish an inflation-regime mechanism or statistical significance.",
            "Overlapping horizons and adjacent monthly origins remain dependent."
        ],
        "outputs": [summary_path.name, yearly_path.name],
    }
    meta_path = args.output_dir / "metadata.json"
    with meta_path.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
