#!/usr/bin/env python3
"""Summarize paired origin-level PIT mask counterfactual loss changes.

The script reads the target-free paired_origin_losses.csv from the archived
PIT counterfactual tarball. It does not load target values, refit models, or
modify the source archive. Outputs are written to a separate directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import tarfile
from collections import defaultdict
from pathlib import Path


VARIANTS = {
    "timestamp_tie_inclusive": "timestamp_tie_inclusive_abs_error_pp",
    "post_origin_backfill": "post_release_leak_abs_error_pp",
    "combined": "combined_counterfactual_abs_error_pp",
}
STRICT = "strict_replay_abs_error_pp"
EPS = 1e-12


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def read_rows(archive: Path) -> list[dict[str, str]]:
    with tarfile.open(archive, "r:gz") as bundle:
        candidates = [m for m in bundle.getmembers()
                      if m.isfile() and m.name.endswith("paired_origin_losses.csv")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one paired_origin_losses.csv, found {len(candidates)}")
        stream = bundle.extractfile(candidates[0])
        if stream is None:
            raise ValueError("Could not read paired loss table")
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", newline="")))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    scored = [r for r in rows if r["valid_for_scoring"].strip().lower() == "true"]
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in scored:
        grouped[(row["country"], int(row["horizon_months"]))].append(row)

    output: list[dict[str, str]] = []
    for (country, horizon), strict_rows in sorted(grouped.items()):
        strict_by_origin = {r["origin_id"]: float(r[STRICT]) for r in strict_rows}
        if len(strict_by_origin) != len(strict_rows):
            raise ValueError(f"Duplicate scored origin: {country}, h={horizon}")
        for variant, column in VARIANTS.items():
            variant_rows = [r for r in strict_rows if r.get(column, "") != ""]
            if {r["origin_id"] for r in variant_rows} != set(strict_by_origin):
                raise ValueError(f"Origin-key mismatch: {country}, h={horizon}, {variant}")
            deltas = [(float(r[column]) - strict_by_origin[r["origin_id"]], r)
                      for r in variant_rows]
            if not all(math.isfinite(delta) for delta, _ in deltas):
                raise ValueError(f"Non-finite paired delta: {country}, h={horizon}, {variant}")
            years = sorted({r["origin_month"][:4] for _, r in deltas})
            subsets: list[tuple[str, str, list[tuple[float, dict[str, str]]]]] = [
                ("overall", "all_origins", deltas)
            ]
            subsets.extend(("origin_year", year,
                            [(d, r) for d, r in deltas if r["origin_month"][:4] == year])
                           for year in years)
            subsets.extend(("omit_origin_year", year,
                            [(d, r) for d, r in deltas if r["origin_month"][:4] != year])
                           for year in years)
            for subset_type, subset_label, subset in subsets:
                values = [delta for delta, _ in subset]
                if not values:
                    continue
                ranked = sorted(subset, key=lambda item: abs(item[0]), reverse=True)
                abs_total = sum(abs(delta) for delta in values)
                top = ranked[:3]
                output.append({
                    "country": country,
                    "variant": variant,
                    "horizon_months": str(horizon),
                    "subset_type": subset_type,
                    "subset_label": subset_label,
                    "n": str(len(values)),
                    "mean_delta_pp": f"{sum(values) / len(values):.9f}",
                    "median_delta_pp": f"{quantile(values, 0.5):.9f}",
                    "q1_delta_pp": f"{quantile(values, 0.25):.9f}",
                    "q3_delta_pp": f"{quantile(values, 0.75):.9f}",
                    "min_delta_pp": f"{min(values):.9f}",
                    "max_delta_pp": f"{max(values):.9f}",
                    "improved_origins": str(sum(d < -EPS for d in values)),
                    "worsened_origins": str(sum(d > EPS for d in values)),
                    "tied_origins": str(sum(abs(d) <= EPS for d in values)),
                    "top3_share_of_absolute_changes": f"{sum(abs(d) for d, _ in top) / abs_total:.9f}" if abs_total else "0.000000000",
                    "top3_origin_deltas": ";".join(
                        f"{r['origin_month']}={d:+.9f}" for d, r in top
                    ),
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="PIT_Mask_Counterfactual_20261101.tar.gz")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="New or existing output directory; source archive is never changed")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(read_rows(args.archive))
    out = args.out_dir / "S11_origin_delta_descriptives.csv"
    columns = list(rows[0]) if rows else []
    with out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} descriptive rows to {out}")


if __name__ == "__main__":
    main()
