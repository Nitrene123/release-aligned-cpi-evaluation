"""Calendar-aware descriptive block-bootstrap contrasts for the unified CPI run.

Reads only the target-free origin-level loss file. It does not load target values,
forecast arrays, the P1 queue, or any model checkpoint. Outputs are new derived
artifacts; the sealed score archive is never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from datetime import date
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUPP = ROOT / "supplementary"
ARCHIVE_SCORE_DIR = (
    ROOT
    / "reproducibility"
    / "matched_origin_2018_2026_unified_v1"
    / "results"
    / "nested_validation_clean_v1"
    / "score_v1_clean_20261031"
)
BUNDLED_SCORE_DIR = SUPP / "paired_uncertainty_inputs"
if (ARCHIVE_SCORE_DIR / "origin_method_losses.csv").is_file():
    SCORE_DIR = ARCHIVE_SCORE_DIR
elif (BUNDLED_SCORE_DIR / "origin_method_losses.csv").is_file():
    SCORE_DIR = BUNDLED_SCORE_DIR
else:
    raise FileNotFoundError(
        "target-free bootstrap inputs were not found in the project archive "
        "or supplementary/paired_uncertainty_inputs/"
    )
LOSS_FILE = SCORE_DIR / "origin_method_losses.csv"
SUMMARY_FILE = SCORE_DIR / "score_summary_by_method_horizon.csv"
PAIRWISE_FILE = SCORE_DIR / "pairwise_method_comparisons.csv"
S8 = SUPP / "S8_unified_rolling_2018_2026.md"
OUT_CSV = SUPP / "paired_uncertainty_2018_2026.csv"
OUT_JSON = SUPP / "paired_uncertainty_2018_2026_metadata.json"

HORIZONS = (1, 3, 6, 12)
BLOCK_LENGTHS = (3, 6, 12)
BOOTSTRAP_DRAWS = 30_000
SEED = 417
BASELINE = "B3 multivariate"
METHOD_ORDER = (
    "No-change",
    "Seasonal-naive",
    "AR(1)",
    "SARIMA",
    "TimesFM-3 target-only",
    "BEAH rolling",
    "Direct rolling",
    "Fixed candidate",
    "Fixed candidate + gate",
)
TOL = 1e-12
START_MONTH = "2018-01"
END_MONTH = "2026-09"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_grid(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out: list[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + (m == 12), 1 if m == 12 else m + 1)
    return out


def cbb_indices(n: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draw circular moving blocks on the full monthly calendar, then truncate."""
    nblocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(draws, nblocks))
    offsets = np.arange(block, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return indices.reshape(draws, -1)[:, :n]


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Method vs B3 | Horizon | Mean paired difference (pp) | Block 3 interval (pp) | Block 6 interval (pp) | Block 12 interval (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        for h in HORIZONS:
            matching = [r for r in rows if r["method"] == method and r["horizon_months"] == h]
            assert len(matching) == 3
            by_block = {int(r["block_length_months"]): r for r in matching}
            point = float(by_block[3]["mean_loss_difference_pp"])
            intervals = []
            for block in BLOCK_LENGTHS:
                item = by_block[block]
                intervals.append(
                    f"[{float(item['percentile_2_5_pp']):.4f}, {float(item['percentile_97_5_pp']):.4f}]"
                )
            lines.append(
                f"| {method} | {h} | {point:+.4f} | {intervals[0]} | {intervals[1]} | {intervals[2]} |"
            )
    return "\n".join(lines)


def markdown_count_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Horizon | Block 3: min / median / max | Block 6: min / median / max | Block 12: min / median / max |",
        "|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        cells = []
        for block in BLOCK_LENGTHS:
            row = next(
                r for r in rows
                if r["method"] == METHOD_ORDER[0]
                and r["horizon_months"] == h
                and r["block_length_months"] == block
            )
            cells.append(
                f"{int(row['resampled_n_min'])} / {int(row['resampled_n_median'])} / {int(row['resampled_n_max'])}"
            )
        lines.append(f"| {h} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)


def main() -> None:
    calendar = month_grid(START_MONTH, END_MONTH)
    assert len(calendar) == 105
    rows: list[dict[str, str]] = []
    with LOSS_FILE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with SUMMARY_FILE.open(encoding="utf-8-sig", newline="") as handle:
        summaries = {
            (row["model"], int(row["horizon_months"])): row
            for row in csv.DictReader(handle)
        }
    with PAIRWISE_FILE.open(encoding="utf-8-sig", newline="") as handle:
        pairwise = list(csv.DictReader(handle))

    keyed: dict[tuple[str, int, str, str], float] = {}
    model_names = {row["model"] for row in rows}
    required = set(METHOD_ORDER) | {BASELINE}
    if model_names != required:
        raise ValueError(f"unexpected method inventory: {sorted(model_names)}")
    for row in rows:
        key = (
            row["model"],
            int(row["horizon_months"]),
            row["origin_id"],
            row["target_month"],
        )
        if key in keyed:
            raise ValueError(f"duplicate origin-level loss key: {key}")
        keyed[key] = float(row["mean_seed_absolute_error_pp"])

    output_rows: list[dict[str, object]] = []
    common_by_horizon: dict[int, set[tuple[str, str]]] = {}
    rng = np.random.default_rng(SEED)
    for h in HORIZONS:
        keys_by_method = {
            method: {
                (origin, target)
                for model, horizon, origin, target in keyed
                if model == method and horizon == h
            }
            for method in required
        }
        common = set.intersection(*(keys_by_method[m] for m in required))
        if len(common) != 99:
            raise ValueError(f"h={h}: expected 99 all-method paired origins, found {len(common)}")
        common_by_horizon[h] = common
        targets = [target for _, target in common]
        if len(set(targets)) != 99 or not set(targets).issubset(calendar):
            raise ValueError(f"h={h}: target-month key is not one-to-one on the planned calendar")

        differences: dict[str, np.ndarray] = {}
        index_by_target = {month: index for index, month in enumerate(calendar)}
        for method in METHOD_ORDER:
            diff = np.full(len(calendar), np.nan, dtype=np.float64)
            for origin, target in common:
                candidate_loss = keyed[(method, h, origin, target)]
                b3_loss = keyed[(BASELINE, h, origin, target)]
                diff[index_by_target[target]] = candidate_loss - b3_loss
            if int(np.isfinite(diff).sum()) != 99:
                raise ValueError(f"h={h}, {method}: expected 99 finite paired differences")
            differences[method] = diff
            point = float(np.nanmean(diff))
            summary = summaries[(method, h)]
            summary_difference = float(summary["pairwise_mean_loss_difference_vs_B3_pp"])
            mae_difference = float(summary["all10_common_MAE_pp"]) - float(summary["all10_common_B3_MAE_pp"])
            if abs(point - summary_difference) > 1e-8 or abs(point - mae_difference) > 1e-8:
                raise ValueError(f"h={h}, {method}: origin-loss mean does not match sealed score summary")
            matches = [
                row for row in pairwise
                if int(row["horizon_months"]) == h
                and (
                    (row["method_a"] == method and row["method_b"] == BASELINE)
                    or (row["method_a"] == BASELINE and row["method_b"] == method)
                )
            ]
            if len(matches) != 1 or int(matches[0]["pairwise_common_origins"]) != 99:
                raise ValueError(f"h={h}, {method}: expected one sealed 99-origin pairwise row vs B3")
            stored = float(matches[0]["mean_absolute_loss_difference_a_minus_b_pp"])
            stored_for_candidate = stored if matches[0]["method_a"] == method else -stored
            if abs(point - stored_for_candidate) > 1e-8:
                raise ValueError(f"h={h}, {method}: origin-loss mean does not match sealed pairwise table")

        for block in BLOCK_LENGTHS:
            indices = cbb_indices(len(calendar), block, BOOTSTRAP_DRAWS, rng)
            for method in METHOD_ORDER:
                diff = differences[method]
                point = float(np.nanmean(diff))
                finite = np.isfinite(diff)
                d = diff[finite]
                lower = int(np.sum(d < -TOL))
                ties = int(np.sum(np.abs(d) <= TOL))
                higher = int(np.sum(d > TOL))
                if (lower, ties, higher) != (99 - ties - higher, ties, higher) or lower + ties + higher != 99:
                    raise AssertionError("paired direction counts do not total 99")
                sampled = diff[indices]
                counts = np.isfinite(sampled).sum(axis=1)
                if np.any(counts == 0):
                    raise AssertionError("a calendar bootstrap replicate contains no paired origins")
                means = np.nansum(sampled, axis=1) / counts
                q025, q975 = np.quantile(means, [0.025, 0.975], method="linear")
                output_rows.append(
                    {
                        "method": method,
                        "baseline": BASELINE,
                        "horizon_months": h,
                        "planned_calendar_months": len(calendar),
                        "paired_origins": int(finite.sum()),
                        "mean_loss_difference_pp": point,
                        "lower_loss_origins": lower,
                        "ties": ties,
                        "higher_loss_origins": higher,
                        "block_length_months": block,
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "seed": SEED,
                        "percentile_2_5_pp": float(q025),
                        "percentile_97_5_pp": float(q975),
                        "resampled_n_min": int(counts.min()),
                        "resampled_n_median": float(np.median(counts)),
                        "resampled_n_max": int(counts.max()),
                        "calendar_missing_months": len(calendar) - int(finite.sum()),
                        "interval_scope": "descriptive percentile interval; not guaranteed-coverage CI",
                    }
                )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    table = markdown_table(output_rows)
    section = (
        "## Dependence-aware paired uncertainty for the unified rolling cohort\n\n"
        "For each horizon, we compared every method with B3 on the exact all-ten common set of 99 target-origin pairs. "
        "The paired loss difference is the method's origin-level mean absolute error minus B3's; negative values favor the named method. "
        "Because the common set omits six of the 105 planned target months, resampling used the full monthly target calendar and retained unavailable months as missing slots rather than compressing time. "
        "We generated 30,000 circular moving-block bootstrap replicates with block lengths of 3, 6, and 12 calendar months, using seed 417 and linear 2.5th/97.5th percentiles. "
        "Within each replicate, only available paired losses contribute to the mean; the observed pair count is 99, while resampled counts vary with the sampled missingness pattern. "
        "All contrasts use common bootstrap index sequences within each horizon and block length. These are descriptive sensitivity intervals, not formal confidence intervals with guaranteed coverage; no p-values or multiplicity-adjusted claims are made. "
        "Seeds are already aggregated within origin and are not extra time observations. The analysis is reproducible with `supplementary/code/paired_block_uncertainty_2018_2026.py`; machine-readable results and run metadata are `supplementary/paired_uncertainty_2018_2026.csv` and `supplementary/paired_uncertainty_2018_2026_metadata.json`. The metadata records the Python and NumPy versions and SHA-256 hashes of the loss input, score summaries, analysis code, and output.\n\n"
        + table
        + "\n\nThe original paired sample is 99 at every horizon; the number of usable pairs in a bootstrap replicate varies because the six unavailable calendar months are resampled as missing slots. The observed min/median/max resample sizes are:\n\n"
        + markdown_count_table(output_rows)
        + "\n\nThe intervals show how uncertainty changes with the temporal block length. Circular wraparound handles calendar edges. These summaries do not remove the retrospective status, overlapping-horizon dependence, six unavailable common months, or the single-country scope. They should be read with the 99-origin effect sizes and direction counts, not as a model ranking.\n"
    )
    text = S8.read_text(encoding="utf-8")
    begin = "<!-- BEGIN DEPENDENCE-AWARE PAIRED UNCERTAINTY -->"
    end = "<!-- END DEPENDENCE-AWARE PAIRED UNCERTAINTY -->"
    block = f"{begin}\n\n{section}\n{end}"
    if begin in text and end in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + block + text[stop:]
    else:
        anchor = "## Audit and reproduction"
        if anchor not in text:
            raise ValueError(f"insertion anchor missing from {S8}")
        text = text.replace(anchor, f"{block}\n\n{anchor}", 1)
    S8.write_text(text, encoding="utf-8", newline="\n")

    metadata = {
        "status": "PASS_DERIVED_LOSS_CALENDAR_CBB",
        "input_file": str(LOSS_FILE.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": sha256_file(LOSS_FILE),
        "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "score_summary_sha256": sha256_file(SUMMARY_FILE),
        "pairwise_summary_sha256": sha256_file(PAIRWISE_FILE),
        "output_csv": str(OUT_CSV.relative_to(ROOT)).replace("\\", "/"),
        "output_csv_sha256": sha256_file(OUT_CSV),
        "s8_sha256_after_update": sha256_file(S8),
        "target_values_loaded": False,
        "p1_accessed": False,
        "cohort": "unified 2018-01 to 2026-09 rolling run",
        "target_calendar_months": 105,
        "common_origins_per_horizon": {str(h): len(common_by_horizon[h]) for h in HORIZONS},
        "methods_compared_with": {"baseline": BASELINE, "comparators": list(METHOD_ORDER)},
        "paired_difference": "origin-level mean absolute loss of comparator minus B3 loss; negative favors comparator",
        "resampling_axis": "planned target-month calendar; missing common-origin months retained as NA positions",
        "resampling": {
            "type": "circular moving-block bootstrap",
            "block_lengths_months": list(BLOCK_LENGTHS),
            "draws_per_horizon_block_method": BOOTSTRAP_DRAWS,
            "seed": SEED,
            "percentiles": [2.5, 97.5],
            "quantile_method": "linear",
            "missing_slots": "preserved in resampled calendar and excluded from each replicate mean",
            "interval_interpretation": "descriptive sensitivity interval; no guaranteed nominal coverage",
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rows": len(output_rows),
        "sealed_score_point_estimate_crosschecks": 36,
    }
    OUT_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print("\nSelected h=3 and point-estimate-minimum contrasts:")
    for row in output_rows:
        if row["horizon_months"] in (3, 6, 12) and row["method"] in (
            "SARIMA",
            "BEAH rolling",
            "Fixed candidate + gate",
            "AR(1)",
        ):
            print(
                row["method"],
                "h=", row["horizon_months"],
                "L=", row["block_length_months"],
                "mean=", f"{float(row['mean_loss_difference_pp']):+.5f}",
                "interval=(", f"{float(row['percentile_2_5_pp']):+.5f}", ",", f"{float(row['percentile_97_5_pp']):+.5f}", ")",
            )


if __name__ == "__main__":
    main()
