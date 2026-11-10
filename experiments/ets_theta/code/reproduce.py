"""Reproduce ETS/Theta forecasts and scores from supplied snapshots in a new directory."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("original_runner", HERE / "run_ets_theta.py")
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def read_csv(path: Path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def configure() -> None:
    runner.US = HERE / "inputs/US"
    runner.US_RUN = runner.US / "run"
    runner.CN = HERE / "inputs/CN"
    runner.CN_RUN = runner.CN / "run"
    for country in ("US", "CN"):
        if country == "US":
            runner.CFG[country].update({
                "planned": runner.US_RUN / "audit_context/planned_target_origin_horizons.csv",
                "training": runner.US / "data/us_cpi_first_release_training_label_panel_2012_2024.csv",
                "old_forecasts": runner.US_RUN / "statistics/statistical_forecasts_unscored.csv",
                "old_losses": runner.US_RUN / "score_us_2018_2024_once_v1/origin_method_losses.csv",
                "targets": runner.US / "data/us_cpi_first_release_targets_2018_2024.csv",
            })
        else:
            runner.CFG[country].update({
                "planned": runner.CN_RUN / "planned_origin_horizon_keys.csv",
                "training": runner.CN / "source_data/unified_cpi_panel.csv",
                "old_forecasts": runner.CN_RUN / "source/statistical_forecast_predictions.csv",
                "old_losses": runner.CN / "score_v2_clean_20261031/origin_method_losses.csv",
                "targets": runner.CN / "score_inputs/score_target_extension_2025_2026.csv",
            })


def compare(country: str, output: Path) -> None:
    old = HERE / "archived" / country
    new = output / "output" / country
    fields = ("model", "origin_id", "origin_month", "origin_timestamp_utc", "target_month", "horizon_months")
    a = {tuple(row[x] for x in fields): row for row in read_csv(old / "forecast/forecasts.csv")}
    b = {tuple(row[x] for x in fields): row for row in read_csv(new / "forecast/forecasts.csv")}
    if len(a) != len(b) or set(a) != set(b):
        raise ValueError(f"{country}: forecast keys differ")
    max_diff = max(abs(float(a[k]["prediction"]) - float(b[k]["prediction"])) for k in a)
    if max_diff > 1e-8:
        raise ValueError(f"{country}: forecast replay difference {max_diff}")
    x = {(r["model"], r["horizon_months"]): r for r in read_csv(old / "score/mae_summary.csv")}
    y = {(r["model"], r["horizon_months"]): r for r in read_csv(new / "score/mae_summary.csv")}
    if set(x) != set(y):
        raise ValueError(f"{country}: score-summary keys differ")
    for k in x:
        for column in ("mae_native_pp", "mae_old_common10_pp", "mae_common12_pp"):
            if abs(float(x[k][column]) - float(y[k][column])) > 1e-8:
                raise ValueError(f"{country}: {k} {column} mismatch")
    print(f"{country}: {len(a)} forecast rows and {len(x)} MAE cells verified; max forecast diff={max_diff:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="new output directory, not inside the supplied bundle")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out == HERE or HERE in out.parents:
        raise ValueError("Choose a new output directory outside the supplied bundle")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(HERE / "PROTOCOL.md", out / "PROTOCOL.md")
    configure()
    runner.HOME = out
    for country in ("US", "CN"):
        runner.forecast(country)
        runner.score(country)
        compare(country, out)


if __name__ == "__main__":
    main()
