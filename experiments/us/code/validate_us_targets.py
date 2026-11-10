#!/usr/bin/env python3
"""Target-free integrity checks for the 2018-2024 U.S. CPI release ledger."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGETS = DATA / "us_cpi_first_release_targets_2018_2024.csv"
EVIDENCE = DATA / "us_bls_release_evidence_2018_2024.jsonl"
CROSSWALK = DATA / "us_cpi_bls_release_crosswalk_2018_2024.csv"
CANDIDATES = DATA / "us_cpi_candidate_first_vintage_2018_2024.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    targets = read_csv(TARGETS)
    evidence = [json.loads(line) for line in EVIDENCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    crosswalk = read_csv(CROSSWALK)
    candidates = read_csv(CANDIDATES)

    expected = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            expected.append(f"{year}-{month:02d}")

    require(len(targets) == 84, f"target row count is {len(targets)}, expected 84")
    require(len(evidence) == 84, f"evidence row count is {len(evidence)}, expected 84")
    require(len(crosswalk) == 84, f"crosswalk row count is {len(crosswalk)}, expected 84")
    require(len(candidates) == 84, f"candidate row count is {len(candidates)}, expected 84")
    require([r["target_reference_month"] for r in targets] == expected, "target months are missing, duplicated, or out of order")
    require([r["target_month"] for r in evidence] == expected, "evidence months are missing, duplicated, or out of order")

    cross_by_month = {r["reference_month"]: r for r in crosswalk}
    candidate_by_month = {r["reference_month"][:7]: r for r in candidates}
    for t, e in zip(targets, evidence, strict=True):
        month = t["target_reference_month"]
        cw = cross_by_month[month]
        cand = candidate_by_month[month]
        official = float(t["target_first_release_yoy_pp"])
        ratio = float(cand["yoy_percent"])
        evidence_value = float(e["first_release_yoy_pp"])
        require(t["country"] == "US", f"{month}: wrong country")
        require(t["series_id"] == "CUUR0000SA0", f"{month}: wrong CPI series")
        require(t["release_page_url"].startswith("https://www.bls.gov/news.release/archives/cpi_"), f"{month}: non-BLS release URL")
        require(t["release_time_et"] == "08:30", f"{month}: unexpected release time")
        require(t["release_date_et"] == cw["bls_release_date"], f"{month}: release date differs from crosswalk")
        require(t["release_timestamp_et"] == cw["bls_release_timestamp_et"], f"{month}: ET timestamp differs from crosswalk")
        require(t["release_timestamp_utc"] == cw["bls_release_timestamp_utc"], f"{month}: UTC timestamp differs from crosswalk")
        require(e["source_url"] == t["release_page_url"], f"{month}: evidence URL mismatch")
        require(e["actual_release_time_page"]["date"] == t["release_date_et"], f"{month}: page date mismatch")
        require(e["actual_release_time_page"]["time"] == "08:30", f"{month}: page time mismatch")
        require(e["checks"]["page_matches_target_month"] is True, f"{month}: page heading mismatch")
        require(e["checks"]["page_date_matches_crosswalk"] is True, f"{month}: page/crosswalk date mismatch")
        require(e["checks"]["page_yoy_matches_candidate_rounded"] is True, f"{month}: release value mismatch")
        require(round(ratio + 1e-10, 1) == official, f"{month}: official target differs from rounded candidate ratio")
        require(evidence_value == official, f"{month}: evidence value differs from target table")
        require(e["raw_response_bytes_archived"] is False, f"{month}: archive provenance must not claim raw bytes")
        require(datetime.fromisoformat(t["release_timestamp_utc"].replace("Z", "+00:00")).utcoffset().total_seconds() == 0, f"{month}: invalid UTC timestamp")

    print(json.dumps({
        "status": "PASS",
        "target_months": 84,
        "first_month": expected[0],
        "last_month": expected[-1],
        "official_release_pages": 84,
        "page_to_crosswalk_release_date_matches": 84,
        "published_value_to_rounded_first_vintage_ratio_matches": 84,
        "raw_release_response_bytes_archived": False,
        "scope": "target/source audit only; no forecast generation or target scoring"
    }, indent=2))


if __name__ == "__main__":
    main()
