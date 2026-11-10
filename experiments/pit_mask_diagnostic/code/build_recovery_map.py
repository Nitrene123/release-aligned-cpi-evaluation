#!/usr/bin/env python3
"""Build the conservative post-origin recovery map from the supplied audit files.

The builder never reads CPI target values. A value is admitted only when a
later unmasked PIT cell has a verified availability bound strictly after the
masked cell's forecast origin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    inp = args.input_dir
    b3 = pd.read_csv(inp / "b3_input_snapshots.csv", dtype={"context_reference_month": str})
    scored = pd.read_csv(inp / "scored_origin_horizon_keys.csv")
    pit = pd.read_csv(inp / "unified_pit_covariates.csv", dtype={"context_reference_month": str})
    audit = pd.read_csv(inp / "unified_covariate_audit.csv", low_memory=False,
                        dtype={"context_reference_month": str})
    if scored.duplicated(["origin_id", "horizon_months"]).any():
        raise ValueError("Duplicate scored origin/horizon key")
    if not (scored.groupby("horizon_months").size() == 99).all():
        raise ValueError("Expected 99 B3 scored origins at each registered horizon")
    score_origins = set(scored.origin_id.astype(str))
    mask = b3[b3.origin_id.astype(str).isin(score_origins) & b3.masked.astype(bool)].copy()
    mask["origin_ts"] = pd.to_datetime(mask.origin_timestamp_utc, utc=True)
    audit["available_ts"] = pd.to_datetime(audit.source_availability_upper_bound, utc=True, errors="coerce")
    pit["value_num"] = pd.to_numeric(pit.feature_value, errors="coerce")
    rows = []
    for r in mask.itertuples(index=False):
        a = audit[(audit.feature_name == r.feature_name) &
                  (audit.context_reference_month == r.context_reference_month)]
        a = a[(a.available_ts > r.origin_ts) &
              a.status.eq("VALUE_AND_CONSERVATIVE_TIME_BOUND_VERIFIED")]
        if a.empty:
            continue
        p = pit[(pit.feature_name == r.feature_name) &
                (pit.context_reference_month == r.context_reference_month)]
        candidates = a.merge(p, on=["origin_id", "feature_name", "context_reference_month"],
                             how="inner", suffixes=("_audit", "_pit"))
        candidates = candidates[(~candidates.masked) & candidates.value_num.notna()]
        candidates = candidates.sort_values(["available_ts", "origin_id"])
        if candidates.empty:
            continue
        first_time = candidates.iloc[0].available_ts
        tied = candidates[candidates.available_ts == first_time]
        values = np.asarray(tied.value_num, dtype=np.float32)
        if len(values) > 1 and not np.all(values == values[0]):
            raise ValueError(f"Ambiguous value at first later release: {r.origin_id}/{r.feature_name}/{r.context_reference_month}")
        x = candidates.iloc[0]
        rows.append({
            "origin_id": r.origin_id,
            "origin_month": r.origin_month,
            "origin_timestamp_utc": r.origin_timestamp_utc,
            "feature_name": r.feature_name,
            "context_reference_month": r.context_reference_month,
            "mask_reason": r.mask_reason,
            "recovered_value": float(np.float32(x.value_num)),
            "first_verified_available_at_utc": x.available_ts.isoformat(),
            "provider_origin_id": x.origin_id,
            "provider_source_status": x.status,
            "provider_source_url": x.source_url,
            "provider_proof_basis": x.proof_basis,
            "provider_availability_evidence": x.availability_evidence,
        })
    out = pd.DataFrame(rows)
    keys = ["origin_id", "feature_name", "context_reference_month"]
    if out.duplicated(keys).any():
        raise ValueError("Duplicate recovery-map key")
    if len(out) and (pd.to_datetime(out.first_verified_available_at_utc, utc=True) <=
                     pd.to_datetime(out.origin_timestamp_utc, utc=True)).any():
        raise ValueError("Recovery map contains a non-post-origin value")
    out = out.sort_values(["origin_month", "origin_id", "context_reference_month", "feature_name"])
    out.to_csv(args.output_dir / "mask_recovery_map.csv", index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "status": "RECOVERY_MAP_BUILT_WITHOUT_TARGET_VALUES",
        "scored_origin_count_by_horizon": {str(h): int(n) for h, n in scored.groupby("horizon_months").size().items()},
        "masked_cells_in_scored_origin_union": int(len(mask)),
        "recovered_cells": int(len(out)),
        "unrecoverable_or_unverified_cells": int(len(mask) - len(out)),
        "recovered_by_feature": out.feature_name.value_counts().to_dict(),
        "recovered_by_mask_reason": out.mask_reason.value_counts().to_dict(),
        "all_availability_bounds_strictly_post_origin": True,
        "target_values_read": False,
    }
    (args.output_dir / "recovery_map_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
