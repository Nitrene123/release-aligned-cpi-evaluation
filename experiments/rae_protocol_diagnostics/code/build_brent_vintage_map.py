from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
S11 = ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/inputs"
REGISTRY = ROOT / "manuscript/ars_full_v2_4/reproducibility/matched_origin_2018_2024_v1/work/pit_audit_complete_rebuild_20261030/run_20261030T044200Z_pit_audit_v2/vintages.csv"
OUT = HERE / "vintage_map"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    b3 = pd.read_csv(S11 / "b3_input_snapshots.csv", low_memory=False, dtype={"context_reference_month": str})
    scored = pd.read_csv(S11 / "scored_origin_horizon_keys.csv")
    v = pd.read_csv(REGISTRY, dtype=str)
    v = v[v.series_id.eq("BRENT_LOG_YOY_PCT")].copy()
    if v.vintage_id.duplicated().any():
        raise ValueError("Non-unique Brent vintage identifiers")
    v["available_ts"] = pd.to_datetime(v.available_at, utc=True, errors="raise")
    v["numeric_value"] = pd.to_numeric(v.value, errors="raise")
    v = v.sort_values(["observation_month", "available_ts", "vintage_id"])
    tie = v.groupby(["observation_month", "available_ts"]).numeric_value.nunique()
    if (tie > 1).any():
        raise ValueError("Conflicting values at a vintage timestamp")
    latest = v.groupby("observation_month", sort=False).tail(1).copy()
    latest = latest.rename(columns={"observation_month": "latest_observation_month", "vintage_id": "latest_vintage_id", "available_ts": "latest_available_ts", "numeric_value": "latest_value", "source_sha256": "latest_source_sha256"})
    x = b3[b3.origin_id.isin(scored.origin_id) & b3.feature_name.eq("BRENT_LOG_YOY_PCT") & ~b3.masked.astype(bool)].copy()
    x["origin_ts"] = pd.to_datetime(x.origin_timestamp_utc, utc=True, errors="raise")
    x = x.merge(v[["vintage_id", "observation_month", "numeric_value"]], on="vintage_id", how="left", validate="many_to_one")
    found = x.observation_month.notna()
    same_month = x.source_reference_month.astype(str).eq(x.observation_month.astype(str))
    same_value = np.isclose(pd.to_numeric(x.value, errors="coerce"), x.numeric_value, atol=1e-5, rtol=0, equal_nan=False)
    if not (same_month[found] & same_value[found]).all():
        raise ValueError("Captured input differs from its source vintage record")
    x = x.merge(latest[["latest_observation_month", "latest_vintage_id", "latest_available_ts", "latest_value", "latest_source_sha256"]], left_on="observation_month", right_on="latest_observation_month", how="left", validate="many_to_one")
    if x.duplicated(["origin_id", "context_reference_month", "feature_name"]).any():
        raise ValueError("Duplicate B3 cell key")
    changed = found & (x.latest_available_ts > x.origin_ts) & (x.value.to_numpy(dtype=np.float32) != x.latest_value.to_numpy(dtype=np.float32))
    chosen = x.loc[changed].copy()
    if chosen.empty:
        raise ValueError("No eligible changed cells; inspect registry coverage")
    chosen["recovered_value"] = chosen.latest_value.astype(np.float32)
    chosen = chosen[["origin_id", "origin_month", "origin_timestamp_utc", "context_reference_month", "feature_name", "source_reference_month", "vintage_id", "value", "latest_vintage_id", "latest_available_ts", "recovered_value", "latest_source_sha256"]].sort_values(["origin_month", "context_reference_month"])
    OUT.mkdir()
    chosen.to_csv(OUT / "brent_vintage_replacement_map.csv", index=False, lineterminator="\n", float_format="%.10g")
    summary = {"status": "PARTIAL_BRENT_VINTAGE_MAP_BUILT", "source_registry_sha256": sha(REGISTRY),
               "snapshot_sha256": sha(S11 / "b3_input_snapshots.csv"), "scored_keys_sha256": sha(S11 / "scored_origin_horizon_keys.csv"),
               "protocol_sha256": sha(HERE / "PROTOCOL_PARTIAL_VINTAGE.md"), "script_sha256": sha(Path(__file__)),
               "observed_brent_cells": len(x), "registry_matched_cells": int(found.sum()),
               "registry_unmatched_cells": int((~found).sum()), "replacement_cells": len(chosen),
               "replacement_origins": int(chosen.origin_id.nunique()), "replacement_source_months": sorted(chosen.source_reference_month.astype(str).unique().tolist()),
               "latest_registry_availability_utc": v.available_ts.max().isoformat(),
               "map_sha256": sha(OUT / "brent_vintage_replacement_map.csv"), "target_outcomes_read": False}
    (OUT / "map_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
