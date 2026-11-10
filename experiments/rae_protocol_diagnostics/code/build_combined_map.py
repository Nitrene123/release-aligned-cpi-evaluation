from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TIMING = ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/inputs/mask_recovery_map.csv"
VINTAGE = HERE / "vintage_map/brent_vintage_replacement_map.csv"
OUT = HERE / "combined_map"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    keys = ["origin_id", "context_reference_month", "feature_name"]
    t = pd.read_csv(TIMING, dtype={"context_reference_month": str})
    v = pd.read_csv(VINTAGE, dtype={"context_reference_month": str})
    if len(t) != 104 or len(v) != 14 or t.duplicated(keys).any() or v.duplicated(keys).any():
        raise ValueError("Unexpected source map coverage or duplicate cell keys")
    if set(map(tuple, t[keys].to_numpy())) & set(map(tuple, v[keys].to_numpy())):
        raise ValueError("Timing and vintage maps must be disjoint")
    t = t[keys + ["recovered_value"]].assign(modification_type="timing")
    v = v[keys + ["recovered_value"]].assign(modification_type="vintage")
    both = pd.concat([t, v], ignore_index=True).sort_values(keys)
    OUT.mkdir()
    both.to_csv(OUT / "mask_recovery_map.csv", index=False, float_format="%.10g", lineterminator="\n")
    data = {"status": "DISJOINT_PARTIAL_MAP_READY", "timing_cells": len(t), "vintage_cells": len(v),
            "combined_cells": len(both), "timing_sha256": sha(TIMING), "vintage_sha256": sha(VINTAGE),
            "protocol_sha256": sha(HERE / "PROTOCOL_COMBINED_PARTIAL.md"), "script_sha256": sha(Path(__file__)),
            "map_sha256": sha(OUT / "mask_recovery_map.csv"), "target_outcomes_read": False}
    (OUT / "map_metadata.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
