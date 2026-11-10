from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
IN = HERE / "inputs"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    meta = json.loads((HERE / "map_metadata.json").read_text(encoding="utf-8"))
    if sha(HERE / "brent_vintage_replacement_map.csv") != meta["map_sha256"]:
        raise ValueError("Vintage map hash mismatch")
    if sha(HERE / "PROTOCOL_PARTIAL_VINTAGE.md") != meta["protocol_sha256"]:
        raise ValueError("Protocol hash mismatch")
    original = json.loads((IN / "files_sha256.json").read_text(encoding="utf-8"))
    for name, expected in original.items():
        if sha(IN / name) != expected:
            raise ValueError(f"Copied S11 input hash mismatch: {name}")
    shutil.copy2(HERE / "brent_vintage_replacement_map.csv", IN / "mask_recovery_map.csv")
    shutil.copy2(HERE / "PROTOCOL_PARTIAL_VINTAGE.md", IN / "PROTOCOL_PARTIAL_VINTAGE.md")
    shutil.copy2(HERE / "map_metadata.json", IN / "map_metadata.json")
    manifest = {p.name: sha(p) for p in sorted(IN.iterdir()) if p.is_file() and p.name != "files_sha256.json"}
    (IN / "files_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "INPUTS_READY", "input_count": len(manifest), "replacement_cells": meta["replacement_cells"], "runner_sha256": sha(HERE / "run_brent_vintage.py"), "input_manifest_sha256": sha(IN / "files_sha256.json")}, indent=2))


if __name__ == "__main__":
    main()
