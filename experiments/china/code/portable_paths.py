"""Resolve the CPI project root without a machine-specific absolute path."""
from __future__ import annotations

import os
from pathlib import Path

EXPECTED_LAYOUT = Path("revision_experiments_v2/rolling_foundation_matched_v2")


def project_root(script_file: str | Path) -> Path:
    configured = os.environ.get("ECOFM_CPI_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / EXPECTED_LAYOUT).is_dir():
            raise RuntimeError(
                f"ECOFM_CPI_ROOT={root} does not contain {EXPECTED_LAYOUT}; "
                "set it to the EconoFM_CPI project root"
            )
        return root
    starts = [Path(script_file).resolve().parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate not in seen:
                seen.add(candidate)
                if (candidate / EXPECTED_LAYOUT).is_dir():
                    return candidate
    raise RuntimeError(
        "Could not locate the EconoFM_CPI root. Set ECOFM_CPI_ROOT to the "
        "directory containing revision_experiments_v2/rolling_foundation_matched_v2."
    )
