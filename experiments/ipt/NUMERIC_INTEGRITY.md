# Numeric integrity and replay comparison

`numeric_integrity.py` keeps two integrity concepts separate:

- Raw SHA-256 identifies the exact bytes of an archived file.
- The normalized numeric fingerprint parses decimal text and hashes records in deterministic key order. Measured values are quantized with decimal `ROUND_HALF_EVEN` to eight decimal places; exact numeric fields such as sample counts are canonicalized without rounding.

For replay comparison, the tool first requires identical row-key sets and matching categorical fields. It then checks input-context values with an absolute tolerance of `1e-9` percentage points and forecast predictions with `1e-6` percentage points (`rtol=0`). Derived score fields use an absolute tolerance of `1e-6` in each field's native units (`rtol=0`), including squared-error fields. The comparison covers contexts, both forecast files, and target-free derived score tables. It does not open the target workbook.

The archived reference fingerprints are in `numeric_integrity_reference.json`. To compare another replay directory, run:

```powershell
python numeric_integrity.py --reference-dir . --candidate-dir D:/path/to/replay --output D:/path/to/new-comparison.json
```

The output path must be new. `PASS_WITHIN_TOLERANCE` means that key and categorical checks match and all compared numerical cells fall within the stated tolerances. It does not imply bitwise equality. The current archive has a baseline fingerprint; no separate cross-platform replay has yet been compared against it.
