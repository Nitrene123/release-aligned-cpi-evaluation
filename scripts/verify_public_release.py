#!/usr/bin/env python3
"""Check the public release for forbidden materials and verify its file manifest."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_DIRS = {".git", "__pycache__", "node_modules"}
FORBIDDEN_SUFFIXES = {
    ".xlsx", ".xls", ".safetensors", ".pt", ".pth", ".npz", ".pkl",
    ".pickle", ".pyc", ".log", ".zip", ".pem", ".key",
}
FORBIDDEN_NAMES = {
    "contexts_unscored.jsonl", "server_private_input_manifest.json",
    "target_ledger.csv", "target_values.csv", "actual_targets.csv",
}
PRIVATE_TEXT = [
    re.compile("/data/" + "zyb/EconoFM_CPI", re.I),
    re.compile(r"D:\\\\?codefield\\\\gpt\\\\" + "ARS_TimesFM3_CPI", re.I),
    re.compile(r"C:\\\\?Users\\\\" + "77235", re.I),
    re.compile("20190058@wfu" + r"\\.edu\\.cn", re.I),
]


def files_to_check() -> list[Path]:
    found = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        found.append(path)
    return sorted(found, key=lambda p: p.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_release(paths: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"forbidden file type: {relative}")
        if path.name.lower() in FORBIDDEN_NAMES:
            issues.append(f"forbidden private/raw-input filename: {relative}")
        if any(part.lower() in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
            issues.append(f"generated or dependency directory present: {relative}")
        try:
            raw = gzip.open(path, "rb").read() if path.suffix.lower() == ".gz" else path.read_bytes()
            content = raw.decode("utf-8", errors="ignore")
        except OSError as exc:
            issues.append(f"cannot inspect {relative}: {exc}")
            continue
        for pattern in PRIVATE_TEXT:
            if pattern.search(content):
                issues.append(f"private path or account detail found in {relative}")
                break
    return issues


def expected_manifest(paths: list[Path]) -> str:
    return "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths)


def verify_manifest(paths: list[Path]) -> list[str]:
    if not MANIFEST.is_file():
        return ["MANIFEST.sha256 is missing; run with --write-manifest after review"]
    expected = expected_manifest(paths)
    actual = MANIFEST.read_text(encoding="utf-8")
    return [] if actual == expected else ["MANIFEST.sha256 does not match the current files"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true", help="write the reviewed SHA-256 manifest")
    args = parser.parse_args()
    paths = files_to_check()
    issues = inspect_release(paths)
    if args.write_manifest and not issues:
        MANIFEST.write_text(expected_manifest(paths), encoding="utf-8", newline="\n")
        print(f"PASS: wrote manifest for {len(paths)} files")
    else:
        issues.extend(verify_manifest(paths))
        if issues:
            print("FAIL")
            for issue in issues:
                print(f"- {issue}")
            return 1
        print(f"PASS: {len(paths)} files checked; manifest matches")
    if issues:
        print("FAIL")
        for issue in issues:
            print(f"- {issue}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
