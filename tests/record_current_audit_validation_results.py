"""Run and record the current I1-I5 portable-auditor contract suites."""
from __future__ import annotations

import hashlib
import io
import json
import platform
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "rae" / "audit_point_in_time.py").exists():
    AUDIT_PATH = ROOT / "src" / "rae" / "audit_point_in_time.py"
    TEST_DIR = ROOT / "tests"
else:
    TEST_DIR = ROOT / "code"
    AUDIT_PATH = TEST_DIR / "audit_point_in_time.py"
SUITES = [
    ("general_input_contract", "test_point_in_time.py"),
    ("dependency_graph_I3", "test_dependency_graph.py"),
    ("model_input_eligibility_I4", "test_model_input_eligibility.py"),
]


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        super().startTest(test)
        self.records.append({"test": test.id(), "status": "RUNNING"})

    def _mark(self, test, status, detail=""):
        for row in reversed(self.records):
            if row["test"] == test.id() and row["status"] == "RUNNING":
                row["status"] = status
                if detail:
                    row["detail"] = detail
                return

    def addSuccess(self, test):
        super().addSuccess(test)
        self._mark(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._mark(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._mark(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._mark(test, "SKIP", reason)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("audit_validation_output"),
                        help="new output directory; existing report files are never overwritten")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "current_audit_validation_results.json"
    dependency_out = output_dir / "dependency_graph_test_results.json"
    if out.exists() or dependency_out.exists():
        parser.error("output report already exists; choose a fresh --output-dir")
    suite_results = []
    all_records = []
    total = failures = errors = skipped = 0
    for suite_name, pattern in SUITES:
        suite = unittest.defaultTestLoader.discover(str(TEST_DIR), pattern=pattern)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult).run(suite)
        all_records.extend(result.records)
        total += result.testsRun
        failures += len(result.failures)
        errors += len(result.errors)
        skipped += len(result.skipped)
        suite_results.append({
            "suite": suite_name,
            "pattern": pattern,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "status": "PASS" if result.wasSuccessful() else "FAIL",
            "text_output": stream.getvalue(),
        })
    code_files = [AUDIT_PATH, *(TEST_DIR / pattern for _, pattern in SUITES), Path(__file__)]
    payload = {
        "schema_version": "1.0",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "command": "python tests/record_current_audit_validation_results.py --output-dir <new-directory>",
        "scope": "Portable target-free audit contract tests for release/vintage/key validation, I3 dependency-DAG checks, and I4 model-origin feature-lag-reference eligibility.",
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in code_files},
        "suite_results": suite_results,
        "tests_run": total,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "status": "PASS" if failures == 0 and errors == 0 else "FAIL",
        "tests": all_records,
        "coverage_denominator_rule": "All planned model-origin-horizon keys remain in the planned denominator; an input-ineligible forecast is separately reported and excluded from evaluation-eligible coverage.",
        "scope_note": "Synthetic bundle tests only. These tests do not re-run TimesFM-3 or independently establish that a supplied model manifest is complete.",
    }
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    dependency = suite_results[1]
    s22_rows = [row for row in all_records if "test_s22_" in row["test"]]
    dependency_payload = {
        "suite": "portable dependency-DAG contract and S22 regression tests",
        "run_at_utc": payload["run_at_utc"],
        "python_version": platform.python_version(),
        "command": "python -m unittest discover -s tests -p \"test_dependency_graph.py\" -v",
        "audit_code_sha256": payload["source_sha256"][str(AUDIT_PATH.relative_to(ROOT)).replace("\\", "/")],
        "test_code_sha256": payload["source_sha256"][str((TEST_DIR / "test_dependency_graph.py").relative_to(ROOT)).replace("\\", "/")],
        "tests_run": dependency["tests_run"],
        "failures": dependency["failures"],
        "errors": dependency["errors"],
        "skipped": dependency["skipped"],
        "status": dependency["status"],
        "tests": [row for row in all_records if row["test"].startswith("test_dependency_graph.")],
        "s22_regression_cases": [
            {"case_id": row["test"].rsplit("test_s22_", 1)[-1].split("_future_parent", 1)[0].upper(),
             "status": row["status"]}
            for row in s22_rows
        ],
        "historical_s22_metrics_modified": False,
        "scope_note": "Targeted regressions for known S22 misses; this report is not a new held-out detection-rate estimate.",
    }
    with dependency_out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dependency_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: payload[k] for k in ("status", "tests_run", "failures", "errors", "skipped")}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
