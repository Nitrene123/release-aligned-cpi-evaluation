"""Run the shipped unittest suite and write a machine-readable test receipt."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


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
        super().addSuccess(test); self._mark(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err); self._mark(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err); self._mark(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason); self._mark(test, "SKIP", reason)


def main() -> int:
    tests_dir = Path(__file__).resolve().parent
    audit_path = tests_dir.parent / "src" / "rae" / "audit_point_in_time.py"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("PIT_AUDIT_TEST_RESULTS.json"))
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_point_in_time.py")
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    payload = {
        "suite": "point-in-time input-contract synthetic tests",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "command": "python record_pit_test_results.py",
        "audit_code_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "test_code_sha256": hashlib.sha256(Path(__file__).with_name("test_point_in_time.py").read_bytes()).hexdigest(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests": result.records,
        "text_output": stream.getvalue(),
        "scope_note": "Synthetic contract fixtures only; not an audit or score of the historical study inputs.",
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "tests_run", "failures", "errors", "skipped")}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
