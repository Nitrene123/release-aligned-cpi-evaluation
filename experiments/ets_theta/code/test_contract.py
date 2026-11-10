"""Synthetic tests for the new historical baseline contract; no study data."""
from __future__ import annotations

import unittest
from datetime import timezone

import numpy as np

import run_ets_theta as experiment


class TestBaselineContract(unittest.TestCase):
    def test_month_arithmetic_across_year(self):
        self.assertEqual(experiment.add_month("2017-12", 1), "2018-01")
        self.assertEqual(experiment.add_month("2018-01", -12), "2017-01")

    def test_timezone_normalization(self):
        china = experiment.stamp("2018-01-10 09:30", china_local=True)
        utc = experiment.stamp("2018-01-10T01:30:00Z")
        self.assertEqual(china, utc)
        self.assertEqual(china.tzinfo, timezone.utc)

    def test_ets_theta_deterministic_and_finite(self):
        index = np.arange(80, dtype=float)
        y = 1.0 + .01 * index + np.sin(.3 * index)
        for method in experiment.NEW_METHODS:
            first, first_audit = experiment.fit_path(method, y, 12)
            second, second_audit = experiment.fit_path(method, y, 12)
            self.assertEqual(first_audit["fit_status"], "FIT_COMPLETE")
            self.assertEqual(second_audit["fit_status"], "FIT_COMPLETE")
            self.assertEqual(len(first), 12)
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_allclose(first, second, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
