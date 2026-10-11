import itertools
import unittest

import numpy as np
from sklearn.metrics import log_loss

from shiftstudy.calibration import (conformal_sets, conformal_threshold,
                                   fit_temperature, set_metrics, temperature_scale)
from shiftstudy.metrics import (calibration_bins, checked_probabilities,
                               nll, prediction_metrics, risk_coverage)


class MetricTests(unittest.TestCase):
    def test_perfect_and_uniform_predictions(self):
        y = np.array([0, 1, 2])
        perfect = prediction_metrics(np.eye(3), y)
        self.assertEqual(perfect["accuracy"], 1.)
        for metric in ("nll", "brier", "ece_15", "aurc"):
            self.assertEqual(perfect[metric], 0.)
        uniform = prediction_metrics(np.full((3, 3), 1 / 3), y)
        self.assertAlmostEqual(uniform["nll"], np.log(3))
        self.assertAlmostEqual(uniform["brier"], 2 / 3)
        self.assertAlmostEqual(uniform["ece_15"], 0.)

    def test_nll_agrees_with_independent_library(self):
        p = np.array([[.2, .8], [.7, .3], [.4, .6]])
        y = np.array([1, 0, 1])
        self.assertAlmostEqual(nll(p, y), log_loss(y, p))

    def test_confidence_one_in_last_bin(self):
        b = calibration_bins(np.eye(2), np.array([0, 1]), 10)
        self.assertEqual(b["count"][-1], 2)
        self.assertEqual(b["count"].sum(), 2)

    def test_ties_are_order_invariant_and_random_reference_exact(self):
        p = np.tile([.7, .3], (4, 1))
        y = np.array([0, 0, 1, 1])
        for order in itertools.permutations(range(4)):
            c = risk_coverage(p[list(order)], y[list(order)])
            np.testing.assert_allclose(c["risk"], .5)
            self.assertAlmostEqual(c["aurc"], .5)

    def test_selective_curve_and_oracle(self):
        p = np.array([[.9, .1], [.8, .2], [.4, .6]])
        y = np.array([0, 1, 1])
        c = risk_coverage(p, y)
        np.testing.assert_allclose(c["risk"], [0., .5, 1 / 3])
        np.testing.assert_allclose(c["oracle_risk"], [0., 0., 1 / 3])
        self.assertEqual(c["coverage"][-1], 1.)

    def test_invalid_inputs_rejected(self):
        for p in ([], [[np.nan, .5]], [[.2, .2]], [[-.1, 1.1]]):
            with self.assertRaises(ValueError):
                checked_probabilities(p)
        with self.assertRaises(ValueError):
            nll([[.5, .5]], np.array([0.5]))

    def test_temperature_calibration_and_argmax(self):
        p = np.tile([.99, .01], (10, 1))
        y = np.array([0] * 7 + [1] * 3)
        fit = fit_temperature(p, y)
        self.assertGreater(fit["temperature"], 1.)
        self.assertLess(fit["nll_after"], fit["nll_before"])
        scaled = temperature_scale(p, fit["temperature"])
        np.testing.assert_array_equal(p.argmax(1), scaled.argmax(1))
        np.testing.assert_allclose(temperature_scale(p, 1.), p)
        with self.assertRaises(ValueError):
            temperature_scale(p, 0.)

    def test_conformal_exact_rank_and_ties(self):
        p = np.column_stack([np.arange(1, 10) / 10, 1 - np.arange(1, 10) / 10])
        y = np.zeros(9, dtype=int)
        self.assertAlmostEqual(conformal_threshold(p, y, .2), .8)
        threshold = conformal_threshold(p, y, .01)
        self.assertEqual(threshold, 1.)
        self.assertTrue(conformal_sets(p, threshold).all())
        self.assertTrue(conformal_sets([[.2, .8]], .8)[0, 0])
        self.assertEqual(set_metrics(conformal_sets([[.5, .5]], .1), np.array([0]))["empty_set_rate"], 1.)


if __name__ == "__main__":
    unittest.main()
