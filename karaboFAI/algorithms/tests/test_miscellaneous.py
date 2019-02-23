import unittest

import numpy as np

from karaboFAI.algorithms import normalize_curve


class TestMiscellaneous(unittest.TestCase):

    def test_normalizecurve(self):
        y = np.array([1, 1, 1, 1, 1, 1])
        x = np.array([0, 1, 2, 3, 4, 5])

        y_normalized = normalize_curve(y, x)
        self.assertTrue(np.array_equal(y_normalized, np.array([0.2]*6)))

        y_normalized = normalize_curve(y, x, 1, 3)
        self.assertTrue(np.array_equal(y_normalized, np.array([0.5]*6)))

        y = np.array([1, -1, 1, -1, 1, -1])
        x = np.array([0, 1, 2, 3, 4, 5])
        # normalized by 0
        with self.assertRaises(ValueError):
            normalize_curve(y, x)
        with self.assertRaises(ValueError):
            normalize_curve(y, x, 2, 3)

        y = np.array([0, 0, 0, 0, 0, 0])
        x = np.array([0, 1, 2, 3, 4, 5])
        y_normalized = normalize_curve(y, x)
        self.assertTrue(np.array_equal(y_normalized, np.array([0]*6)))
