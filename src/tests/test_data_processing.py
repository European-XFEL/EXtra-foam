import unittest

import numpy as np

from fxeAzimuthalIntegration.data_processing import (
    sub_array_with_range, integrate_curve
)


class TestDataProcessor(unittest.TestCase):
    def test_subarraywithrange(self):
        y = np.array([6, 5, 4, 3, 2, 1])
        x = np.array([0, 1, 2, 3, 4, 5])

        new_y, new_x = sub_array_with_range(y, x)
        self.assertTrue(np.array_equal(new_y, y))
        self.assertTrue(np.array_equal(new_x, x))

        new_y, new_x = sub_array_with_range(y, x, (1, 3))
        self.assertTrue(np.array_equal(new_y, np.array([5, 4, 3])))
        self.assertTrue(np.array_equal(new_x, np.array([1, 2, 3])))

    def test_integratecurve(self):
        y = np.array([6, 5, 4, 3, 2, 1])
        x = np.array([0, 1, 2, 3, 4, 5])

        itgt = integrate_curve(y, x)
        self.assertAlmostEqual(itgt, 17.5)

        itgt = integrate_curve(y, x, (1, 3))
        self.assertAlmostEqual(itgt, 8)

        y = np.array([1, -1, 1, -1, 1, -1])
        x = np.array([0, 1, 2, 3, 4, 5])
        itgt = integrate_curve(y, x)
        self.assertAlmostEqual(itgt, 1)
