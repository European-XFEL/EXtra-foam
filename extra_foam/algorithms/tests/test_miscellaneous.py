import unittest

import math

import numpy as np

from extra_foam.algorithms import (
    compute_statistics, find_actual_range, normalize_auc
)


class TestMiscellaneous(unittest.TestCase):

    def testNormalizeAuc(self):
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        x = np.array([0, 1, 2, 3, 4, 5])

        # default x_min and x_max are both None
        y_normalized = normalize_auc(y, x)
        self.assertTrue(np.array_equal(y_normalized, np.array([0.2]*6)))

        # the following test also ensures that the normalized y does not
        # share memory space with the original y

        # normal case
        y_normalized = normalize_auc(y, x, (1, 3))
        self.assertTrue(np.array_equal(y_normalized, np.array([0.5]*6)))

        # x_min and x_max are -inf/inf
        y_normalized = normalize_auc(y, x, (-np.inf, np.inf))
        self.assertTrue(np.array_equal(y_normalized, np.array([0.2]*6)))

        # AUC is zero
        y = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        x = np.array([0, 1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            normalize_auc(y, x)
        with self.assertRaises(ValueError):
            normalize_auc(y, x, (2, 3))

        # normalize an all-zero curve
        y = np.array([0, 0, 0, 0, 0, 0])
        x = np.array([0, 1, 2, 3, 4, 5])
        y_normalized = normalize_auc(y, x)
        self.assertTrue(np.array_equal(y_normalized, np.array([0]*6)))
        # test data is copied in this case
        y[0] = 1
        self.assertEqual(0, y_normalized[0])

    def testFindActualRange(self):
        arr = np.array([1, 2, 3, 4])
        self.assertTupleEqual((-1.5, 2.5), find_actual_range(arr, (-1.5, 2.5)))

        arr = np.array([1, 2, 3, 4])
        self.assertTupleEqual((1, 4), find_actual_range(arr, (-math.inf, math.inf)))

        arr = np.array([1, 1, 1, 1])
        self.assertTupleEqual((0.5, 1.5), find_actual_range(arr, (-math.inf, math.inf)))

        arr = np.array([1, 2, 3, 4])
        self.assertTupleEqual((3, 4), find_actual_range(arr, (3, math.inf)))

        arr = np.array([1, 2, 3, 4])
        self.assertTupleEqual((4, 5), find_actual_range(arr, (4, math.inf)))

        arr = np.array([1, 2, 3, 4])
        self.assertTupleEqual((5, 6), find_actual_range(arr, (5, math.inf)))

        arr = np.array([1, 2, 3, 4])
        self.assertTupleEqual((0, 1), find_actual_range(arr, (-math.inf, 1)))

        arr = np.array([1, 2, 3, 4])
        self.assertTupleEqual((-1, 0), find_actual_range(arr, (-math.inf, 0)))

        arr = np.array([1, 2, 3, 4])
        self.assertTupleEqual((-1, 0), find_actual_range(arr, (-math.inf, 0)))

    def testComputeStatistics(self):
        arr = np.array([1, 1, 2, 1, 1])
        for v in compute_statistics(arr[arr > 2]):
            self.assertTrue(np.isnan(v))

        self.assertTupleEqual((1.2, 1.0, 0.4), compute_statistics(arr))
