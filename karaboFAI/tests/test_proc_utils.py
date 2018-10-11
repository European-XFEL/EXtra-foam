import unittest

import numpy as np

from karaboFAI.data_processing import (
    down_sample, integrate_curve, sub_array_with_range, up_sample
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

    def test_downsample(self):
        x1 = np.array([1, 2])
        x1_gt = np.array([1])
        self.assertTrue(x1_gt, down_sample(x1))

        x2 = np.array([[1, 1, 2, 2],
                       [1, 1, 2, 2],
                       [3, 3, 4, 4],
                       [3, 3, 4, 4]])
        x2_gt = np.array([[1, 2],
                          [3, 4]])
        self.assertTrue(np.array_equal(x2_gt, down_sample(x2)))
        x22 = np.array([[1, 1, 2],
                        [1, 1, 2],
                        [3, 3, 4]])
        self.assertTrue(np.array_equal(x2_gt, down_sample(x22)))

        x3 = np.array([[[1, 1, 2],
                        [1, 1, 2],
                        [3, 3, 4]],
                       [[1, 1, 2],
                        [1, 1, 2],
                        [3, 3, 4]]])
        x3_gt = np.array([[[1, 2],
                           [3, 4]],
                          [[1, 2],
                           [3, 4]]])
        self.assertTrue(np.array_equal(x3_gt, down_sample(x3)))

        with self.assertRaises(ValueError):
            down_sample(np.arange(16).reshape(2, 2, 2, 2))

    def test_upsample(self):
        x1 = np.array([1, 2])
        x1_gt = np.array([1, 1, 2, 2])
        self.assertTrue(np.array_equal(x1_gt, up_sample(x1, (4,))))

        x2 = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
        with self.assertRaises(ValueError):
            up_sample(x2, (4, 5))
        with self.assertRaises(TypeError):
            up_sample(x2, 4)
        x2_gt1 = np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(x2_gt1, up_sample(x2, (6, 6))))
        x2_gt2 = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(x2_gt2, up_sample(x2, (6, 5))))
        x2_gt3 = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(x2_gt3, up_sample(x2, (5, 5))))

        x3 = np.array([[[1, 2, 3],
                        [4, 5, 6]],
                       [[1, 2, 3],
                        [4, 5, 6]]])
        with self.assertRaises(ValueError):
            up_sample(x3, (5, 6))
        x3_gt1 = np.array([[[1, 1, 2, 2, 3, 3],
                            [1, 1, 2, 2, 3, 3],
                            [4, 4, 5, 5, 6, 6],
                            [4, 4, 5, 5, 6, 6]],
                           [[1, 1, 2, 2, 3, 3],
                            [1, 1, 2, 2, 3, 3],
                            [4, 4, 5, 5, 6, 6],
                            [4, 4, 5, 5, 6, 6]]])
        self.assertTrue(np.array_equal(x3_gt1, up_sample(x3, (2, 4, 6))))
        x3_gt_2 = np.array([[[1, 1, 2, 2, 3],
                             [1, 1, 2, 2, 3],
                             [4, 4, 5, 5, 6]],
                            [[1, 1, 2, 2, 3],
                             [1, 1, 2, 2, 3],
                             [4, 4, 5, 5, 6]]])
        self.assertTrue(np.array_equal(x3_gt_2, up_sample(x3, (2, 3, 5))))

        with self.assertRaises(ValueError):
            up_sample(np.arange(16).reshape(2, 2, 2, 2), (2, 4, 4, 4))
