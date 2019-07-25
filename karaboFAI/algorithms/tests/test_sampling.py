import unittest
import math

import numpy as np

from karaboFAI.algorithms import down_sample, slice_curve, up_sample


class TestSampling(unittest.TestCase):
    def test_slicecurve(self):
        y = np.array([6, 5, 4, 3, 2, 1])
        x = np.array([-5.0, -3.0, 2.0, 3.0, 4.0, 5.0])

        # default x_min and x_max are both None
        new_y, new_x = slice_curve(y, x)
        self.assertTrue(np.array_equal(new_y, y))
        self.assertTrue(np.array_equal(new_x, x))

        # normal slice
        new_y, new_x = slice_curve(y, x, 1, 3)
        np.testing.assert_array_equal(np.array([4, 3]), new_y)
        np.testing.assert_array_equal(np.array([2.0, 3.0]), new_x)

        # x_min >= x_max. It returns empty array instead of raising Exception
        new_y, new_x = slice_curve(y, x, 3, 1)
        np.testing.assert_array_equal(np.array([], dtype=y.dtype), new_y)
        np.testing.assert_array_equal(np.array([], dtype=x.dtype), new_x)

        # x_min and x_max are -inf/inf
        new_y, new_x = slice_curve(y, x, -math.inf, math.inf)
        np.testing.assert_array_equal(y, new_y)
        np.testing.assert_array_equal(x, new_x)
        new_y, new_x = slice_curve(y, x, -np.inf, np.inf)
        np.testing.assert_array_equal(y, new_y)
        np.testing.assert_array_equal(x, new_x)

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
