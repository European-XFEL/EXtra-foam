import unittest
import time

import numpy as np

from karaboFAI.data_processing import (
    down_sample, nanmean_axis0_para, normalize_curve, quick_min_max,
    slice_curve, up_sample
)


class TestDataProcessor(unittest.TestCase):
    def test_slicecurve(self):
        y = np.array([6, 5, 4, 3, 2, 1])
        x = np.array([0, 1, 2, 3, 4, 5])

        new_y, new_x = slice_curve(y, x)
        self.assertTrue(np.array_equal(new_y, y))
        self.assertTrue(np.array_equal(new_x, x))

        new_y, new_x = slice_curve(y, x, 1, 3)
        self.assertTrue(np.array_equal(new_y, np.array([5, 4, 3])))
        self.assertTrue(np.array_equal(new_x, np.array([1, 2, 3])))

    def test_integratecurve(self):
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

    def test_nanmeanparaimp(self):
        data = np.ones([2, 4, 2])
        data[0, 0, 1] = np.nan
        data[1, 0, 1] = np.nan
        data[1, 2, 0] = np.nan

        ret = nanmean_axis0_para(data, chunk_size=2, max_workers=2)

        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., 1.]])
        np.testing.assert_array_almost_equal(expected, ret)

        ret = nanmean_axis0_para(data, chunk_size=1, max_workers=1)

        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., 1.]])
        np.testing.assert_array_almost_equal(expected, ret)

    def test_quickminmax(self):
        with self.assertRaises(ValueError):
            quick_min_max(np.arange(8).reshape(2, 2, 2))
        with self.assertRaises(TypeError):
            quick_min_max([])

        self.assertEqual((0, 9), quick_min_max(np.arange(10).reshape(2, 5)))

        x = np.arange(1e6).reshape(1000, 1000)
        self.assertEqual((0, 996996), quick_min_max(x))
