import unittest
import time

import numpy as np

from karaboFAI.cpp import (
    xt_nanmean_images, xt_nanmean_two_images, xt_moving_average,
)
from karaboFAI.algorithms import nanmean_images, mask_image

from karaboFAI.cpp.xtnumpy import xt_nanmean_images_old


class TestXtnumpy(unittest.TestCase):
    def _nanmean_images_performance(self, data_type):
        data = np.ones((64, 1024, 1024), dtype=data_type)
        data[::2, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        ret_cpp = xt_nanmean_images(data)
        dt_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        ret_cpp = xt_nanmean_images_old(data)
        dt_cpp_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        ret_py = nanmean_images(data)
        dt_py = time.perf_counter() - t0

        print(f"nanmean_images with {data_type} - "
              f"dt (cpp): {dt_cpp:.4f}, dt (cpp) old: {dt_cpp_old:.4f}, dt (numpy_para): {dt_py:.4f}")

    def _nanmean_two_images_performance(self, data_type):
        img = np.ones((1024, 1024), dtype=data_type)
        img[::2, ::2] = np.nan

        t0 = time.perf_counter()
        ret_cpp = xt_nanmean_two_images(img, img)
        dt_cpp = time.perf_counter() - t0

        imgs = np.ones((2, 1024, 1024), dtype=data_type)
        imgs[:, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        ret_py = nanmean_images(imgs)
        dt_py = time.perf_counter() - t0

        print(f"nanmean_two_images with {data_type} - "
              f"dt (cpp): {dt_cpp:.4f}, dt (numpy_para): {dt_py:.4f}")

    def testXtNanmeanImage(self):
        # test invalid shapes
        data = np.ones([2, 2])
        with self.assertRaises(ValueError):
            nanmean_images(data)

        data = np.ones([2, 2, 2, 2])
        with self.assertRaises(ValueError):
            nanmean_images(data)

        # test the correctness
        data = np.array([[[np.nan,       2, np.nan], [     1, 2, -np.inf]],
                         [[     1, -np.inf, np.nan], [np.nan, 3,  np.inf]],
                         [[np.inf,       4, np.nan], [     1, 4,      1]]], dtype=np.float32)

        # Note that mean of -np.inf, np.inf and 1 are np.nan!!!
        expected = np.array([[np.inf, -np.inf, np.nan], [  1, 3,  np.nan]], dtype=np.float32)
        np.testing.assert_array_almost_equal(expected, np.nanmean(data, axis=0))
        np.testing.assert_array_almost_equal(expected, xt_nanmean_images(data))

        # test performance
        self._nanmean_images_performance(np.float32)
        self._nanmean_images_performance(np.float64)

    def testXtNanMeanTwoImages(self):
        # test nanmean

        img1 = np.array([[1, 1, 2], [np.inf, np.nan, 0]], dtype=np.float32)
        img2 = np.array([[np.nan, 0, 4], [2, np.nan, -np.inf]], dtype=np.float32)

        expected = np.array([[1., 0.5, 3], [np.inf, 0, -np.inf]])
        np.testing.assert_array_almost_equal(expected, xt_nanmean_two_images(img1, img2))

        # test performance
        self._nanmean_two_images_performance(np.float32)
        self._nanmean_two_images_performance(np.float64)

    def testMovingAverage(self):
        arr = np.ones(100, dtype=np.float32)
        ma = arr.copy()
        data = 3 * arr

        ma = xt_moving_average(ma, data, 2)

        np.testing.assert_array_equal(2 * arr, ma)
