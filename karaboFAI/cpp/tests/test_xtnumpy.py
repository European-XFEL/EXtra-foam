import unittest
import time

import numpy as np

from karaboFAI.cpp import (
    nanmeanImages, xtNanmeanImages, xt_nanmean_two_images, xt_moving_average,
)
from karaboFAI.algorithms import nanmean_images, mask_image


class TestXtnumpy(unittest.TestCase):
    def _nanmean_images_performance(self, data_type):
        data = np.ones((64, 1024, 1024), dtype=data_type)
        data[::2, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        nanmeanImages(data)
        dt_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        nanmeanImages(data, list(range(len(data))))
        dt_cpp_sliced = time.perf_counter() - t0

        t0 = time.perf_counter()
        xtNanmeanImages(data)
        dt_cpp_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        nanmean_images(data)
        dt_py = time.perf_counter() - t0

        print(f"\nnanmean_images with {data_type} - "
              f"dt (cpp para): {dt_cpp:.4f}, dt (cpp para sliced): {dt_cpp_sliced:.4f}, "
              f"dt (cpp xtensor): {dt_cpp_old:.4f}, dt (numpy para): {dt_py:.4f}")

    def testXtNanmeanImages(self):
        # test invalid shapes
        data = np.ones([2, 2])
        with self.assertRaises(RuntimeError):
            nanmeanImages(data)

        data = np.ones([2, 2, 2, 2])
        with self.assertRaises(RuntimeError):
            nanmeanImages(data)

        # test passing empty keep list
        data = np.ones([2, 2, 2])
        with self.assertRaises(ValueError):
            nanmeanImages(data, [])

        # test nanmean on the whole array
        data = np.array([[[np.nan,       2, np.nan], [     1, 2, -np.inf]],
                         [[     1, -np.inf, np.nan], [np.nan, 3,  np.inf]],
                         [[np.inf,       4, np.nan], [     1, 4,      1]]], dtype=np.float32)

        # Note that mean of -np.inf, np.inf and 1 are np.nan!!!
        expected = np.array([[np.inf, -np.inf, np.nan], [  1, 3,  np.nan]], dtype=np.float32)
        np.testing.assert_array_almost_equal(expected, np.nanmean(data, axis=0))
        np.testing.assert_array_almost_equal(expected, nanmeanImages(data))

        # test nanmean on the sliced array
        np.testing.assert_array_almost_equal(np.nanmean(data[[0, 1, 2], ...], axis=0),
                                             nanmeanImages(data, [0, 1, 2]))
        np.testing.assert_array_almost_equal(np.nanmean(data[[1], ...], axis=0),
                                             nanmeanImages(data, [1]))
        np.testing.assert_array_almost_equal(np.nanmean(data[0:3:2, ...], axis=0),
                                             nanmeanImages(data, [0, 2]))

    def testXtNanmeanImagesPerformance(self):
        self._nanmean_images_performance(np.float32)
        self._nanmean_images_performance(np.float64)

    def _nanmean_two_images_performance(self, data_type):
        img = np.ones((1024, 1024), dtype=data_type)
        img[::2, ::2] = np.nan

        t0 = time.perf_counter()
        ret_cpp = xt_nanmean_two_images(img, img)
        dt_cpp = time.perf_counter() - t0

        imgs = np.ones((2, 1024, 1024), dtype=data_type)
        imgs[:, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        ret_py = nanmeanImages(imgs)
        dt_py = time.perf_counter() - t0

        print(f"\nnanmean_two_images with {data_type} - "
              f"dt (cpp): {dt_cpp:.4f}, dt (numpy para): {dt_py:.4f}")

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
