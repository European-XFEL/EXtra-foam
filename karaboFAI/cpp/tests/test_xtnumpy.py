import unittest
import time

import numpy as np

from karaboFAI.cpp import xt_nanmean_images, xt_nanmean_two_images

from karaboFAI.algorithms import nanmean_images


class TestXtnumpy(unittest.TestCase):
    def nanmean_images_compare_cpp_py(self, data_type):
        data = np.ones((64, 1024, 1024), dtype=data_type)
        data[::2, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        ret_cpp = xt_nanmean_images(data)
        dt_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        ret_py = nanmean_images(data)
        dt_py = time.perf_counter() - t0

        print(f"nanmean_images with {data_type} - "
              f"dt (cpp): {dt_cpp:.4f}, dt (numpy_para): {dt_py:.4f}")

    def nanmean_two_images_compare_cpp_py(self, data_type):
        img = np.ones((1024, 1024), dtype=np.float32)
        img[::2, ::2] = np.nan

        t0 = time.perf_counter()
        ret_cpp = xt_nanmean_two_images(img, img)
        dt_cpp = time.perf_counter() - t0

        imgs = np.ones((2, 1024, 1024), dtype=np.float32)
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

        # test nanmean

        data = np.ones([2, 4, 2], dtype=np.float64)
        data[0, 0, 1] = np.nan
        data[1, 0, 1] = np.nan
        data[1, 2, 0] = np.nan
        data[0, 3, 1] = np.inf

        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., np.inf]])
        np.testing.assert_array_almost_equal(expected, xt_nanmean_images(data))

        # test performance
        self.nanmean_images_compare_cpp_py(np.float32)
        self.nanmean_images_compare_cpp_py(np.float64)

    def testXtNanMeanTwoImages(self):
        # test nanmean

        img1 = np.array([[1, 1, 2], [np.inf, np.nan, 0]], dtype=np.float32)
        img2 = np.array([[np.nan, 0, 4], [2, np.nan, -np.inf]], dtype=np.float32)

        expected = np.array([[1., 0.5, 3], [np.inf, 0, -np.inf]])
        np.testing.assert_array_almost_equal(expected, xt_nanmean_two_images(img1, img2))

        # test performance
        self.nanmean_two_images_compare_cpp_py(np.float32)
        self.nanmean_two_images_compare_cpp_py(np.float64)
