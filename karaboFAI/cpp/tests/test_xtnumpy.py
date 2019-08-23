import unittest
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from karaboFAI.cpp import (
    nanmeanTrain, nanmeanTwo, xtNanmeanTrain,
    movingAveragePulse, movingAverageTrain,
    nanToZeroPulse, nanToZeroTrain,
    maskPulse, maskTrain, xtMaskTrain
)


def nanmean_images_para(data, *, chunk_size=10, max_workers=4):
    """Calculate nanmean of an array of images.

    This function is only used for benchmark.

    :param numpy.ndarray data: an array of images. (index, y, x).
    :param int chunk_size: the slice size of along the second dimension
        of the input data.
    :param int max_workers: The maximum number of threads that can be
        used to execute the given calls.

    :return numpy.ndarray: averaged input data along the first axis if
        the dimension of input data is larger than 3, otherwise the
        original data.
    """
    def nanmean_imp(out, start, end):
        """Implementation of parallelized nanmean.

        :param numpy.ndarray out: result 2D array. (y, x)
        :param int start: start index
        :param int end: end index (not included)
        """
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', category=RuntimeWarning)

            out[start:end, :] = np.nanmean(data[:, start:end, :], axis=0)

    if data.ndim != 3:
        raise ValueError("Input must be a three dimensional numpy.array!")

    ret = np.zeros_like(data[0, ...])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        start = 0
        max_idx = data.shape[1]
        while start < max_idx:
            executor.submit(nanmean_imp, ret, start,
                            min(start + chunk_size, max_idx))
            start += chunk_size

    return ret


class TestPynumpy(unittest.TestCase):
    def test_nanmeanparaimp(self):
        # test invalid shapes
        data = np.ones([2, 2])
        with self.assertRaises(ValueError):
            nanmean_images_para(data)

        data = np.ones([2, 2, 2, 2])
        with self.assertRaises(ValueError):
            nanmean_images_para(data)

        # test 3D array
        data = np.ones([2, 4, 2])
        data[0, 0, 1] = np.nan
        data[1, 0, 1] = np.nan
        data[1, 2, 0] = np.nan
        data[0, 3, 1] = np.inf

        ret = nanmean_images_para(data, chunk_size=2, max_workers=2)
        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., np.inf]])
        np.testing.assert_array_almost_equal(expected, ret)

        ret = nanmean_images_para(data, chunk_size=1, max_workers=1)
        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., np.inf]])
        np.testing.assert_array_almost_equal(expected, ret)


class TestXtnumpy(unittest.TestCase):
    def testNanmeanImages(self):
        # test invalid shapes
        data = np.ones([2, 2])
        with self.assertRaises(RuntimeError):
            nanmeanTrain(data)

        with self.assertRaises(RuntimeError):
            nanmeanTrain(data, data)

        data = np.ones([2, 2, 2, 2])
        with self.assertRaises(RuntimeError):
            nanmeanTrain(data)

        # test passing empty keep list
        data = np.ones([2, 2, 2])
        with self.assertRaises(ValueError):
            nanmeanTrain(data, [])

        data = np.array([[[np.nan,       2, np.nan], [     1, 2, -np.inf]],
                         [[     1, -np.inf, np.nan], [np.nan, 3,  np.inf]],
                         [[np.inf,       4, np.nan], [     1, 4,      1]]], dtype=np.float32)

        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore", category=RuntimeWarning)

            # Note that mean of -np.inf, np.inf and 1 are np.nan!!!
            expected = np.array([[np.inf, -np.inf, np.nan], [  1, 3,  np.nan]], dtype=np.float32)
            np.testing.assert_array_almost_equal(expected, np.nanmean(data, axis=0))
            np.testing.assert_array_almost_equal(expected, nanmeanTrain(data))
            np.testing.assert_array_almost_equal(expected, xtNanmeanTrain(data))

            # test nanmean on the sliced array
            np.testing.assert_array_almost_equal(np.nanmean(data[0:3, ...], axis=0),
                                                 nanmeanTrain(data, [0, 1, 2]))
            np.testing.assert_array_almost_equal(np.nanmean(data[1:2, ...], axis=0),
                                                 nanmeanTrain(data, [1]))
            np.testing.assert_array_almost_equal(np.nanmean(data[0:3:2, ...], axis=0),
                                                 nanmeanTrain(data, [0, 2]))

    def _nanmean_images_performance(self, data_type):
        data = np.ones((64, 1024, 512), dtype=data_type)
        data[::2, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        nanmeanTrain(data)
        dt_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        nanmeanTrain(data, list(range(len(data))))
        dt_cpp_sliced = time.perf_counter() - t0

        t0 = time.perf_counter()
        xtNanmeanTrain(data)
        dt_cpp_xt = time.perf_counter() - t0

        t0 = time.perf_counter()
        nanmean_images_para(data)
        dt_py = time.perf_counter() - t0

        print(f"\nnanmeanTrain with {data_type} - "
              f"dt (cpp para): {dt_cpp:.4f}, dt (cpp para sliced): {dt_cpp_sliced:.4f}, "
              f"dt (cpp xtensor): {dt_cpp_xt:.4f}, dt (numpy para): {dt_py:.4f}")

    @unittest.skipIf(os.environ.get("FAI_WITH_TBB", '1') == '0', "TBB only")
    def testNanmeanImagesPerformance(self):
        self._nanmean_images_performance(np.float32)
        self._nanmean_images_performance(np.float64)

    def testNanmeanWithTwoImages(self):
        with self.assertRaises(ValueError):
            nanmeanTwo(np.ones((2, 2)), np.ones((2, 3)))

        img1 = np.array([[1, 1, 2], [np.inf, np.nan, 0]], dtype=np.float32)
        img2 = np.array([[np.nan, 0, 4], [2, np.nan, -np.inf]], dtype=np.float32)

        expected = np.array([[1., 0.5, 3], [np.inf, np.nan, -np.inf]])
        np.testing.assert_array_almost_equal(expected, nanmeanTwo(img1, img2))

    def _nanmean_two_images_performance(self, data_type):
        img = np.ones((1024, 512), dtype=data_type)
        img[::2, ::2] = np.nan

        t0 = time.perf_counter()
        nanmeanTwo(img, img)
        dt_cpp = time.perf_counter() - t0

        imgs = np.ones((2, 1024, 512), dtype=data_type)
        imgs[:, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        nanmeanTrain(imgs)
        dt_cpp_2 = time.perf_counter() - t0

        print(f"\nnanmeanTwo with {data_type} - "
              f"dt (cpp para): {dt_cpp:.4f}, dt (cpp para2): {dt_cpp_2:.4f}")

    @unittest.skipIf(os.environ.get("FAI_WITH_TBB", '1') == '0', "TBB only")
    def testNanmeanWithTwoImagesPerformance(self):
        self._nanmean_two_images_performance(np.float32)
        self._nanmean_two_images_performance(np.float64)

    def testMovingAverage(self):
        arr1d = np.ones(2, dtype=np.float32)
        arr2d = np.ones((2, 2), dtype=np.float32)
        arr3d = np.ones((2, 2, 2), dtype=np.float32)

        # test invalid input
        with self.assertRaises(TypeError):
            movingAveragePulse()
        # test incorrect shape
        with self.assertRaises(RuntimeError):
            movingAveragePulse(arr1d, arr1d, 2)
        with self.assertRaises(RuntimeError):
            movingAveragePulse(arr3d, arr3d, 2)
        with self.assertRaises(ValueError):
            # 0 count
            movingAveragePulse(arr2d, arr2d, 0)
        with self.assertRaises(ValueError):
            # inconsistent shape
            movingAveragePulse(arr2d, np.ones((2, 3), dtype=np.float32), 2)

        # test invalid input
        with self.assertRaises(TypeError):
            movingAverageTrain()
        with self.assertRaises(RuntimeError):
            movingAverageTrain(arr1d, arr1d, 2)
        with self.assertRaises(RuntimeError):
            movingAverageTrain(arr2d, arr2d, 2)
        with self.assertRaises(ValueError):
            # 0 count
            movingAverageTrain(arr3d, arr3d, 0)
        with self.assertRaises(ValueError):
            # inconsistent shape
            movingAverageTrain(arr3d, np.ones((2, 3, 2), dtype=np.float32), 2)

        # ------------
        # single image
        # ------------

        img1 = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)
        img2 = np.array([[2, 3, 4], [4, 5, 6]], dtype=np.float32)
        movingAveragePulse(img1, img2, 2)
        ma_gt = np.array([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, img1)

        # ------------
        # train images
        # ------------

        imgs1 = np.array([[[1, 2, 3], [3, 4, 5]],
                          [[1, 2, 3], [3, 4, 5]]], dtype=np.float32)
        imgs2 = np.array([[[2, 3, 4], [4, 5, 6]],
                          [[2, 3, 4], [4, 5, 6]]], dtype=np.float32)
        movingAverageTrain(imgs1, imgs2, 2)
        ma_gt = np.array([[[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]],
                         [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, imgs1)

    def testMovingAverageWithNan(self):
        # ------------
        # single image
        # ------------

        img1 = np.array([[1, np.nan, 3], [np.nan, 4, 5]], dtype=np.float32)
        img2 = np.array([[2,      3, 4], [np.nan, 5, 6]], dtype=np.float32)
        movingAveragePulse(img1, img2, 2)
        ma_gt = np.array([[1.5, np.nan, 3.5], [np.nan, 4.5, 5.5]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, img1)

        # ------------
        # train images
        # ------------

        imgs1 = np.array([[[1, np.nan, 3], [np.nan, 4, 5]],
                          [[1,      2, 3], [np.nan, 4, 5]]], dtype=np.float32)
        imgs2 = np.array([[[2,      3, 4], [     4, 5, 6]],
                          [[2,      3, 4], [     4, 5, 6]]], dtype=np.float32)
        movingAverageTrain(imgs1, imgs2, 2)
        ma_gt = np.array([[[1.5, np.nan, 3.5], [np.nan, 4.5, 5.5]],
                          [[1.5,    2.5, 3.5], [np.nan, 4.5, 5.5]]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, imgs1)

    def _moving_average_performance(self, data_type):
        imgs = np.ones((64, 1024, 512), dtype=data_type)

        t0 = time.perf_counter()
        movingAverageTrain(imgs, imgs, 5)
        dt_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        imgs + (imgs - imgs) / 5
        dt_py = time.perf_counter() - t0

        print(f"\nmoving average with {data_type} - "
              f"dt (cpp para): {dt_cpp:.4f}, dt (numpy): {dt_py:.4f}")

    @unittest.skipIf(os.environ.get("FAI_WITH_TBB", '1') == '0', "TBB only")
    def testMovingAveragePerformance(self):
        self._moving_average_performance(np.float32)
        self._moving_average_performance(np.float64)

    def testMaskImage(self):
        # test invalid input
        with self.assertRaises(TypeError):
            maskPulse()
        # test incorrect shape
        with self.assertRaises(RuntimeError):
            maskPulse(np.ones((2, 2, 2)), 1, 2)
        with self.assertRaises(RuntimeError):
            maskPulse(np.ones(2), 1, 2)

        # test invalid input
        with self.assertRaises(TypeError):
            maskTrain()
        with self.assertRaises(RuntimeError):
            maskTrain(np.ones((2, 2)), 1, 2)
        with self.assertRaises(RuntimeError):
            maskTrain(np.ones(2), 1, 2)

        # ------------
        # single image
        # ------------

        # threshold mask
        img = np.array([[1, 2, np.nan], [3, 4, 5]], dtype=np.float32)
        maskPulse(img, 2, 3)
        np.testing.assert_array_equal(
            np.array([[0, 2, np.nan], [3, 0, 0]], dtype=np.float32), img)

        # image mask
        img = np.array([[1, np.nan, np.nan], [3, 4, 5]], dtype=np.float32)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        maskPulse(img, img_mask)
        np.testing.assert_array_equal(
            np.array([[0, 0, np.nan], [0, 4, 0]], dtype=np.float32), img)

        # ------------
        # train images
        # ------------

        # threshold mask
        img = np.array([[[1, 2, 3], [3, np.nan, 5]],
                        [[1, 2, 3], [3, np.nan, 5]]], dtype=np.float32)
        maskTrain(img, 2, 3)
        np.testing.assert_array_equal(np.array([[[0, 2, 3], [3, np.nan, 0]],
                                                [[0, 2, 3], [3, np.nan, 0]]],
                                               dtype=np.float32), img)

        # image mask
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=np.float32)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        maskTrain(img, img_mask)
        np.testing.assert_array_equal(np.array([[[0, 0, 3], [0, np.nan, 0]],
                                                [[0, 0, 3], [0, np.nan, 0]]],
                                               dtype=np.float32), img)

    def _mask_image_performance(self, data_type):
        # mask by threshold
        data = np.ones((64, 1024, 512), dtype=data_type)
        t0 = time.perf_counter()
        maskTrain(data, 2., 3.)  # every elements are masked
        dt_cpp_th = time.perf_counter() - t0

        data = np.ones((64, 1024, 512), dtype=data_type)
        t0 = time.perf_counter()
        xtMaskTrain(data, 2., 3.)
        dt_cpp_xt = time.perf_counter() - t0

        data = np.ones((64, 1024, 512), dtype=data_type)
        t0 = time.perf_counter()
        data[(data > 3) | (data < 2)] = 0
        dt_py_th = time.perf_counter() - t0

        # mask by image
        mask = np.ones((1024, 512), dtype=np.bool)

        data = np.ones((64, 1024, 512), dtype=data_type)
        t0 = time.perf_counter()
        maskTrain(data, mask)
        dt_cpp = time.perf_counter() - t0

        data = np.ones((64, 1024, 512), dtype=data_type)
        t0 = time.perf_counter()
        data[:, mask] = 0
        dt_py = time.perf_counter() - t0

        print(f"\nmaskTrain with {data_type} - \n"
              f"dt (cpp para) threshold: {dt_cpp_th:.4f}, "
              f"dt (cpp xtensor) threshold: {dt_cpp_xt:.4f}, "
              f"dt (numpy) threshold: {dt_py_th:.4f}, \n"
              f"dt (cpp para) image: {dt_cpp:.4f}, dt (numpy) image: {dt_py:.4f}")

    @unittest.skipIf(os.environ.get("FAI_WITH_TBB", '1') == '0', "TBB only")
    def testMaskImagePerformance(self):
        self._mask_image_performance(np.float32)
        self._mask_image_performance(np.float64)

    def _nan2zero_performance(self, data_type):
        # mask by threshold
        data = np.ones((64, 1024, 512), dtype=data_type)
        data[::2, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        nanToZeroTrain(data)
        dt_cpp = time.perf_counter() - t0

        # need a fresh data since number of nans determines the performance
        data = np.ones((64, 1024, 512), dtype=data_type)
        data[::2, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        data[np.isnan(data)] = 0
        dt_py = time.perf_counter() - t0

        print(f"\nnanToZeroTrain with {data_type} - "
              f"dt (cpp para): {dt_cpp:.4f}, dt (numpy): {dt_py:.4f}")

    @unittest.skipIf(os.environ.get("FAI_WITH_TBB", '1') == '0', "TBB only")
    def testNanToZeroPerformance(self):
        self._nan2zero_performance(np.float32)
        self._nan2zero_performance(np.float64)
