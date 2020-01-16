import unittest

import numpy as np

from extra_foam.algorithms import (
    correct_image_data, mask_image_data, movingAvgImageData, nanmean_image_data
)


class TestImageProc(unittest.TestCase):
    def testNanmeanImageData(self):
        arr1d = np.ones(2, dtype=np.float32)
        arr2d = np.ones((2, 2), dtype=np.float32)
        arr3d = np.ones((2, 2, 2), dtype=np.float32)
        arr4d = np.ones((2, 2, 2, 2), dtype=np.float32)

        # test invalid shapes
        with self.assertRaises(TypeError):
            nanmean_image_data(arr4d)
        with self.assertRaises(TypeError):
            nanmean_image_data(arr1d)

        # kept is an empty list
        with self.assertRaises(ValueError):
            nanmean_image_data(arr3d, [])

        # test two images have different shapes
        with self.assertRaises(ValueError):
            nanmean_image_data([arr2d, np.ones((2, 3), dtype=np.float32)])

        # test two images have different dtype
        with self.assertRaises(TypeError):
            nanmean_image_data([arr2d, np.ones((2, 3), dtype=np.float64)])

        # input is a 2D array
        data = np.random.randn(2, 2)
        ret = nanmean_image_data(data)
        np.testing.assert_array_equal(data, ret)
        self.assertIsNot(ret, data)

        # input is a 3D array
        data = np.array([[[np.nan,       2, np.nan], [     1, 2, -np.inf]],
                         [[     1, -np.inf, np.nan], [np.nan, 3,  np.inf]],
                         [[np.inf,       4, np.nan], [     1, 4,      1]]], dtype=np.float32)

        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore", category=RuntimeWarning)

            # Note that mean of -np.inf, np.inf and 1 are np.nan!!!
            expected = np.array([[np.inf, -np.inf, np.nan], [  1, 3,  np.nan]], dtype=np.float32)
            np.testing.assert_array_almost_equal(expected, np.nanmean(data, axis=0))
            np.testing.assert_array_almost_equal(expected, nanmean_image_data(data))

            # test nanmean on the sliced array
            np.testing.assert_array_almost_equal(np.nanmean(data[0:3, ...], axis=0),
                                                 nanmean_image_data(data, [0, 1, 2]))
            np.testing.assert_array_almost_equal(np.nanmean(data[1:2, ...], axis=0),
                                                 nanmean_image_data(data, [1]))
            np.testing.assert_array_almost_equal(np.nanmean(data[0:3:2, ...], axis=0),
                                                 nanmean_image_data(data, [0, 2]))

        # input are a list/tuple of two images
        img1 = np.array([[1, 1, 2], [np.inf, np.nan, 0]], dtype=np.float32)
        img2 = np.array([[np.nan, 0, 4], [2, np.nan, -np.inf]], dtype=np.float32)
        expected = np.array([[1., 0.5, 3], [np.inf, np.nan, -np.inf]])
        np.testing.assert_array_almost_equal(expected, nanmean_image_data((img1, img2)))
        np.testing.assert_array_almost_equal(expected, nanmean_image_data([img1, img2]))

    def testMovingAverage(self):
        arr1d = np.ones(2, dtype=np.float32)
        arr2d = np.ones((2, 2), dtype=np.float32)
        arr3d = np.ones((2, 2, 2), dtype=np.float32)
        arr4d = np.ones((2, 2, 2, 2), dtype=np.float32)

        # test invalid input
        with self.assertRaises(TypeError):
            movingAvgImageData()
        with self.assertRaises(TypeError):
            movingAvgImageData(arr1d, arr1d, 2)
        with self.assertRaises(TypeError):
            movingAvgImageData(arr4d, arr4d, 2)

        # count is 0
        with self.assertRaises(ValueError):
            movingAvgImageData(arr2d, arr2d, 0)
        with self.assertRaises(ValueError):
            movingAvgImageData(arr3d, arr3d, 0)

        # inconsistent shape
        with self.assertRaises(TypeError):
            movingAvgImageData(arr2d, arr3d)
        with self.assertRaises(ValueError):
            movingAvgImageData(arr2d, np.ones((2, 3), dtype=np.float32), 2)
        with self.assertRaises(ValueError):
            movingAvgImageData(arr3d, np.ones((2, 3, 2), dtype=np.float32), 2)

        # inconsistent dtype
        with self.assertRaises(TypeError):
            movingAvgImageData(arr2d, np.ones((2, 2), dtype=np.float64), 2)
        with self.assertRaises(TypeError):
            movingAvgImageData(arr3d, np.ones((2, 2, 2), dtype=np.float64), 2)

        # ------------
        # single image
        # ------------

        img1 = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)
        img2 = np.array([[2, 3, 4], [4, 5, 6]], dtype=np.float32)
        movingAvgImageData(img1, img2, 2)
        ma_gt = np.array([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, img1)

        # ------------
        # train images
        # ------------

        imgs1 = np.array([[[1, 2, 3], [3, 4, 5]],
                          [[1, 2, 3], [3, 4, 5]]], dtype=np.float32)
        imgs2 = np.array([[[2, 3, 4], [4, 5, 6]],
                          [[2, 3, 4], [4, 5, 6]]], dtype=np.float32)
        movingAvgImageData(imgs1, imgs2, 2)
        ma_gt = np.array([[[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]],
                         [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, imgs1)

    def testMovingAverageWithNan(self):
        # ------------
        # single image
        # ------------

        img1 = np.array([[1, np.nan, 3], [np.nan, 4, 5]], dtype=np.float32)
        img2 = np.array([[2,      3, 4], [np.nan, 5, 6]], dtype=np.float32)
        movingAvgImageData(img1, img2, 2)
        ma_gt = np.array([[1.5, np.nan, 3.5], [np.nan, 4.5, 5.5]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, img1)

        # ------------
        # train images
        # ------------

        imgs1 = np.array([[[1, np.nan, 3], [np.nan, 4, 5]],
                          [[1,      2, 3], [np.nan, 4, 5]]], dtype=np.float32)
        imgs2 = np.array([[[2,      3, 4], [     4, 5, 6]],
                          [[2,      3, 4], [     4, 5, 6]]], dtype=np.float32)
        movingAvgImageData(imgs1, imgs2, 2)
        ma_gt = np.array([[[1.5, np.nan, 3.5], [np.nan, 4.5, 5.5]],
                          [[1.5,    2.5, 3.5], [np.nan, 4.5, 5.5]]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, imgs1)

    def testMaskImageData(self):
        arr1d = np.ones(2, dtype=np.float32)
        arr2d = np.ones((2, 2), dtype=np.float32)
        arr3d = np.ones((2, 2, 2), dtype=np.float32)
        arr4d = np.ones((2, 2, 2, 2), dtype=np.float32)

        # test invalid input
        with self.assertRaises(TypeError):
            mask_image_data()
        with self.assertRaises(TypeError):
            mask_image_data(arr1d, threshold_mask=(1, 2))
        with self.assertRaises(TypeError):
            mask_image_data(arr4d, threshold_mask=(1, 2))

        # test inconsistent shape
        with self.assertRaises(TypeError):
            mask_image_data(arr2d, image_mask=arr3d, threshold_mask=(1, 2))
        with self.assertRaises(TypeError):
            mask_image_data(arr3d, image_mask=arr2d, threshold_mask=(1, 2))
        with self.assertRaises(TypeError):
            mask_image_data(arr3d, image_mask=arr3d, threshold_mask=(1, 2))
        with self.assertRaises(ValueError):
            mask_image_data(arr3d, image_mask=np.ones((3, 2), dtype=bool))
        with self.assertRaises(ValueError):
            mask_image_data(arr2d, image_mask=np.ones((3, 2), dtype=bool))

        # test inconsistent dtype
        with self.assertRaises(TypeError):
            mask_image_data(arr3d, image_mask=np.ones((2, 2), dtype=int))

        # ------------
        # single image
        # ------------

        # threshold mask
        img = np.array([[1, 2, np.nan], [3, 4, 5]], dtype=np.float32)
        mask_image_data(img, threshold_mask=(2, 3))
        np.testing.assert_array_equal(
            np.array([[0, 2, 0], [3, 0, 0]], dtype=np.float32), img)

        # image mask
        img = np.array([[1, np.nan, np.nan], [3, 4, 5]], dtype=np.float32)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        mask_image_data(img, image_mask=img_mask)
        np.testing.assert_array_equal(
            np.array([[0, 0, 0], [0, 4, 0]], dtype=np.float32), img)

        # ------------
        # train images
        # ------------

        # threshold mask
        img = np.array([[[1, 2, 3], [3, np.nan, 5]],
                        [[1, 2, 3], [3, np.nan, 5]]], dtype=np.float32)
        mask_image_data(img, threshold_mask=(2, 3))
        np.testing.assert_array_equal(np.array([[[0, 2, 3], [3, 0, 0]],
                                                [[0, 2, 3], [3, 0, 0]]],
                                               dtype=np.float32), img)

        # image mask
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=np.float32)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        mask_image_data(img, image_mask=img_mask)
        np.testing.assert_array_equal(np.array([[[0, 0, 3], [0, 0, 0]],
                                                [[0, 0, 3], [0, 0, 0]]],
                                               dtype=np.float32), img)

    def testCorrectImageData(self):
        arr1d = np.ones(2, dtype=np.float32)
        arr2d = np.ones((2, 2), dtype=np.float32)
        arr3d = np.ones((2, 2, 2), dtype=np.float32)
        arr4d = np.ones((2, 2, 2, 2), dtype=np.float32)

        # test invalid input
        with self.assertRaises(TypeError):
            correct_image_data()
        with self.assertRaises(TypeError):
            correct_image_data(arr1d, offset=arr1d)
        with self.assertRaises(TypeError):
            correct_image_data(arr4d, gain=arr4d)

        # test incorrect shape
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2, 2)), offset=arr2d)
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2, 2)), gain=arr2d)
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2)), offset=arr3d)
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2)), gain=arr3d)
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2)), gain=arr2d, offset=arr3d)

        # test incorrect dtype
        with self.assertRaises(TypeError):
            correct_image_data(arr3d, offset=np.ones((2, 2, 2), dtype=np.float64))
        with self.assertRaises(TypeError):
            correct_image_data(arr3d, gain=arr3d, offset=np.ones((2, 2, 2), dtype=np.float64))

        # test without gain and offset
        for img in [np.ones([2, 2]), np.ones([2, 2, 2])]:
            img_gt = img.copy()
            correct_image_data(img)
            np.testing.assert_array_equal(img_gt, img)

        # ------------
        # single image
        # ------------

        # offset only
        img = np.array([[1, 2, 3], [3, np.nan, np.nan]], dtype=np.float32)
        offset = np.array([[1, 2, 1], [2, np.nan, np.nan]], dtype=np.float32)
        correct_image_data(img, offset=offset)
        np.testing.assert_array_equal(
            np.array([[0, 0, 2], [1, np.nan, np.nan]], dtype=np.float32), img)

        # gain only
        gain = np.array([[1, 2, 1], [2, 2, 1]], dtype=np.float32)
        correct_image_data(img, gain=gain)
        np.testing.assert_array_equal(
            np.array([[0, 0, 2], [2, np.nan, np.nan]], dtype=np.float32), img)

        # both gain and offset
        img = np.array([[1, 2, 3], [3, np.nan, np.nan]], dtype=np.float32)
        correct_image_data(img, gain=gain, offset=offset)
        np.testing.assert_array_equal(
            np.array([[0, 0, 2], [2, np.nan, np.nan]], dtype=np.float32), img)

        # ------------
        # train images
        # ------------

        # offset only
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=np.float32)
        offset = np.array([[[1, 2, 1], [3, np.nan, np.nan]],
                           [[2, 1, 2], [2, np.nan, np.nan]]], dtype=np.float32)
        correct_image_data(img, offset=offset)
        np.testing.assert_array_equal(np.array([[[0, 0, 2], [0, np.nan, np.nan]],
                                                [[-1, 1, 1], [1, np.nan, np.nan]]],
                                               dtype=np.float32), img)

        # gain only
        gain = np.array([[[1, 2, 1], [2, 2, 1]],
                         [[2, 1, 2], [2, 1, 2]]], dtype=np.float32)
        correct_image_data(img, gain=gain)
        np.testing.assert_array_equal(np.array([[[0, 0, 2], [0, np.nan, np.nan]],
                                                [[-2, 1, 2], [2, np.nan, np.nan]]],
                                               dtype=np.float32), img)

        # both gain and offset
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=np.float32)
        correct_image_data(img, gain=gain, offset=offset)
        np.testing.assert_array_equal(np.array([[[0, 0, 2], [0, np.nan, np.nan]],
                                                [[-2, 1, 2], [2, np.nan, np.nan]]],
                                               dtype=np.float32), img)
