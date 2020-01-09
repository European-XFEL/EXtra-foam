import unittest

import numpy as np

from extra_foam.algorithms import (
    movingAverageImage, movingAverageImageArray,
    mask_image, mask_image_array,
    correct_image_data, nanmean_image_data
)


class TestImageProc(unittest.TestCase):
    def testNanmeanImageData(self):
        # test invalid shapes
        with self.assertRaises(TypeError):
            nanmean_image_data(np.ones([2, 2, 2, 2]))
        with self.assertRaises(TypeError):
            nanmean_image_data(np.ones([2]))

        # test two images have different shapes
        with self.assertRaises(ValueError):
            nanmean_image_data([np.ones((2, 2)), np.ones((2, 3))])

        # kept is an empty list
        data = np.ones([2, 2, 2])
        with self.assertRaises(ValueError):
            nanmean_image_data((data, []))

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

        # test invalid input
        with self.assertRaises(TypeError):
            movingAverageImage()
        # test incorrect shape
        with self.assertRaises(TypeError):
            movingAverageImage(arr1d, arr1d, 2)
        with self.assertRaises(TypeError):
            movingAverageImage(arr3d, arr3d, 2)
        with self.assertRaises(ValueError):
            # 0 count
            movingAverageImage(arr2d, arr2d, 0)
        with self.assertRaises(ValueError):
            # inconsistent shape
            movingAverageImage(arr2d, np.ones((2, 3), dtype=np.float32), 2)

        # test invalid input
        with self.assertRaises(TypeError):
            movingAverageImageArray()
        with self.assertRaises(TypeError):
            movingAverageImageArray(arr1d, arr1d, 2)
        with self.assertRaises(TypeError):
            movingAverageImageArray(arr2d, arr2d, 2)
        with self.assertRaises(ValueError):
            # 0 count
            movingAverageImageArray(arr3d, arr3d, 0)
        with self.assertRaises(ValueError):
            # inconsistent shape
            movingAverageImageArray(arr3d, np.ones((2, 3, 2), dtype=np.float32), 2)

        # ------------
        # single image
        # ------------

        img1 = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)
        img2 = np.array([[2, 3, 4], [4, 5, 6]], dtype=np.float32)
        movingAverageImage(img1, img2, 2)
        ma_gt = np.array([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, img1)

        # ------------
        # train images
        # ------------

        imgs1 = np.array([[[1, 2, 3], [3, 4, 5]],
                          [[1, 2, 3], [3, 4, 5]]], dtype=np.float32)
        imgs2 = np.array([[[2, 3, 4], [4, 5, 6]],
                          [[2, 3, 4], [4, 5, 6]]], dtype=np.float32)
        movingAverageImageArray(imgs1, imgs2, 2)
        ma_gt = np.array([[[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]],
                         [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, imgs1)

    def testMovingAverageWithNan(self):
        # ------------
        # single image
        # ------------

        img1 = np.array([[1, np.nan, 3], [np.nan, 4, 5]], dtype=np.float32)
        img2 = np.array([[2,      3, 4], [np.nan, 5, 6]], dtype=np.float32)
        movingAverageImage(img1, img2, 2)
        ma_gt = np.array([[1.5, np.nan, 3.5], [np.nan, 4.5, 5.5]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, img1)

        # ------------
        # train images
        # ------------

        imgs1 = np.array([[[1, np.nan, 3], [np.nan, 4, 5]],
                          [[1,      2, 3], [np.nan, 4, 5]]], dtype=np.float32)
        imgs2 = np.array([[[2,      3, 4], [     4, 5, 6]],
                          [[2,      3, 4], [     4, 5, 6]]], dtype=np.float32)
        movingAverageImageArray(imgs1, imgs2, 2)
        ma_gt = np.array([[[1.5, np.nan, 3.5], [np.nan, 4.5, 5.5]],
                          [[1.5,    2.5, 3.5], [np.nan, 4.5, 5.5]]], dtype=np.float32)

        np.testing.assert_array_equal(ma_gt, imgs1)

    def testMaskImage(self):
        # test invalid input
        with self.assertRaises(TypeError):
            mask_image()
        # test incorrect shape
        with self.assertRaises(TypeError):
            mask_image(np.ones((2, 2, 2)), 1, 2)
        with self.assertRaises(TypeError):
            mask_image(np.ones(2), 1, 2)

        # test invalid input
        with self.assertRaises(TypeError):
            mask_image_array()
        with self.assertRaises(TypeError):
            mask_image_array(np.ones((2, 2)), 1, 2)
        with self.assertRaises(TypeError):
            mask_image_array(np.ones(2), 1, 2)

        # ------------
        # single image
        # ------------

        # threshold mask
        img = np.array([[1, 2, np.nan], [3, 4, 5]], dtype=np.float32)
        mask_image(img, threshold_mask=(2, 3))
        np.testing.assert_array_equal(
            np.array([[0, 2, 0], [3, 0, 0]], dtype=np.float32), img)

        # image mask
        img = np.array([[1, np.nan, np.nan], [3, 4, 5]], dtype=np.float32)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        mask_image(img, image_mask=img_mask)
        np.testing.assert_array_equal(
            np.array([[0, 0, 0], [0, 4, 0]], dtype=np.float32), img)

        # ------------
        # train images
        # ------------

        # threshold mask
        img = np.array([[[1, 2, 3], [3, np.nan, 5]],
                        [[1, 2, 3], [3, np.nan, 5]]], dtype=np.float32)
        mask_image_array(img, threshold_mask=(2, 3))
        np.testing.assert_array_equal(np.array([[[0, 2, 3], [3, 0, 0]],
                                                [[0, 2, 3], [3, 0, 0]]],
                                               dtype=np.float32), img)

        # image mask
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=np.float32)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        np.array([[1, 1, 0], [1, 0, 1]], dtype=np.bool)
        mask_image_array(img, image_mask=img_mask)
        np.testing.assert_array_equal(np.array([[[0, 0, 3], [0, 0, 0]],
                                                [[0, 0, 3], [0, 0, 0]]],
                                               dtype=np.float32), img)

    def testCorrectImageData(self):
        # test invalid input
        with self.assertRaises(TypeError):
            correct_image_data()
        with self.assertRaises(TypeError):
            correct_image_data(np.ones(2), offset=np.ones(2))
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2, 2, 2)), gain=np.ones((2, 2, 2, 2)))

        # test incorrect shape
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2, 2)), offset=np.ones([2, 2]))
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2, 2)), gain=np.ones([2, 2]))
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2)), offset=np.ones([2, 2, 2]))
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2)), gain=np.ones([2, 2, 2]))
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2)), gain=np.ones(2, 2), offset=np.ones([2, 2, 2]))

        # test incorrect dtype
        with self.assertRaises(TypeError):
            correct_image_data(np.ones((2, 2, 2), dtype=np.float64),
                               offset=np.ones((2, 2, 2), dtype=np.float32))
            correct_image_data(np.ones((2, 2, 2), dtype=np.float32),
                               gain=np.ones((2, 2, 2), dtype=np.float64))
            correct_image_data(np.ones((2, 2, 2), dtype=np.float32),
                               gain=np.ones((2, 2, 2), dtype=np.float32),
                               offset=np.ones((2, 2, 2), dtype=np.float64))

        # test without gain and offset
        for img in [np.ones([2, 2]), np.ones([2, 2, 2])]:
            img_gt = img.copy()
            correct_image_data(img)
            np.testing.assert_array_equal(img_gt, img)

        # ------------
        # single image
        # ------------

        img = np.array([[1, 2, 3], [3, np.nan, np.nan]], dtype=np.float32)
        offset = np.array([[1, 2, 1], [2, np.nan, np.nan]], dtype=np.float32)
        correct_image_data(img, offset=offset)
        np.testing.assert_array_equal(
            np.array([[0, 0, 2], [1, np.nan, np.nan]], dtype=np.float32), img)
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

        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=np.float32)
        offset = np.array([[[1, 2, 1], [3, np.nan, np.nan]],
                           [[2, 1, 2], [2, np.nan, np.nan]]], dtype=np.float32)
        correct_image_data(img, offset=offset)
        np.testing.assert_array_equal(np.array([[[0, 0, 2], [0, np.nan, np.nan]],
                                                [[-1, 1, 1], [1, np.nan, np.nan]]],
                                               dtype=np.float32), img)
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
