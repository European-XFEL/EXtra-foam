import unittest

import numpy as np

from karaboFAI.algorithms import (
    first_tensor, mask_by_threshold, nanmean_axis0_para
)


class TestPynumpy(unittest.TestCase):

    def test_nanmeanparaimp(self):
        # test 2D array
        data = np.ones([2, 2])
        ret = nanmean_axis0_para(data)
        self.assertIs(ret, data)

        # test 3D array
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

    def test_firstTensor(self):
        self.assertEqual(1.0, first_tensor())

    def testMaskByThreshold(self):
        # test 2D array
        imgs = np.ones((4, 4), dtype=np.float32)
        masked = mask_by_threshold(imgs)
        np.testing.assert_array_equal(imgs, masked)
        self.assertIsNot(masked, imgs)

        # test 2D array inplace
        masked = mask_by_threshold(imgs, inplace=True)
        self.assertIs(masked, imgs)

        # test 3D array
        imgs = np.ones((2, 4, 4), dtype=np.float32)
        masked = mask_by_threshold(imgs)
        np.testing.assert_array_equal(imgs, masked)
