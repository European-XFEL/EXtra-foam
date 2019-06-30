import unittest

import numpy as np

from karaboFAI.algorithms import (
    first_tensor, mask_image, nanmean_image
)


class TestPynumpy(unittest.TestCase):

    def test_nanmeanparaimp(self):
        # test invalid shapes
        data = np.ones([2, 2])
        with self.assertRaises(ValueError):
            nanmean_image(data)

        data = np.ones([2, 2, 2, 2])
        with self.assertRaises(ValueError):
            nanmean_image(data)

        # test 3D array
        data = np.ones([2, 4, 2])
        data[0, 0, 1] = np.nan
        data[1, 0, 1] = np.nan
        data[1, 2, 0] = np.nan
        data[0, 3, 1] = np.inf

        ret = nanmean_image(data, chunk_size=2, max_workers=2)
        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., np.inf]])
        np.testing.assert_array_almost_equal(expected, ret)

        ret = nanmean_image(data, chunk_size=1, max_workers=1)
        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., np.inf]])
        np.testing.assert_array_almost_equal(expected, ret)

    def test_firstTensor(self):
        self.assertEqual(1.0, first_tensor())

    def testMaskImage(self):
        # test not in-place
        img = np.ones((4, 4), dtype=np.float32)
        masked = mask_image(img)
        np.testing.assert_array_equal(img, masked)
        self.assertIsNot(masked, img)

        # test in-place
        masked = mask_image(img, inplace=True)
        self.assertIs(masked, img)

        # test with threshold mask only
        img = np.array([
            [1, 2],
            [3, 4]
        ])
        masked = mask_image(img, threshold_mask=(2, 3))
        np.testing.assert_array_equal(np.array([[0, 2], [3, 0]]), masked)

        # test with threshold mask and image mask
        img = np.array([
            [1, 2],
            [3, 4]
        ])
        image_mask = np.array([
            [0, 1],
            [0, 1]
        ], dtype=np.bool)
        masked = mask_image(img, threshold_mask=(2, 3), image_mask=image_mask)
        np.testing.assert_array_equal(np.array([[0, 0], [3, 0]]), masked)
