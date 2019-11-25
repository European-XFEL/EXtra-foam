import unittest

import numpy as np

from extra_foam.algorithms import mask_image


class TestPynumpy(unittest.TestCase):

    def testMaskImage(self):

        with self.assertRaises(ValueError):
            mask_image(np.ones((3, 3, 3)))

        with self.assertRaises(ValueError):
            mask_image(np.ones(3))

        # test not in-place
        img = np.ones((4, 4), dtype=np.float32)
        masked = mask_image(img)
        np.testing.assert_array_equal(img, masked)
        self.assertIsNot(masked, img)

        # test in-place
        masked = mask_image(img, inplace=True)
        self.assertIs(masked, img)

        # test with threshold mask only
        img = np.array([[np.nan, np.nan, 3], [4, 5, 6]], dtype=np.float32)
        masked = mask_image(img, threshold_mask=(2, 5))
        np.testing.assert_array_equal(np.array([[0, 0, 3], [4, 5, 0]]), masked)

        # test with image mask only
        img = np.array([[np.nan, np.nan, 3], [4, 5, 6]], dtype=np.float32)
        image_mask = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.bool)
        masked = mask_image(img, image_mask=image_mask)
        np.testing.assert_array_equal(np.array([[0, 0, 3], [4, 0, 0]]), masked)

        # test with threshold mask and image mask
        img = np.array([[np.nan, np.nan, 3], [4, 5, 6]], dtype=np.float32)
        image_mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.bool)
        masked = mask_image(img, threshold_mask=(2, 5), image_mask=image_mask)
        np.testing.assert_array_equal(np.array([[0, 0, 3], [4, 0, 0]]), masked)
