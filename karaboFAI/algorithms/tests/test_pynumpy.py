import unittest

import numpy as np

from karaboFAI.algorithms import mask_image


class TestPynumpy(unittest.TestCase):

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
