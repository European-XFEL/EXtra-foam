import unittest
from unittest.mock import patch

import numpy as np

from extra_foam.algorithms import mask_image
from extra_foam.config import config
from extra_foam.gui.image_tool.simple_image_data import _SimpleImageData


class TestSimpleImageData(unittest.TestCase):
    @patch.dict(config._data, {"PIXEL_SIZE": 1e-3})
    def testGeneral(self):
        with self.assertRaises(TypeError):
            _SimpleImageData([1, 2, 3])

        gt_data = np.random.randn(3, 3).astype(np.float32)
        img_data = _SimpleImageData.from_array(gt_data)

        img_data.threshold_mask = (3, 6)
        masked_gt = gt_data.copy()
        mask_image(masked_gt, threshold_mask=(3, 6))
        np.testing.assert_array_almost_equal(masked_gt, img_data.masked)

        self.assertEqual(1.0e-3, img_data.pixel_size)

    @patch.dict(config._data, {"PIXEL_SIZE": 1e-3})
    def testInstantiateFromArray(self):
        gt_data = np.ones((2, 2, 2))

        image_data = _SimpleImageData.from_array(gt_data)

        np.testing.assert_array_equal(np.ones((2, 2)), image_data.masked)
        self.assertEqual(1e-3, image_data.pixel_size)
        self.assertEqual(None, image_data.threshold_mask)
