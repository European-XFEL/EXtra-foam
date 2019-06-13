import unittest

import numpy as np

from karaboFAI.pipeline.processors.image_processor import (
    _RawImageData, PumpProbeImageProcessor
)


class TestRawImageData(unittest.TestCase):

    def testInvalidInput(self):
        with self.assertRaises(TypeError):
            _RawImageData([1, 2, 3])

        with self.assertRaises(ValueError):
            _RawImageData(np.arange(2))

        with self.assertRaises(ValueError):
            _RawImageData(np.arange(16).reshape(2, 2, 2, 2))

    def test_trainresolved_ma(self):
        """Test the case with moving average of image."""
        imgs_orig = np.arange(16, dtype=np.float).reshape(4, 4)

        img_data = _RawImageData()
        self.assertEqual(0, img_data.n_images)
        img_data.images = np.copy(imgs_orig)
        self.assertEqual(1, img_data.n_images)

        img_data.ma_window = 3
        img_data.images = imgs_orig - 2
        self.assertEqual(2, img_data.ma_count)
        self.assertEqual(1, img_data.n_images)

        img_data.images = imgs_orig + 2
        self.assertEqual(3, img_data.ma_count)

        img_data.images = imgs_orig
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig, img_data.images)

        # test moving average window size change
        img_data.ma_window = 4
        img_data.images = imgs_orig - 4
        self.assertEqual(4, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig - 1, img_data.images)

        img_data.ma_window = 2
        self.assertEqual(2, img_data.ma_window)
        self.assertEqual(0, img_data.ma_count)
        self.assertIsNone(img_data.images)

        img_data.clear()
        self.assertEqual(1, img_data.ma_window)

    def test_pulseresolved_ma(self):
        imgs_orig = np.arange(32, dtype=np.float).reshape((2, 4, 4))

        # test if images' shapes are different
        img_data = _RawImageData(np.copy(imgs_orig[0, ...]))
        img_data.ma_window = 3
        img_data.images = np.copy(imgs_orig)
        self.assertEqual(1, img_data.ma_count)

        img_data.images = imgs_orig - 2
        self.assertEqual(2, img_data.ma_count)

        img_data.images = imgs_orig + 2
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig, img_data.images)

        img_data.images = imgs_orig + 3
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig+1, img_data.images)

        img_data.clear()
        self.assertEqual(1, img_data.ma_window)
        self.assertEqual(0, img_data.ma_count)
        self.assertIsNone(img_data.images)


class TestPumpProbeImageProcessor(unittest.TestCase):
    def setUp(self):
        self._proc = PumpProbeImageProcessor()
