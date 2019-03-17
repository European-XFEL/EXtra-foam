import unittest
import tempfile

import numpy as np

from karaboFAI.gui.plot_widgets.image_view import ImageAnalysis
from karaboFAI.pipeline.data_model import ImageData
from karaboFAI.logger import logger
from karaboFAI.config import config

config['PIXEL_SIZE'] = 1e-6
config["MASK_RANGE"] = (None, None)


class TestPlotWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config["PIXEL_SIZE"] = 1e-6
        config["MASK_RANGE"] = (None, None)

        cls._widget = ImageAnalysis(color_map="thermal")

    def testSaveLoadImageMask(self):
        fp = tempfile.TemporaryFile()
        # if image_data is None, it does not raise but only logger.error()
        with self.assertLogs(logger, level="ERROR"):
            self._widget._saveImageMaskImp(fp)

        with self.assertLogs(logger, level="ERROR") as cm:
            self._widget._loadImageMaskImp(fp)
        self.assertEqual(cm.output[0].split(':')[-1],
                         'Cannot load image mask without image!')

        imgs = np.arange(100, dtype=np.float).reshape(10, 10)
        mask = np.zeros_like(imgs, dtype=bool)
        self._widget.setImageData(ImageData(imgs))

        # the IOError
        with self.assertLogs(logger, level="ERROR") as cm:
            self._widget._loadImageMaskImp('abc')
        self.assertEqual(cm.output[0].split(':')[-1],
                         'Cannot load mask from abc')

        self._widget._saveImageMaskImp(fp)

        fp.seek(0)
        self._widget._loadImageMaskImp(fp)
        # the change of 'image_mask' will only reflect on the new instance
        # of ImageData
        np.testing.assert_array_equal(mask,
                                      ImageData(imgs).image_mask)

        # save and load another mask
        mask[0, 0] = 1
        mask[5, 5] = 1
        mask_item = self._widget._mask_item
        mask_item._mask.setPixelColor(0, 0, mask_item._OPAQUE)
        mask_item._mask.setPixelColor(5, 5, mask_item._OPAQUE)
        fp.seek(0)
        self._widget._saveImageMaskImp(fp)
        fp.seek(0)
        self._widget._loadImageMaskImp(fp)

        np.testing.assert_array_equal(mask,
                                      ImageData(imgs).image_mask)

        # load a mask with different shape
        new_mask = np.array((3, 3), dtype=bool)
        fp.seek(0)
        np.save(fp, new_mask)
        fp.seek(0)
        with self.assertLogs(logger, level='ERROR'):
            self._widget._loadImageMaskImp(fp)

        np.testing.assert_array_equal(mask,
                                      ImageData(imgs).image_mask)
