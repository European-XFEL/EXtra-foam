import unittest
from unittest.mock import MagicMock
import tempfile

import numpy as np

from karaboFAI.gui import mkQApp, pyqtgraph
from karaboFAI.gui.windows.image_tool import _SimpleImageData
from karaboFAI.gui.plot_widgets.plot_items import ImageItem, MaskItem
from karaboFAI.gui.plot_widgets.roi import RectROI
from karaboFAI.gui.plot_widgets.image_view import ImageView, ImageAnalysis
from karaboFAI.logger import logger

app = mkQApp()

logger.setLevel("CRITICAL")


class TestImageView(unittest.TestCase):
    def testGeneral(self):
        widget = ImageView()
        plot_items = widget._plot_widget.plotItem.items
        self.assertIsInstance(plot_items[0], pyqtgraph.ImageItem)
        self.assertIsInstance(plot_items[1], MaskItem)
        for i in range(2, 6):
            self.assertIsInstance(plot_items[i], RectROI)

        widget = ImageView(has_mask=False)
        self.assertIsInstance(widget._plot_widget.plotItem.items[1], RectROI)

        widget = ImageView(has_mask=False, has_roi=False)
        self.assertEqual(1, len(widget._plot_widget.plotItem.items))


class TestImageAnalysis(unittest.TestCase):
    def testGeneral(self):
        widget = ImageAnalysis()
        plot_items = widget._plot_widget.plotItem.items
        self.assertIsInstance(plot_items[0], ImageItem)
        self.assertIsInstance(plot_items[1], MaskItem)
        for i in range(2, 6):
            self.assertIsInstance(plot_items[i], RectROI)

    def testSaveLoadImageMask(self):
        widget = ImageAnalysis()

        fp = tempfile.TemporaryFile()
        # if image_data is None, it does not raise but only logger.error()
        with self.assertLogs(logger, level="ERROR"):
            widget._saveImageMaskImp(fp)

        with self.assertLogs(logger, level="ERROR") as cm:
            widget._loadImageMaskImp(fp)
        self.assertEqual(cm.output[0].split(':')[-1],
                         'Cannot load image mask without image!')

        imgs = np.arange(100, dtype=np.float).reshape(10, 10)
        mask = np.zeros_like(imgs, dtype=bool)
        widget.setImageData(_SimpleImageData.from_array(imgs))

        # the IOError
        with self.assertLogs(logger, level="ERROR") as cm:
            widget._loadImageMaskImp('abc')
        self.assertEqual(cm.output[0].split(':')[-1],
                         'Cannot load mask from abc')

        widget._publish_image_mask = MagicMock()

        widget._saveImageMaskImp(fp)

        fp.seek(0)
        widget._loadImageMaskImp(fp)
        widget._publish_image_mask.assert_called_once()
        widget._publish_image_mask.reset_mock()

        # save and load another mask
        mask[0, 0] = 1
        mask[5, 5] = 1
        mask_item = widget._mask_item
        mask_item._mask.setPixelColor(0, 0, mask_item._OPAQUE)
        mask_item._mask.setPixelColor(5, 5, mask_item._OPAQUE)
        fp.seek(0)
        widget._saveImageMaskImp(fp)
        fp.seek(0)
        widget._loadImageMaskImp(fp)
        widget._publish_image_mask.assert_called_once()

        # load a mask with different shape
        new_mask = np.array((3, 3), dtype=bool)
        fp.seek(0)
        np.save(fp, new_mask)
        fp.seek(0)
        with self.assertLogs(logger, level='ERROR'):
            widget._loadImageMaskImp(fp)

        fp.close()
