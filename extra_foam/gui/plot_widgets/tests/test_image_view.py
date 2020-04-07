import unittest
from unittest.mock import MagicMock
import tempfile

import numpy as np

from extra_foam.gui import mkQApp, pyqtgraph
from extra_foam.gui.image_tool.simple_image_data import _SimpleImageData
from extra_foam.gui.plot_widgets.plot_items import ImageItem, MaskItem, RectROI
from extra_foam.gui.plot_widgets.image_view_base import ImageViewF, TimedImageViewF
from extra_foam.gui.plot_widgets.image_views import (
    ImageAnalysis, RoiImageView,
)
from extra_foam.pipeline.data_model import ProcessedData, RectRoiGeom
from extra_foam.logger import logger

app = mkQApp()

logger.setLevel("CRITICAL")


class TestImageView(unittest.TestCase):
    def testGeneral(self):
        widget = ImageViewF(has_roi=True)
        plot_items = widget._plot_widget._plot_item.items
        self.assertIsInstance(plot_items[0], pyqtgraph.ImageItem)
        for i in range(1, 5):
            self.assertIsInstance(plot_items[i], RectROI)

        widget = ImageViewF()
        self.assertEqual(1, len(widget._plot_widget._plot_item.items))

        with self.assertRaisesRegex(TypeError, "numpy array"):
            widget.setImage([[1, 2, 3], [4, 5, 6]])

        # test setting a valid image
        widget.setImage(np.random.randn(4, 4))
        widget.updateImageWithAutoLevel()  # test not raise

        # test setting image to None
        widget.setImage(None)
        self.assertIsNone(widget._image)
        self.assertIsNone(widget._image_item.image)

    def testRoiImageView(self):
        widget = RoiImageView(1)
        widget.setImage = MagicMock()
        processed = ProcessedData(1)
        processed.image.masked_mean = np.ones((3, 3))

        # invalid ROI rect
        self.assertListEqual(RectRoiGeom.INVALID, list(processed.roi.geom1.geometry))
        widget.updateF(processed)
        widget.setImage.assert_not_called()

        # invalid ROI rect
        processed.roi.geom1.geometry = [0, 0, -1, 0]
        widget.updateF(processed)
        widget.setImage.assert_not_called()

        # valid ROI rect
        processed.roi.geom1.geometry = [0, 0, 2, 2]
        widget.updateF(processed)
        widget.setImage.assert_called_once()


class TestTimedImageView(unittest.TestCase):
    def testUpdate(self):
        view = TimedImageViewF()
        view.refresh = MagicMock()

        self.assertIsNone(view._data)
        view._refresh_imp()
        view.refresh.assert_not_called()

        view.updateF(1)
        view._refresh_imp()
        view.refresh.assert_called_once()


class TestImageAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MaskItem.resetMask()

    def testGeneral(self):
        widget = ImageAnalysis()
        plot_items = widget._plot_widget._plot_item.items
        self.assertIsInstance(plot_items[0], ImageItem)
        self.assertIsInstance(plot_items[1], MaskItem)
        for i in range(2, 6):
            self.assertIsInstance(plot_items[i], RectROI)

    def testSaveLoadImageMask(self):
        widget = ImageAnalysis()
        widget._mask_item.loadMask = MagicMock()

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

        widget._saveImageMaskImp(fp)

        fp.seek(0)
        widget._loadImageMaskImp(fp)
        widget._mask_item.loadMask.assert_called_once()
        widget._mask_item.loadMask.reset_mock()

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
        widget._mask_item.loadMask.assert_called_once()
        widget._mask_item.loadMask.reset_mock()

        # load a mask with different shape
        new_mask = np.array((3, 3), dtype=bool)
        fp.seek(0)
        np.save(fp, new_mask)
        fp.seek(0)
        with self.assertLogs(logger, level='ERROR'):
            widget._loadImageMaskImp(fp)

        fp.close()
