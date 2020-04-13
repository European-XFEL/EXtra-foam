import unittest
from unittest.mock import MagicMock, patch, PropertyMock
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
from extra_foam.config import config
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

    @patch('extra_foam.gui.plot_widgets.image_views.QFileDialog.getSaveFileName')
    @patch('extra_foam.gui.plot_widgets.image_views.QFileDialog.getOpenFileName')
    @patch("extra_foam.gui.plot_widgets.plot_items.MaskItem.setMask")
    @patch("extra_foam.gui.plot_widgets.plot_items.ImageMaskPub")
    def testSaveLoadImageMask(self, patched_pub, patched_setMask, patched_open, patched_save):

        def save_mask_in_file(_fp, arr):
            fp.seek(0)
            np.save(_fp, arr)
            fp.seek(0)

        widget = ImageAnalysis()

        # if image_data is None, it does not raise but only logger.error()
        with self.assertLogs(logger, level="ERROR") as cm:
            widget.saveImageMask()
            self.assertEqual('No image is available!', cm.output[0].split(':')[-1])

        with self.assertLogs(logger, level="ERROR") as cm:
            widget.loadImageMask()
            self.assertEqual(cm.output[0].split(':')[-1], 'Cannot load image mask without image!')

        imgs = np.arange(100, dtype=np.float).reshape(10, 10)
        mask = np.zeros_like(imgs, dtype=bool)
        widget.setImageData(_SimpleImageData.from_array(imgs))

        # the IOError
        patched_open.return_value = ['abc']
        with self.assertLogs(logger, level="ERROR") as cm:
            widget.loadImageMask()
            self.assertIn('Cannot load mask from abc', cm.output[0])

        fp = tempfile.NamedTemporaryFile(suffix=".npy")

        patched_save.return_value = [fp.name]
        widget.saveImageMask()

        fp.seek(0)
        patched_open.return_value = [fp.name]
        with self.assertLogs(logger, level="INFO"):
            widget.loadImageMask()
        patched_setMask.assert_called_once()
        patched_setMask.reset_mock()

        # save and load another mask
        mask[0, 0] = 1
        mask[5, 5] = 1
        mask_item = widget._mask_item
        mask_item._mask.setPixelColor(0, 0, mask_item._OPAQUE)
        mask_item._mask.setPixelColor(5, 5, mask_item._OPAQUE)
        fp.seek(0)
        widget.saveImageMask()
        fp.seek(0)
        widget.loadImageMask()
        patched_setMask.assert_called_once()
        patched_setMask.reset_mock()

        # test mask data with dimension not equal to 2 or 3
        new_mask = np.ones(3, dtype=bool)
        save_mask_in_file(fp, new_mask)
        with self.assertLogs(logger, level='ERROR') as cm:
            widget.loadImageMask()
            self.assertIn('Expect array with dimensions (2, 3): actual 1', cm.output[0])
            patched_setMask.assert_not_called()

        # test loading a mask in modules without geometry
        new_mask = np.ones((3, 3, 3), dtype=bool)
        save_mask_in_file(fp, new_mask)
        with self.assertLogs(logger, level='ERROR') as cm:
            with patch("extra_foam.gui.items.GeometryItem._detector",
                       new_callable=PropertyMock, create=True) as mocked:
                mocked.return_value = "LPD"
                widget.loadImageMask()
                self.assertIn('Failed to create geometry to assemble mask', cm.output[0])
                patched_setMask.assert_not_called()

        fp.seek(0)
        with self.assertLogs(logger, level='ERROR') as cm:
            with patch("extra_foam.gui.items.GeometryItem.geometry",
                       new_callable=PropertyMock, create=True) as mocked:
                mocked.return_value = None
                widget.loadImageMask()
                self.assertIn('Mask in modules requires a geometry', cm.output[0])
                patched_setMask.assert_not_called()

        fp.seek(0)
        with self.assertLogs(logger, level='ERROR') as cm:
            with patch("extra_foam.gui.items.GeometryItem.geometry",
                       new_callable=PropertyMock, create=True) as mocked:
                mocked.return_value = None
                widget.loadImageMask()
                self.assertIn('Mask in modules requires a geometry', cm.output[0])
                patched_setMask.assert_not_called()

        # TODO: test position_all_modules fail

        # test (assembled) mask shape is different from the image
        new_mask = np.ones((3, 3), dtype=bool)
        save_mask_in_file(fp, new_mask)
        with self.assertLogs(logger, level='ERROR') as cm:
            widget.loadImageMask()
            self.assertIn('Shape of the image mask (3, 3) is different from the image (10, 10)',
                          cm.output[0])
            patched_setMask.assert_not_called()

        fp.close()
