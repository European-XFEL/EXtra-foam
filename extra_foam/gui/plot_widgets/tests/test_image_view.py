import pytest
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile

import numpy as np

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.image_items import ImageItem, MaskItem, RectROI
from extra_foam.gui.plot_widgets.image_view_base import ImageViewF, TimedImageViewF
from extra_foam.gui.plot_widgets.image_views import (
    ImageAnalysis, RoiImageView,
)
from extra_foam.pipeline.data_model import ImageData, ProcessedData, RectRoiGeom
from extra_foam.logger import logger

from . import _display

app = mkQApp()

logger.setLevel("CRITICAL")


class TestImageView:
    def testComponents(self):
        widget = ImageViewF(has_roi=True)
        items = widget._plot_widget._plot_area._vb.addedItems
        assert isinstance(items[0], ImageItem)
        for i in range(1, 5):
            assert isinstance(items[i], RectROI)

        widget = ImageViewF()
        assert len(widget._plot_widget._plot_area._items) == 1

        with pytest.raises(TypeError, match="numpy array"):
            widget.setImage([[1, 2, 3], [4, 5, 6]])

        # test log X/Y menu is disabled
        menu = widget._plot_widget._plot_area.getContextMenus(None)
        assert len(menu) == 1
        assert menu[0].title() == "Grid"

    def testForwardMethod(self):
        widget = ImageViewF(has_roi=True)

        for method in ["setAspectLocked", "setLabel", "setTitle", "addItem",
                       "removeItem", "invertX", "invertY", "autoRange"]:
            with patch.object(widget._plot_widget, method) as mocked:
                getattr(widget, method)()
                mocked.assert_called_once()

    @pytest.mark.parametrize("dtype", [np.uint8, np.int, np.float32])
    def testSetImage(self, dtype):
        widget = ImageViewF(has_roi=True)

        if _display():
            widget.show()

        _display()

        # test setImage
        img = np.arange(64).reshape(8, 8).astype(dtype)
        widget.setImage(img, auto_range=False, auto_levels=False)
        _display()

        widget.setImage(img, auto_range=True, auto_levels=True)
        _display()

        # test setting image to None
        widget.setImage(None)
        assert widget._image is None
        assert widget._image_item._image is None

        _display()

    def testRoiImageView(self):
        widget = RoiImageView(1)
        widget.setImage = MagicMock()
        processed = ProcessedData(1)
        processed.image.masked_mean = np.ones((3, 3))

        # invalid ROI rect
        assert list(processed.roi.geom1.geometry) == RectRoiGeom.INVALID
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

    def testGeneral(self):
        widget = ImageAnalysis()
        items = widget._plot_widget._plot_area._vb.addedItems
        self.assertIsInstance(items[0], ImageItem)
        self.assertIsInstance(items[1], MaskItem)
        for i in range(2, 6):
            self.assertIsInstance(items[i], RectROI)

    def testSetImage(self):
        widget = ImageAnalysis()
        item = widget._mask_item

        with self.assertRaisesRegex(TypeError, "ImageData"):
            widget.setImage([1, 2])

        with patch.object(item, "maybeInitializeMask") as init:
            with patch.object(item, "setMask") as set_mask:
                # test set valid data
                image_data = ImageData.from_array(np.ones((2, 10, 10)))
                widget.setImage(image_data)
                np.testing.assert_array_equal(image_data.image_mask_in_modules, widget._mask_in_modules)
                np.testing.assert_array_equal(image_data.masked_mean, widget._image)
                init.assert_called_once()
                init.reset_mock()
                set_mask.assert_not_called()

                # test set image with different shape
                with patch("extra_foam.gui.items.GeometryItem.geometry") as geom:
                    image_data = ImageData.from_array(np.ones((2, 4, 4)))
                    # image_mask_in_modules is None
                    widget.setImage(image_data)
                    geom.output_array_for_position_fast.assert_not_called()
                    geom.position_all_modules.assert_not_called()
                    init.assert_called_once()
                    init.reset_mock()
                    set_mask.assert_not_called()
                    # set image_mask_in_modules
                    image_data = ImageData.from_array(np.ones((2, 8, 8)))
                    image_data.image_mask_in_modules = np.ones((4, 2, 2))
                    widget.setImage(image_data)
                    geom.output_array_for_position_fast.assert_called_once()
                    geom.position_all_modules.assert_called_once()
                    init.assert_called_once()
                    init.reset_mock()
                    set_mask.assert_called_once()

                # test set with image = None
                image_data = ImageData()
                widget.setImage(image_data)
                self.assertIsNone(widget._mask_in_modules)
                self.assertIsNone(widget._image)
                init.assert_not_called()

    @patch('extra_foam.gui.plot_widgets.image_views.QFileDialog.getSaveFileName')
    @patch('extra_foam.gui.plot_widgets.image_views.QFileDialog.getOpenFileName')
    @patch("extra_foam.gui.plot_widgets.image_items.MaskItem.setMask")
    @patch("extra_foam.gui.plot_widgets.image_items.ImageMaskPub")
    def testSaveLoadImageMask(self, mocked_pub, mocked_set_mask, mocked_open, mocked_save):

        def save_mask_in_file(_fp, arr):
            _fp.seek(0)
            np.save(_fp, arr)
            _fp.seek(0)

        widget = ImageAnalysis()
        fp = tempfile.NamedTemporaryFile(suffix=".npy")
        img = np.arange(100, dtype=np.float).reshape(10, 10)

        # if image_data is None, it does not raise but only logger.error()
        with self.assertLogs(logger, level="ERROR") as cm:
            widget.saveImageMask()
            self.assertEqual('Image mask does not exist without an image!',
                             cm.output[0].split(':')[-1])

        with self.assertLogs(logger, level="ERROR") as cm:
            widget.loadImageMask()
            self.assertEqual(cm.output[0].split(':')[-1], 'Cannot load image mask without image!')

        widget.setImage(ImageData.from_array(img))

        # --------------------
        # test failing to save
        # --------------------

        # test saving a mask in modules without geometry
        widget._mask_save_in_modules = True
        with self.assertLogs(logger, level='ERROR') as cm:
            widget.saveImageMask()
            # geometry file is not specified
            self.assertIn('Failed to create geometry to dismantle image mask', cm.output[0])

        # --------------------
        # test fail to load
        # --------------------

        # the IOError
        mocked_open.return_value = ['abc']
        with self.assertLogs(logger, level="ERROR") as cm:
            widget.loadImageMask()
            self.assertIn('Cannot load mask from abc', cm.output[0])

        # test mask data with dimension not equal to 2 or 3
        mocked_open.return_value = [fp.name]
        new_mask = np.ones(3, dtype=bool)
        save_mask_in_file(fp, new_mask)
        with self.assertLogs(logger, level='ERROR') as cm:
            widget.loadImageMask()
            self.assertIn('Expect array with dimensions (2, 3): actual 1', cm.output[0])
            mocked_set_mask.assert_not_called()

        # test loading a mask in modules without geometry
        new_mask = np.ones((3, 3, 3), dtype=bool)
        save_mask_in_file(fp, new_mask)
        with patch("extra_foam.gui.items.GeometryItem._detector",
                   new_callable=PropertyMock, create=True) as mocked:
            with self.assertLogs(logger, level='ERROR') as cm:
                mocked.return_value = "FastCCD"
                widget._require_geometry = False
                widget.loadImageMask()
                self.assertIn('Only detectors with a geometry can have image mask in modules', cm.output[0])

            with self.assertLogs(logger, level='ERROR') as cm:
                mocked.return_value = "LPD"
                widget._require_geometry = True
                widget.loadImageMask()
                # geometry file is not specified
                self.assertIn('Failed to create geometry to assemble image mask', cm.output[0])
                mocked_set_mask.assert_not_called()

        # test (assembled) mask shape is different from the image
        new_mask = np.ones((3, 3), dtype=bool)
        save_mask_in_file(fp, new_mask)
        with self.assertLogs(logger, level='ERROR') as cm:
            widget.loadImageMask()
            self.assertIn('Shape of the image mask (3, 3) is different from the image (10, 10)',
                          cm.output[0])
            mocked_set_mask.assert_not_called()

        # --------------------
        # test save and load
        # --------------------

        # first save a valid assembled mask
        widget._mask_save_in_modules = False
        mocked_save.return_value = [fp.name]
        with self.assertLogs(logger, level="INFO"):
            widget.saveImageMask()

        # then load
        fp.seek(0)
        with self.assertLogs(logger, level="INFO"):
            widget.loadImageMask()
        mocked_set_mask.assert_called_once()
        mocked_set_mask.reset_mock()

        # save and load another mask
        mask_item = widget._mask_item
        mask_item._mask.setPixelColor(0, 0, mask_item._fill_color)
        mask_item._mask.setPixelColor(5, 5, mask_item._fill_color)
        fp.seek(0)
        widget.saveImageMask()
        fp.seek(0)
        widget.loadImageMask()
        mocked_set_mask.assert_called_once()
        mocked_set_mask.reset_mock()

        # -----------------------------
        # test saving mask in modules
        # -----------------------------
        widget._mask_save_in_modules = True
        def dismantle_side_effect(*args, **kwargs):
            raise ValueError
        with patch("extra_foam.gui.items.GeometryItem.geometry") as geom:
            mask_in_modules = np.ones_like((3, 3, 3), dtype=bool)
            geom.output_array_for_dismantle_fast.return_value = mask_in_modules
            widget.saveImageMask()
            geom.output_array_for_dismantle_fast.assert_called_once()
            geom.dismantle_all_modules.assert_called_once()

            geom.dismantle_all_modules.side_effect = dismantle_side_effect
            with self.assertLogs(logger, level='ERROR') as cm:
                widget.saveImageMask()
                self.assertIn("Geometry does not match the assembled image", cm.output[0])

        # -----------------------------
        # test loading mask in modules
        # -----------------------------

        new_mask = np.ones((3, 3, 3), dtype=bool)
        save_mask_in_file(fp, new_mask)
        fp.seek(0)
        with patch("extra_foam.gui.items.GeometryItem.geometry") as geom:
            assembled_mask = np.ones_like(img, dtype=bool)
            geom.output_array_for_position_fast.return_value = assembled_mask
            widget.loadImageMask()
            geom.output_array_for_position_fast.assert_called_once()
            geom.position_all_modules.assert_called_once()
            mocked_set_mask.assert_called_once_with(assembled_mask)

        fp.close()
