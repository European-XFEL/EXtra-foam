"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

import numpy as np

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from .image_view_base import ImageViewF
from .plot_items import ImageItem, MaskItem
from ...file_io import write_image
from ...logger import logger


class ImageAnalysis(ImageViewF):
    """ImageAnalysis widget.

    Advance image analysis widget built on top of ImageViewF widget.
    It provides tools like masking, etc.
    """

    IMAGE_FILE_FILTER = "All supported files (*.tif *.npy)"

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._plot_widget.setTitle('')  # reserve space for displaying

        # set the customized ImageItem
        self._image_item = ImageItem()
        self._image_item.mouse_moved_sgn.connect(self.onMouseMoved)
        self._mask_item = MaskItem(self._image_item)

        # re-add items to keep the order
        self._plot_widget.clear()
        self._plot_widget.addItem(self._image_item)
        self._plot_widget.addItem(self._mask_item)
        for roi in self._rois:
            roi.setLocked(False)
            self._plot_widget.addItem(roi)

        self.invertY(True)
        self.setAspectLocked(True)
        self._hist_widget.setImageItem(self._image_item)

        self._image_data = None

    def setImage(self, *args, **kwargs):
        """Overload."""
        super().setImage(*args, **kwargs)
        self._mask_item.onSetImage()

    def setImageData(self, image_data, **kwargs):
        """Set the ImageData.

        :param _SimpleImageData image_data: _SimpleImageData instance.
        """
        self._image_data = image_data
        if image_data is not None:
            self.setImage(image_data.masked)

    def writeImage(self):
        """Write the current detector image to file.

        Note: image mask is not included.
        """
        if self._image is None:
            logger.error(f"[Image tool] Detector image is not available!")
            return

        filepath = QFileDialog.getSaveFileName(
            caption="Save image",
            directory=osp.expanduser("~"),
            filter=self.IMAGE_FILE_FILTER)[0]

        try:
            write_image(self._image, filepath)
            logger.info(f"[Image tool] Image saved in {filepath}")
        except ValueError as e:
            logger.error(f"[Image tool] {str(e)}")

    @pyqtSlot(int, int, float)
    def onMouseMoved(self, x, y, v):
        if x < 0 or y < 0:
            self._plot_widget.setTitle('')
        else:
            self._plot_widget.setTitle(f'x={x}, y={y}, value={round(v, 1)}')

    @pyqtSlot(float)
    def onBkgChange(self, bkg):
        if self._image_data is None:
            return

        self._image_data.background = bkg
        self.setImage(self._image_data.masked)

    @pyqtSlot(object)
    def onThresholdMaskChange(self, mask_range):
        if self._image_data is None:
            return

        self._image_data.threshold_mask = mask_range
        self.setImage(self._image_data.masked)

    @pyqtSlot(bool)
    def onDrawToggled(self, state, checked):
        self._mask_item.state = state
        self._image_item.drawing = checked

    @pyqtSlot()
    def onClearImageMask(self):
        self._mask_item.removeMask()

    def saveImageMask(self):
        filepath = QFileDialog.getSaveFileName()[0]
        if not filepath:
            logger.error("Please specify the image mask file!")
            return

        self._saveImageMaskImp(filepath)

    def _saveImageMaskImp(self, filepath):
        if self._image_data is None:
            logger.error("Image is not found!")
            return

        np.save(filepath, self._mask_item.toNDArray())
        logger.info(f"Image mask saved in {filepath}.npy")

    def loadImageMask(self):
        filepath = QFileDialog.getOpenFileName()[0]
        if not filepath:
            logger.error("Please specify the image mask file!")
            return

        self._loadImageMaskImp(filepath)

    def _loadImageMaskImp(self, filepath):
        if self._image is None:
            logger.error("Cannot load image mask without image!")
            return

        try:
            image_mask = np.load(filepath)
            if image_mask.shape != self._image.shape:
                logger.error(f"The shape of image mask {image_mask.shape} is "
                             f"different from the image {self._image.shape}!")
                return

            logger.info(f"Image mask loaded from {filepath}!")
            self._mask_item.loadMask(image_mask)

        except (IOError, OSError) as e:
            logger.error(f"Cannot load mask from {filepath}")


class RoiImageView(ImageViewF):
    """RoiImageView class.

    Widget for displaying the ROI for the assembled image.
    """
    def __init__(self, idx, **kwargs):
        """Initialization."""
        super().__init__(has_roi=False, **kwargs)

        self._index = idx
        self.setTitle(f"ROI{idx}")

    def updateF(self, data):
        """Override."""
        image = data.image.masked_mean

        x, y, w, h = getattr(getattr(
            data.roi, f"geom{self._index}"), "geometry")
        if w < 0 or h < 0:
            return
        self.setImage(image[y:y+h, x:x+w], auto_range=True, auto_levels=True)
