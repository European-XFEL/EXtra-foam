"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

import numpy as np

from PyQt5.QtWidgets import QFileDialog

from .image_view_base import ImageViewF
from .plot_items import MaskItem
from ...file_io import write_image, read_numpy_array
from ...logger import logger
from ..items import GeometryItem


class ImageAnalysis(ImageViewF):
    """ImageAnalysis widget.

    Advance image analysis widget built on top of ImageViewF widget.
    It provides tools like masking, etc.
    """

    IMAGE_FILE_FILTER = "All supported files (*.tif *.npy)"

    def __init__(self, has_roi=True, **kwargs):
        """Initialization."""
        super().__init__(has_roi=has_roi, **kwargs)

        self._geom = GeometryItem()

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
        self._mask_save_in_modules = False

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
            write_image(filepath, self._image)
            logger.info(f"[Image tool] Image saved in {filepath}")
        except ValueError as e:
            logger.error(f"[Image tool] {str(e)}")

    def setMaskingState(self, state, checked):
        self._mask_item.state = state
        self._image_item.drawing = checked

    def removeMask(self):
        self._mask_item.removeMask()

    def setMaskSaveInModules(self, state):
        self._mask_save_in_modules = state

    def saveImageMask(self):
        if self._image is None:
            logger.error("No image is available!")
            return

        filepath = QFileDialog.getSaveFileName()[0]
        if not filepath:
            return

        mask = self._mask_item.mask()
        if self._mask_save_in_modules:
            # TODO: convert assembled mask to mask in modules
            raise NotImplementedError

        np.save(filepath, mask)

        logger.info(f"Image mask saved in {filepath}.npy")

    def loadImageMask(self):
        if self._image is None:
            logger.error("Cannot load image mask without image!")
            return

        filepath = QFileDialog.getOpenFileName()[0]
        if not filepath:
            return

        try:
            image_mask = read_numpy_array(filepath, dimensions=(2, 3))
        except ValueError as e:
            logger.error(f"Cannot load mask from {filepath}: {str(e)}")
            return

        logger.info(f"Loaded mask data with shape = {image_mask.shape}, "
                    f"dtype = {image_mask.dtype}")

        if image_mask.ndim == 3:
            try:
                geom = self._geom.geometry
            except Exception as e:
                logger.error(f"Failed to create geometry to assemble mask: "
                             f"{str(e)}")
                return

            if geom is None:
                logger.error(f"Mask in modules requires a geometry!")
                return

            try:
                assembled = geom.output_array_for_position_fast(dtype=bool)
                geom.position_all_modules(image_mask, out=assembled)
                image_mask = assembled
            except Exception as e:
                logger.error(f"Failed to assemble mask in modules: {str(e)}")
                return

        if image_mask.shape != self._image.shape:
            logger.error(f"Shape of the image mask {image_mask.shape} is "
                         f"different from the image {self._image.shape}!")
            return

        self._mask_item.setMask(image_mask)


class RoiImageView(ImageViewF):
    """RoiImageView class.

    Widget for displaying the ROI for the assembled image.
    """
    def __init__(self, idx, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)

        self._index = idx
        self.setTitle(f"ROI{idx}")

    def updateF(self, data):
        """Override."""
        image = data.image.masked_mean

        x, y, w, h = getattr(getattr(
            data.roi, f"geom{self._index}"), "geometry")
        if w < 0 or h < 0:
            return
        self.setImage(image[y:y+h, x:x+w])
