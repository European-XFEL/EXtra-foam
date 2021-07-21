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
from .image_items import MaskItem
from ..items import GeometryItem
from ...config import config
from ...file_io import write_image, read_numpy_array
from ...logger import logger
from ...pipeline.data_model import ImageData


class ImageAnalysis(ImageViewF):
    """ImageAnalysis widget.

    Advance image analysis widget built on top of ImageViewF widget.
    It provides tools like masking, etc.
    """

    IMAGE_FILE_FILTER = "All supported files (*.tif *.npy)"

    def __init__(self, has_roi=True, **kwargs):
        """Initialization."""
        super().__init__(has_roi=has_roi, **kwargs)

        self._geom_item = GeometryItem()

        self._mask_item = MaskItem(self._image_item)

        # re-add items to keep the order
        self._plot_widget.removeAllItems()
        self._plot_widget.addItem(self._image_item)
        self._plot_widget.addItem(self._mask_item)
        for roi in self._rois:
            roi.setLocked(False)
            self._plot_widget.addItem(roi)

        self._require_geometry = config["REQUIRE_GEOMETRY"]
        self._mask_in_modules = None
        self._mask_save_in_modules = False

    def setImage(self, image_data, **kwargs):
        """Overload."""
        if not isinstance(image_data, ImageData):
            raise TypeError(
                "The first argument must be an ImageData instance!")

        # It will be None for detectors without a geometry and detector
        # with a geometry but has no masking operation yet.
        self._mask_in_modules = image_data.image_mask_in_modules

        image = image_data.masked_mean

        # re-assemble a mask if image shape changes
        # caveat: the image shape checked must be done before updating image
        if self._mask_in_modules is not None \
                and image is not None \
                and self._image is not None \
                and image.shape != self._image.shape:
            geom = self._geom_item.geometry
            assembled = geom.output_array_for_position_fast(dtype=bool)
            geom.position_all_modules(self._mask_in_modules, out=assembled)
            self._mask_item.setMask(assembled)

        self._updateImageImp(image, **kwargs)
        if image is not None:
            self._mask_item.maybeInitializeMask(image.shape)

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
            logger.error("Image mask does not exist without an image!")
            return

        filepath = QFileDialog.getSaveFileName()[0]
        if not filepath:
            return

        image_mask = self._mask_item.mask()
        if self._mask_save_in_modules:
            try:
                geom = self._geom_item.geometry
            except Exception as e:
                logger.error(
                    f"Failed to create geometry to dismantle image mask: "
                    f"{str(e)}")
                return

            # We do not use self._mask_in_modules in order to allow users
            # to save mask which has not been applied and send back by the
            # pipeline.
            modules = geom.output_array_for_dismantle_fast(dtype=bool)
            try:
                geom.dismantle_all_modules(image_mask, out=modules)
            except ValueError as e:
                logger.error(f"{str(e)}. "
                             f"Geometry does not match the assembled "
                             f"image! Change the geometry back or wait "
                             f"until update of new assembled image.")
                return

            image_mask = modules

        np.save(filepath, image_mask)

        if not filepath.endswith('.npy'):
            filepath += '.npy'

        if self._mask_save_in_modules:
            logger.info(f"Image mask saved in modules in {filepath}")
        else:
            logger.info(f"Image mask saved in {filepath}")

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
            if not self._require_geometry:
                logger.error(f"Only detectors with a geometry can have image "
                             f"mask in modules!")
                return

            try:
                geom = self._geom_item.geometry
            except Exception as e:
                logger.error(f"Failed to create geometry to assemble image mask: "
                             f"{str(e)}")
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
