"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Concrete ImageViewF widgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

import numpy as np
from PyQt5 import QtCore, QtGui

from .image_view_base import ImageViewF
from .plot_items import ImageItem, MaskItem
from ...file_io import read_image, write_image
from ...ipc import ReferencePub
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

        self._ref_pub = ReferencePub()

    def setImage(self, *args, **kwargs):
        """Overload."""
        super().setImage(*args, **kwargs)
        self._mask_item.onSetImage()

    def setImageData(self, image_data, **kwargs):
        """Set the ImageData.

        :param _SimpleImageData image_data: _SimpleImageData instance.
        """
        self._image_data = image_data
        if image_data is None:
            return

        self.setImage(image_data.masked)

    def writeImage(self):
        """Write the current detector image to file.

        Note: image mask is not included.
        """
        if self._image is None:
            logger.error(f"[Image tool] Detector image is not available!")
            return

        filepath = QtGui.QFileDialog.getSaveFileName(
            caption="Save image",
            directory=osp.expanduser("~"),
            filter=self.IMAGE_FILE_FILTER)[0]

        try:
            write_image(self._image, filepath)
            logger.info(f"[Image tool] Image saved in {filepath}")
        except ValueError as e:
            logger.error(f"[Image tool] {str(e)}")

    def setReferenceImage(self):
        """Set the displayed image as reference image.

        Note: image mask is not included.
        """
        self._ref_pub.set(self._image)

    def removeReferenceImage(self):
        """Remove reference image."""
        self._ref_pub.remove()

    def loadReferenceImage(self):
        """Load the reference image from a file."""
        if self._image is None:
            logger.error("[Image tool] Cannot load reference image without "
                         "detector image!")
            return

        filepath = QtGui.QFileDialog.getOpenFileName(
            caption="Load reference image",
            directory=osp.expanduser("~"),
            filter=self.IMAGE_FILE_FILTER)[0]

        try:
            img = read_image(filepath, expected_shape=self._image.shape)
            logger.info(f"[Image tool] Loaded reference image from {filepath}")
            self._ref_pub.set(img)
        except ValueError as e:
            logger.error(f"[Image tool] {str(e)}")

    @QtCore.pyqtSlot(int, int, float)
    def onMouseMoved(self, x, y, v):
        if x < 0 or y < 0:
            self._plot_widget.setTitle('')
        else:
            self._plot_widget.setTitle(f'x={x}, y={y}, value={round(v, 1)}')

    @QtCore.pyqtSlot(float)
    def onBkgChange(self, bkg):
        if self._image_data is None:
            return

        self._image_data.background = bkg
        self.setImage(self._image_data.masked)

    @QtCore.pyqtSlot(object)
    def onThresholdMaskChange(self, mask_range):
        if self._image_data is None:
            return

        self._image_data.threshold_mask = mask_range
        self.setImage(self._image_data.masked)

    @QtCore.pyqtSlot(bool)
    def onDrawToggled(self, state, checked):
        self._mask_item.state = state
        self._image_item.drawing = checked

    @QtCore.pyqtSlot()
    def onClearImageMask(self):
        self._mask_item.removeMask()

    def saveImageMask(self):
        filepath = QtGui.QFileDialog.getSaveFileName()[0]
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
        filepath = QtGui.QFileDialog.getOpenFileName()[0]
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


class AssembledImageView(ImageViewF):
    """AssembledImageView class.

    Widget for displaying the assembled image of the average of all pulses
    in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

    def update(self, data):
        """Override."""
        self.setImage(data.image.masked_mean,
                      auto_levels=(not self._is_initialized))
        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class PumpProbeImageView(ImageViewF):
    """PumpProbeImageView class.

    Widget for displaying the on or off image in the pump-probe analysis.
    """
    def __init__(self, on=True, *, parent=None):
        """Initialization.

        :param bool on: True for display the on image while False for
            displaying the off image.
        """
        super().__init__(parent=parent)

        self._on = on

    def update(self, data):
        """Override."""
        if self._on:
            img = data.pp.image_on
        else:
            img = data.pp.image_off

        if img is None:
            return

        self.setImage(img, auto_levels=(not self._is_initialized))
        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class SinglePulseImageView(ImageViewF):
    """SinglePulseImageView class.

    Widget for displaying the assembled image of a single pulse.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.pulse_index = idx

    def update(self, data):
        """Override."""
        images = data.image.images

        try:
            self.setImage(images[self.pulse_index],
                          auto_levels=(not self._is_initialized))
        except IndexError:
            self.clear()
            return

        self.updateROI(data)

        if not self._is_initialized:
            self._is_initialized = True


class RoiImageView(ImageViewF):
    """RoiImageView class.

    Widget for displaying the ROI for the assembled image.
    """
    def __init__(self, rank, **kwargs):
        """Initialization."""
        super().__init__(has_roi=False, **kwargs)

        self._rank = rank

    def update(self, data):
        """Override."""
        image = data.image.masked_mean

        roi = getattr(data.roi, f"rect{self._rank}")

        x, y, w, h = roi
        if w < 0 or h < 0:
            return
        self.setImage(image[y:y+h, x:x+w], auto_range=True, auto_levels=True)


class Bin1dHeatmap(ImageViewF):
    """Bin1dHeatmap class.

    Widget for visualizing the heatmap of 1D binning.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization.

        :param int idx: index of the binning parameter (must be 1 or 2).
        """
        super().__init__(has_roi=False, hide_axis=False, parent=parent)
        self.invertY(False)
        self.setAspectLocked(False)

        self._idx = idx

        self.setDefautLabels()

    def setDefautLabels(self):
        self.setLabel('bottom', 'Bin center')
        self.setLabel('left', "VFOM")
        self.setTitle('')

    def update(self, data):
        """Override."""
        bin = getattr(data.bin, f"bin{self._idx}")

        if not bin.has_vfom and self._image is not None:
            # clear the heatmap if VFOM does not exists for the analysis type
            self.clear()
            self.setDefautLabels()
            return

        if not bin.updated:
            return

        heatmap = bin.vfom_heat
        if heatmap is not None:
            h, w = heatmap.shape
            w_range = bin.center
            h_range = bin.vfom_x

            self.setLabel('left', bin.vfom_x_label)
            self.setLabel('bottom', bin.label)
            self.setTitle(bin.vfom_label)

            self.setImage(heatmap,
                          auto_levels=True,
                          auto_range=True,
                          pos=[w_range[0], h_range[0]],
                          scale=[(w_range[-1] - w_range[0])/w,
                                 (h_range[-1] - h_range[0])/h])


class Bin2dHeatmap(ImageViewF):
    """Bin2dHeatmap class.

    Widget for visualizing the heatmap of 2D binning.
    """
    def __init__(self, *, count=False, parent=None):
        """Initialization.

        :param bool count: True for count plot and False for value plot.
        """
        self._count = count
        super().__init__(has_roi=False, hide_axis=False, parent=parent)
        self.invertY(False)
        self.setAspectLocked(False)

        self.setLabel('bottom', '')
        self.setLabel('left', '')
        if count:
            self.setTitle("Count")
        else:
            self.setTitle("FOM")

    def update(self, data):
        """Override."""
        bin = data.bin.bin12
        if not bin.updated:
            return

        if self._count:
            heatmap = bin.count_heat
        else:
            heatmap = bin.fom_heat

        # do not update if FOM is None
        if heatmap is not None:
            h, w = heatmap.shape
            w_range = bin.center_x
            h_range = bin.center_y

            self.setLabel('bottom', bin.x_label)
            self.setLabel('left', bin.y_label)

            self.setImage(heatmap,
                          auto_levels=True,
                          auto_range=True,
                          pos=[w_range[0], h_range[0]],
                          scale=[(w_range[-1] - w_range[0])/w,
                                 (h_range[-1] - h_range[0])/h])
