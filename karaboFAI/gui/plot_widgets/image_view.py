"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageView and other derivative ImageView widgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .. import pyqtgraph as pg
from ..pyqtgraph import HistogramLUTWidget, QtCore, QtGui

from .base_plot_widget import PlotWidget
from .plot_items import ImageItem, MaskItem, RectROI
from ..misc_widgets import colorMapFactory, make_pen
from ..mediator import Mediator
from ...command import CommandProxy
from ...algorithms import quick_min_max
from ...config import config
from ...logger import logger


class ImageView(QtGui.QWidget):
    """ImageView class.

    A widget used for displaying 2D image data.

    * Four ROIs and one MaskItem are included in this widget by default.

    Note: it is different from the ImageView in pyqtgraph!

    Attributes:
        _image_item (pyqtgraph.ImageItem): This object will be used to
            display the image.

    """
    ROI_X0 = 50
    ROI_Y0 = 50
    ROI_SIZE0 = (100, 100)

    def __init__(self, *,
                 level_mode='mono',
                 has_roi=True,
                 hide_axis=True,
                 color_map=None,
                 parent=None):
        """Initialization.

        :param str level_mode: 'mono' or 'rgba'. If 'mono', then only
            a single set of black/white level lines is drawn, and the
            levels apply to all channels in the image. If 'rgba', then
            one set of levels is drawn for each channel.
        :param bool has_roi: True for adding 4 ROIs on top of the other
            PlotItems.
        :param bool hide_axis: True for hiding left and bottom axes.
        """
        super().__init__(parent=parent)
        try:
            parent.registerPlotWidget(self)
        except AttributeError:
            pass

        self._mediator = Mediator()

        self._rois = []
        if has_roi:
            self._initializeROIs()

        self._plot_widget = PlotWidget()
        if hide_axis:
            self._plot_widget.hideAxis()

        self._image_item = pg.ImageItem()
        self._plot_widget.addItem(self._image_item)

        for roi in self._rois:
            self._plot_widget.addItem(roi)

        self.invertY(True)  # y-axis points from top to bottom
        self.setAspectLocked(True)

        self._hist_widget = HistogramLUTWidget()
        self._hist_widget.setLevelMode(level_mode)
        self._hist_widget.setImageItem(self._image_item)

        if color_map is None:
            self.setColorMap(colorMapFactory[config["GUI"]["COLOR_MAP"]])
        else:
            self.setColorMap(colorMapFactory["thermal"])

        self._is_initialized = False
        self._image = None
        self._image_levels = None

        self.initUI()

        self._mediator.reset_image_level_sgn.connect(self._updateImage)

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._plot_widget)
        layout.addWidget(self._hist_widget)
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

    def reset(self):
        self.clear()

    def update(self, data):
        """karaboFAI interface."""
        pass

    def _initializeROIs(self):
        for i, color in enumerate(config["ROI_COLORS"], 1):
            roi = RectROI(i,
                          pos=(self.ROI_X0 + 10*i, self.ROI_Y0 + 10*i),
                          size=self.ROI_SIZE0,
                          pen=make_pen(color, width=2, style=QtCore.Qt.SolidLine))
            roi.hide()
            self._rois.append(roi)

    def updateROI(self, data):
        """Update ROIs.

        Update ROI through data instead of passing signals to ensure that
        visualization of ROIs and calculation of ROI data are synchronized.
        """
        for i, roi in enumerate(self._rois, 1):
            x, y, w, h = getattr(data.roi, f"rect{i}")
            if w > 0 and h > 0:
                roi.show()
                roi.setSize((w, h), update=False)
                roi.setPos((x, y), update=False)
            else:
                roi.hide()

    @property
    def image(self):
        return self._image

    @property
    def rois(self):
        return self._rois

    def setImage(self, img, *, auto_range=False, auto_levels=False,
                 scale=None, pos=None):
        """Set the current displayed image.

        :param np.ndarray img: the image to be displayed.
        :param bool auto_range: whether to scale/pan the view to fit
            the image. defaut = False
        :param bool auto_levels: whether to update the white/black levels
            to fit the image. default = False
        :param tuple/list pos: the origin of the displayed image in (x, y).
        :param tuple/list scale: the origin of the displayed image image in
            (x_scale, y_scale).
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Image data must be a numpy array!")

        self._image_item.setImage(img, autoLevels=False)
        self._image = img

        self._image_item.resetTransform()
        if scale is not None:
            self._image_item.scale(*scale)
        if pos is not None:
            self._image_item.setPos(*pos)

        if auto_levels:
            self._image_levels = quick_min_max(self._image)
            self.setLevels(rgba=[self._image_levels])

        if auto_range:
            self._plot_widget.plotItem.vb.autoRange()

    def clear(self):
        self._image = None
        self._image_item.clear()

    def _updateImage(self):
        """Re-display the current image with auto_levels."""
        if self._image is None:
            return
        self.setImage(self._image, auto_levels=True)

    def setLevels(self, *args, **kwargs):
        """Set the min/max (bright and dark) levels.

        See HistogramLUTItem.setLevels.
        """
        self._hist_widget.setLevels(*args, **kwargs)

    def setColorMap(self, cm):
        """Set colormap for the displayed image.

        :param cm: a ColorMap object.
        """
        self._hist_widget.gradient.setColorMap(cm)

    def setBorder(self, *args, **kwargs):
        self._image_item.setBorder(*args, **kwargs)

    def setAspectLocked(self, *args, **kwargs):
        self._plot_widget.setAspectLocked(*args, **kwargs)

    def setLabel(self, *args, **kwargs):
        self._plot_widget.setLabel(*args, **kwargs)

    def invertY(self, *args, **kwargs):
        self._plot_widget.plotItem.invertY(*args, **kwargs)

    def addItem(self, *args, **kwargs):
        self._plot_widget.addItem(*args, **kwargs)

    def removeItem(self, *args, **kwargs):
        self._plot_widget.removeItem(*args, **kwargs)

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()


class ImageAnalysis(ImageView):
    """ImageAnalysis widget.

    Advance image analysis widget built on top of ImageView widget.
    It provides tools like masking, etc.
    """

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

        self._cmd_proxy = CommandProxy()

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

    def setReferenceImage(self):
        """Set the displayed image as reference image.

        Note: image mask is not included in self._image.
        """
        self._cmd_proxy.set_ref_image(self._image)

    def removeReferenceImage(self):
        """Remove reference image."""
        self._cmd_proxy.remove_ref_image()

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
        self._mask_item.clearMask()

    def saveImageMask(self):
        file_path = QtGui.QFileDialog.getSaveFileName()[0]
        if not file_path:
            logger.error("Please specify the image mask file!")
            return

        self._saveImageMaskImp(file_path)

    def _saveImageMaskImp(self, file_path):
        if self._image_data is None:
            logger.error("Image is not found!")
            return

        np.save(file_path, self._mask_item.toNDArray())
        logger.info(f"Image mask saved in {file_path}.npy!")

    def loadImageMask(self):
        file_path = QtGui.QFileDialog.getOpenFileName()[0]
        if not file_path:
            logger.error("Please specify the image mask file!")
            return

        self._loadImageMaskImp(file_path)

    def _loadImageMaskImp(self, file_path):
        if self._image is None:
            logger.error("Cannot load image mask without image!")
            return

        try:
            image_mask = np.load(file_path)
            if image_mask.shape != self._image.shape:
                logger.error(f"The shape of image mask {image_mask.shape} is "
                             f"different from the image {self._image.shape}!")
                return

            logger.info(f"Image mask loaded from {file_path}!")

            self._mask_item.loadMask(image_mask)

        except (IOError, OSError) as e:
            logger.error(f"Cannot load mask from {file_path}")


class AssembledImageView(ImageView):
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


class PumpProbeImageView(ImageView):
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


class SinglePulseImageView(ImageView):
    """SinglePulseImageView class.

    Widget for displaying the assembled image of a single pulse.
    """
    def __init__(self, *, pulse_index=0, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.pulse_index = pulse_index

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


class RoiImageView(ImageView):
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


class Bin1dHeatmap(ImageView):
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

        self.setLabel('bottom', f'Label{idx}')
        self.setLabel('left', f'Vec{idx}')

    def update(self, data):
        """Override."""
        heatmap = getattr(data.bin, f"vec{self._idx}_hist")
        vec = getattr(data.bin, f"vec{self._idx}")

        reset = getattr(data.bin, f"reset{self._idx}")
        # do not update if vec is None
        if heatmap is not None and (reset or vec is not None):
            h, w = heatmap.shape
            w_range = getattr(data.bin, f"edge{self._idx}")
            h_range = data.bin.vec_x

            self.setImage(heatmap,
                          auto_levels=True,
                          auto_range=True,
                          pos=[w_range[0], h_range[0]],
                          scale=[(w_range[-1] - w_range[0])/w,
                                 (h_range[-1] - h_range[0])/h])

            self.setLabel('left', data.bin.vec_label)
            self.setLabel('bottom', getattr(data.bin, f"label{self._idx}"))
        elif heatmap is None and reset:
            w_range = getattr(data.bin, f"edge{self._idx}")
            n_bins = getattr(data.bin, f"n_bins{self._idx}")
            # A temporary all-zero array is used to reset the vector heatmap
            # if no vector result is available.
            self.setImage(np.zeros((1, n_bins), dtype=np.float32),
                          auto_levels=True,
                          auto_range=True,
                          pos=[w_range[0], 0],
                          scale=[(w_range[-1] - w_range[0])/n_bins, 1.0])

            self.setLabel('left', f"vec{self._idx}")
            self.setLabel('bottom', getattr(data.bin, f"label{self._idx}"))


class Bin2dHeatmap(ImageView):
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

        self.setLabel('bottom', 'Label1')
        self.setLabel('left', 'Label2')

    def update(self, data):
        """Override."""
        if self._count:
            heatmap = data.bin.count12_hist
        else:
            heatmap = data.bin.fom12_hist

        # do not update if FOM is None
        reset = data.bin.reset1 or data.bin.reset2
        if heatmap is not None and (reset or data.bin.fom12 is not None):
            h, w = heatmap.shape
            w_range = data.bin.edge1
            h_range = data.bin.edge2

            self.setImage(heatmap,
                          auto_levels=True,
                          auto_range=True,
                          pos=[w_range[0], h_range[0]],
                          scale=[(w_range[-1] - w_range[0])/w,
                                 (h_range[-1] - h_range[0])/h])

            self.setLabel('bottom', data.bin.label1)
            self.setLabel('left', data.bin.label2)
