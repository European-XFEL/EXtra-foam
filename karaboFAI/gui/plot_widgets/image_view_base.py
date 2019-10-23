"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from PyQt5 import QtCore, QtGui

from .. import pyqtgraph as pg
from ..pyqtgraph import HistogramLUTWidget

from .plot_widget_base import PlotWidgetF
from .plot_items import RectROI
from ..misc_widgets import colorMapFactory, make_pen
from ..mediator import Mediator
from ...algorithms import quick_min_max
from ...config import config


class ImageViewF(QtGui.QWidget):
    """ImageView base class.

    A widget used for displaying 2D image data.

    * Four ROIs are included in this widget by default.

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

        self._plot_widget = PlotWidgetF()
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

    def setTitle(self, *args, **kwargs):
        self._plot_widget.setTitle(*args, **kwargs)

    def invertY(self, *args, **kwargs):
        self._plot_widget.plotItem.invertY(*args, **kwargs)

    def addItem(self, *args, **kwargs):
        self._plot_widget.addItem(*args, **kwargs)

    def removeItem(self, *args, **kwargs):
        self._plot_widget.removeItem(*args, **kwargs)

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()
