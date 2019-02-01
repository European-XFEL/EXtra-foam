"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageView and SinglePulseImageView.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from ..widgets.pyqtgraph import QtGui, QtCore, HistogramLUTWidget, ImageItem

from .misc_widgets import colorMapFactory
from .plot_widget import PlotWidget
from ..data_processing import quick_min_max
from ..logger import logger
from ..config import config


class ImageView(QtGui.QWidget):
    """ImageView class.

    Widget used for displaying and analyzing a single image.

    Note: this ImageView widget is different from the one implemented
          in pyqtgraph!!!
    """
    def __init__(self, *, parent=None, level_mode='mono'):
        """Initialization.

        :param str level_mode: 'mono' or 'rgba'. If 'mono', then only
            a single set of black/white level lines is drawn, and the
            levels apply to all channels in the image. If 'rgba', then
            one set of levels is drawn for each channel.
        """
        super().__init__(parent=parent)
        try:
            parent.registerPlotWidget(self)
        except AttributeError:
            pass

        self._plot_widget = PlotWidget()
        self._image_item = ImageItem(border='w')
        self._plot_widget.addItem(self._image_item)
        self.invertY(True)
        self.setAspectLocked(True)

        self._hist_widget = HistogramLUTWidget()
        self._hist_widget.setLevelMode(level_mode)
        self._hist_widget.setImageItem(self._image_item)

        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

        self._is_initialized = False
        self._image = None
        self._image_levels = None

        self.initUI()
        # TODO: logarithmic level

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._plot_widget)
        layout.addWidget(self._hist_widget)
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

    def update(self, data):
        """karaboFAI interface."""
        self.setImage(data.image_mean,
                      auto_range=False,
                      auto_levels=(not self._is_initialized))

        if not self._is_initialized:
            self._is_initialized = True

    def setImage(self, img, *, auto_range=True, auto_levels=True):
        """Set the current displayed image.

        :param np.ndarray img: the image to be displayed.
        :param bool auto_range: whether to scale/pan the view to fit
            the image.
        :param bool auto_levels: whether to update the white/black levels
            to fit the image.
        """
        self._image = img

        self._image_item.setImage(self._image, autoLevels=False)

        if auto_levels:
            self._image_levels = quick_min_max(self._image)
            self.setLevels(rgba=[self._image_levels])

        if auto_range:
            self._plot_widget.plotItem.vb.autoRange()

    def setLevels(self, *args, **kwargs):
        """Set the min/max (bright and dark) levels.

        See HistogramLUTItem.setLevels.
        """
        self._hist_widget.setLevels(*args, **kwargs)

    def clear(self):
        self._image_item.clear()

    def setColorMap(self, cm):
        """Set colormap for the displayed image.

        :param cm: a ColorMap object.
        """
        self._hist_widget.gradient.setColorMap(cm)

    def setAspectLocked(self, *args, **kwargs):
        self._plot_widget.setAspectLocked(*args, **kwargs)

    def invertY(self, *args, **kwargs):
        self._plot_widget.plotItem.invertY(*args, **kwargs)

    def addItem(self, *args, **kwargs):
        self._plot_widget.addItem(*args, **kwargs)

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()


class SinglePulseImageView(ImageView):
    """SinglePulseImageView class.

    Widget used for displaying the assembled image for a single pulse.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        parent.registerPlotWidget(self)

        self.pulse_id = 0

        self._mask_range_sp = None
        parent.parent().analysis_ctrl_widget.mask_range_sgn.connect(
            self.onMaskRangeChanged)

        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def update(self, data):
        """Override."""
        image = data.image

        try:
            np.clip(image[self.pulse_id], *self._mask_range_sp,
                    image[self.pulse_id])
        except IndexError as e:
            logger.error("<VIP pulse ID 1/2>: " + str(e))
            return

        self.setImage(image[self.pulse_id],
                      auto_range=False,
                      auto_levels=(not self._is_initialized))

        if not self._is_initialized:
            self._is_initialized = True

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()

    @QtCore.pyqtSlot(float, float)
    def onMaskRangeChanged(self, lb, ub):
        self._mask_range_sp = (lb, ub)
