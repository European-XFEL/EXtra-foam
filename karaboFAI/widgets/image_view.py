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

from ..widgets.pyqtgraph import QtGui, QtCore, ImageItem

from .misc_widgets import colorMapFactory
from .plot_widget import PlotWidget
from ..logger import logger
from ..config import config


class ImageView(QtGui.QWidget):
    """ImageView class.

    Widget used for displaying and analyzing a single image.

    Note: this ImageView widget is different from the one implemented
          in pyqtgraph!!!
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        parent.registerPlotWidget(self)

        self._plot_widget = PlotWidget()
        self._image_item = ImageItem(border='w')
        self._plot_widget.addItem(self._image_item)

        self._is_initialized = False

        self.initUI()
        # TODO: logarithmic level
        # self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._plot_widget)
        self.setLayout(layout)

    def update(self, data):
        """karaboFAI interface."""
        self.updateImage(data.image_mean,
                         auto_levels=(not self._is_initialized))

        if not self._is_initialized:
            self._is_initialized = True

    def updateImage(self, img, auto_levels=False):
        """Update the current displayed image."""
        self._image_item.setImage(img, auto_levels=auto_levels)

    def clear(self):
        self._image_item.clear()

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

        self._is_initialized = False

        self._mask_range_sp = None
        parent.parent().analysis_ctrl_widget.mask_range_sgn.connect(
            self.onMaskRangeChanged)

        # self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def update(self, data):
        """Override."""
        image = data.image

        try:
            np.clip(image[self.pulse_id], *self._mask_range_sp,
                    image[self.pulse_id])
        except IndexError as e:
            logger.error("<VIP pulse ID 1/2>: " + str(e))
            return

        self.updateImage(image[self.pulse_id],
                         auto_levels=(not self._is_initialized))

        if not self._is_initialized:
            self._is_initialized = True

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()

    @QtCore.pyqtSlot(float, float)
    def onMaskRangeChanged(self, lb, ub):
        self._mask_range_sp = (lb, ub)
