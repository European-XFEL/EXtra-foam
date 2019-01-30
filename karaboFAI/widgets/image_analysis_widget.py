"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ImageAnalysisWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from ..widgets.pyqtgraph import ImageView, QtCore

from .misc_widgets import colorMapFactory
from ..logger import logger
from ..config import config


class ImageAnalysisWidget(ImageView):
    """ImageAnalysisWidget class.

    Widget used for displaying and analyzing the mean assembled image.
    TODO: the ImageView widget needs to be customized.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        parent.registerPlotWidget(self)

        self._is_initialized = False
        # TODO: logarithmic level
        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def update(self, data):
        self.setImage(data.image_mean, autoRange=False,
                      autoLevels=(not self._is_initialized))

        if not self._is_initialized:
            self._is_initialized = True

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()


class SinglePulseImageWidget(ImageView):
    """SinglePulseImageWidget class.

    Widget used for displaying the assembled image for a single pulse.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        parent.registerPlotWidget(self)

        self.pulse_id = 0

        self._is_initialized = False

        self._image_mask_range_sp = None
        parent.parent().analysis_ctrl_widget.image_mask_range_sgn.connect(
            self.onMaskRangeChanged)

        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def update(self, data):
        image = data.image

        try:
            np.clip(image[self.pulse_id], *self._image_mask_range_sp,
                    image[self.pulse_id])
        except IndexError as e:
            logger.error("<VIP pulse ID>: " + str(e))
            return

        self.setImage(image[self.pulse_id], autoRange=False,
                      autoLevels=(not self._is_initialized))

        if not self._is_initialized:
            self._is_initialized = True

    def close(self):
        self.parent().unregisterPlotWidget(self)
        super().close()

    @QtCore.pyqtSlot(float, float)
    def onMaskRangeChanged(self, lb, ub):
        self._image_mask_range_sp = (lb, ub)
