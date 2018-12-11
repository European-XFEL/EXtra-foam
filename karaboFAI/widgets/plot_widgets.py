"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold classes of various plot widgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .pyqtgraph import (
    GraphicsLayoutWidget, ImageView, intColor, mkPen
)
from .misc_widgets import colorMapFactory

from ..config import config


class AiMultiLinePlotWidget(GraphicsLayoutWidget):
    def __init__(self, data, parent=None, **kwargs):
        """Initialization."""
        super().__init__(parent=parent, **kwargs)
        self.parent().registerPlotWidget(self)

        self._data = data

        self._plot = self.addPlot()
        self._plot.setTitle("")
        self._plot.setLabel('bottom', "Momentum transfer (1/A)")
        self._plot.setLabel('left', "Scattering signal (arb. u.)")

    def clearPlots(self):
        self._plot.clear()

    def updatePlots(self):
        data = self._data.get()

        momentum = data.momentum
        for i, intensity in enumerate(data.intensity):
            self._plot.plot(momentum, intensity,
                            pen=mkPen(intColor(i, hues=9, values=5), width=2))
        self._plot.setTitle("Train ID: {}, number of pulses: {}".
                            format(data.tid, len(data.intensity)))


class AiImageViewWidget(ImageView):

    def __init__(self, data, parent=None, **kwargs):
        """Initialization."""
        super().__init__(parent=parent, **kwargs)
        self.parent().registerPlotWidget(self)

        self._data = data

        self.setColorMap(colorMapFactory[config["COLOR_MAP"]])

    def clearPlots(self):
        self.clear()

    def updatePlots(self):
        data = self._data.get()

        self.setImage(np.flip(data.image_mean, axis=0), autoRange=False)
