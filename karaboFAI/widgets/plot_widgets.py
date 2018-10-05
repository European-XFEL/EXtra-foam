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
    GraphicsLayoutWidget, ImageItem, intColor, mkPen
)
from ..config import Config as cfg
from .misc_widgets import ColorMapFactory


class MainGuiLinePlotWidget(GraphicsLayoutWidget):
    def __init__(self, data, parent=None, **kwargs):
        """Initialization."""
        super().__init__(parent, **kwargs)

        self._data = data

        w = cfg.MAIN_WINDOW_WIDTH - cfg.MAIN_LINE_PLOT_HEIGHT - 25
        h = cfg.MAIN_LINE_PLOT_HEIGHT
        self.setFixedSize(w, h)

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


class MainGuiImageViewWidget(GraphicsLayoutWidget):

    def __init__(self, data, parent=None, **kwargs):
        """Initialization."""
        super().__init__(parent, **kwargs)

        self._data = data

        self.setFixedSize(cfg.MAIN_LINE_PLOT_HEIGHT, cfg.MAIN_LINE_PLOT_HEIGHT)

        self._img = ImageItem(border='w')
        self._img.setLookupTable(ColorMapFactory.thermal.getLookupTable())

        self._view = self.addViewBox(lockAspect=True)
        self._view.addItem(self._img)

    def clearPlots(self):
        self._img.clear()

    def updatePlots(self):
        data = self._data.get()

        self._img.setImage(np.flip(data.image_mean, axis=0))
        self._view.autoRange()
