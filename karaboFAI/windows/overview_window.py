"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

OverviewWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph.dockarea import Dock, DockArea
from ..widgets.pyqtgraph import (
    PlotWidget, ImageView, intColor, mkPen, QtGui
)

from .base_window import AbstractWindow
from ..logger import logger
from ..widgets.misc_widgets import colorMapFactory
from ..config import config


class OverviewWindow(AbstractWindow):
    """OverviewWindow class."""

    title = "overview"

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)
        self.parent().registerPlotWidget(self)

        self._docker_area = DockArea()
        self._assembled = ImageView()
        self._multiline = PlotWidget()

        self.initUI()
        self.updatePlots()

        self.resize(1500, 600)
        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        self.initPlotUI()
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._docker_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cw.setLayout(layout)

    def initPlotUI(self):
        """Override."""
        assembled_dock = Dock("Assembled Detector Image", size=(600, 600))
        self._docker_area.addDock(assembled_dock, 'left')
        assembled_dock.addWidget(self._assembled)

        imageview_dock = Dock("Azimuthal Integration", size=(900, 600))
        self._docker_area.addDock(imageview_dock, 'right')
        imageview_dock.addWidget(self._multiline)

        self._assembled.setColorMap(colorMapFactory[config["COLOR_MAP"]])

        self._multiline.setTitle("")
        self._multiline.setLabel('bottom', "Momentum transfer (1/A)")
        self._multiline.setLabel('left', "Scattering signal (arb. u.)")

    def clearPlots(self):
        """Clear plots.

        This method is called by the main GUI.
        """
        self._multiline.clear()
        self._assembled.clear()

    def updatePlots(self):
        """Update plots.

        This method is called by the main GUI.
        """
        data = self._data.get()
        if data.empty():
            return

        self._assembled.setImage(data.image_mean,
                                 autoRange=False, autoLevels=False)

        momentum = data.momentum
        line = self._multiline
        for i, intensity in enumerate(data.intensity):
            line.plot(momentum, intensity,
                      pen=mkPen(intColor(i, hues=9, values=5), width=2))
            line.setTitle("Train ID: {}, number of pulses: {}".
                          format(data.tid, len(data.intensity)))
