"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

CorrelationWindow

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph.dockarea import Dock

from .base_window import DockerWindow, SingletonWindow
from ..logger import logger
from ..widgets import CorrelationWidget


@SingletonWindow
class CorrelationWindow(DockerWindow):
    """CorrelationWindow class.

    Plot correlations between different parameters.
    """
    title = "correlation"

    _WIDGET_W = 600
    _WIDGET_H = 600

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._plots = []
        for i in range(2):
            self._plots.append(CorrelationWidget(parent=self))

        self.initUI()

        self.resize(self._WIDGET_W, 2 * self._WIDGET_H)

        self.update()

        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        docks = []
        for i, widget in enumerate(self._plots):
            dock = Dock(f"Correlation {i}", size=(400, 400))
            docks.append(dock)
            self._docker_area.addDock(dock)
            dock.addWidget(widget)
