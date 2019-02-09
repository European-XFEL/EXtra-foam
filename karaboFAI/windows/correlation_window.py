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

    _n_plots = 4

    _WIDGET_W = 600
    _WIDGET_H = 450

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._plots = []
        for i in range(self._n_plots):
            self._plots.append(CorrelationWidget(parent=self))

        self.initUI()

        self.resize(self._WIDGET_W * 2, self._WIDGET_H * 2)

        self.update()

        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        docks = []
        for i, widget in enumerate(self._plots, 1):
            dock = Dock(f'correlation {i}', closable=True)
            docks.append(dock)
            self._docker_area.addDock(dock, 'right')
            dock.addWidget(widget)

        self._docker_area.moveDock(docks[2], 'bottom', docks[0])
        self._docker_area.moveDock(docks[3], 'bottom', docks[1])
