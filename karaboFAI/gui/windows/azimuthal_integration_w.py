"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AzimuthalIntegrationWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..plot_widgets import TrainAiWidget
from ...config import config


class AzimuthalIntegrationWindow(DockerWindow):
    """AzimuthalIntegrationWindow class."""
    title = "azimuthal integration"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _LW = 0.5 * _TOTAL_W
    _LH = 0.5 * _TOTAL_H
    _RW = 0.5 * _TOTAL_W
    _RH1 = 0.3 * _TOTAL_H
    _RH2 = 0.4 * _TOTAL_H

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._ai = TrainAiWidget(parent=self)
        self.resize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.initConnections()
        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""

        ai_dock = Dock("Normalized azimuthal Integration",
                       size=(self._TOTAL_W, self._TOTAL_H))
        self._docker_area.addDock(ai_dock)
        ai_dock.addWidget(self._ai)
