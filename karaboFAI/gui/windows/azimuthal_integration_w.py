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
    title = "Azimuthal Integration"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._ai = TrainAiWidget(parent=self)

        self.initUI()

        self.resize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

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
