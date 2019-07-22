"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

StatisticsWindow

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..plot_widgets import PulsesInTrainFomWidget
from ...config import config


class StatisticsWindow(DockerWindow):
    """StatisticsWindow class.

    Visualize statistics.
    """
    title = "statistics"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _W = _TOTAL_W / 2
    _H = _TOTAL_H / 2

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._plot1 = PulsesInTrainFomWidget(parent=self)
        self._plot2 = PulsesInTrainFomWidget(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        dock1 = Dock("Analysis type 1", size=(self._W, self._H))
        self._docker_area.addDock(dock1, 'right')
        dock1.addWidget(self._plot1)

        dock2 = Dock("Analysis type 2", size=(self._W, self._H))
        self._docker_area.addDock(dock2, 'top', dock1)
        dock2.addWidget(self._plot2)
