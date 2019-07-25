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
from ..plot_widgets import PulsesInTrainFomWidget, FomHistogramWidget
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

        self._pulse_fom = PulsesInTrainFomWidget(parent=self)
        self._fom_historgram = FomHistogramWidget(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        dock1 = Dock("Pulse resolved FOM", size=(self._W, self._H),
                     hideTitle=True)
        self._docker_area.addDock(dock1)
        dock1.addWidget(self._pulse_fom)

        dock2 = Dock("FOM Histogram", size=(self._W, self._H),
                     hideTitle=True)
        self._docker_area.addDock(dock2, 'bottom')
        dock2.addWidget(self._fom_historgram)
