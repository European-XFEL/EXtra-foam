"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Bin1DWindow

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..plot_widgets import BinCountWidget, BinWidget
from ...config import config


class Bin1DWindow(DockerWindow):
    """Bin1DWindow class.

    Plot data in selected bins.
    """
    title = "binning"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _UW = _TOTAL_W
    _UH = 3 * _TOTAL_H / 4
    _BW = _TOTAL_W
    _BH = _TOTAL_H / 4

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._bin_count = BinCountWidget(parent=self)

        self._value_plot = BinWidget(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        # -----------
        # upper
        # -----------

        value_plot = Dock("Value plot", size=(self._UW, self._UH))
        self._docker_area.addDock(value_plot, 'right')
        value_plot.addWidget(self._value_plot)

        # -----------
        # bottom
        # -----------

        bin_count_dock = Dock("Bin count", size=(self._BW, self._BH))
        self._docker_area.addDock(bin_count_dock, 'bottom', value_plot)
        bin_count_dock.addWidget(self._bin_count)
