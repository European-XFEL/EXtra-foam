"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

BinningWindow

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..plot_widgets import BinningImageView, BinningCountWidget, BinningWidget
from ...config import config


class BinningWindow(DockerWindow):
    """BinningWindow class.

    Plot data in selected bins.
    """
    title = "binning"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _UW = _TOTAL_W / 3
    _UH = _TOTAL_H / 4
    _MW = _TOTAL_W / 2
    _MH = _TOTAL_H / 2
    _BW = _TOTAL_W
    _BH = _TOTAL_H / 4

    _n_views = len(config['BINNING_COLORS'])

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._images = []
        for i in range(self._n_views):
            self._images.append(BinningImageView(i, parent=self))

        self._bin_count = BinningCountWidget(parent=self)

        self._plot1 = BinningWidget(parent=self)
        self._plot2 = BinningWidget(parent=self)

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
        prev_view_dock = None
        for i, view in enumerate(self._images, 1):
            view_dock = Dock(f"Bin", size=(self._UW, self._UH))
            if prev_view_dock is None:
                self._docker_area.addDock(view_dock, 'left')
            else:
                self._docker_area.addDock(
                    view_dock, 'right', prev_view_dock)
            prev_view_dock = view_dock
            view_dock.addWidget(view)

        # -----------
        # middle
        # -----------

        plot1_dock = Dock("Plot1", size=(self._MW, self._MH))
        self._docker_area.addDock(plot1_dock, 'bottom')
        plot1_dock.addWidget(self._plot1)

        plot2_dock = Dock("Plot2", size=(self._MW, self._MH))
        self._docker_area.addDock(plot2_dock, 'right', plot1_dock)
        plot2_dock.addWidget(self._plot2)

        # -----------
        # bottom
        # -----------

        bin_count_dock = Dock("Bin count", size=(self._BW, self._BH))
        self._docker_area.addDock(bin_count_dock, 'bottom')
        bin_count_dock.addWidget(self._bin_count)
