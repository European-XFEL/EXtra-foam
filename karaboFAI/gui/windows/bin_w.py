"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Bin1dWindow

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..plot_widgets import Bin1dHist, Bin1dHeatmap, Bin2dHeatmap
from ...config import config


class Bin1dWindow(DockerWindow):
    """Bin1dWindow class.

    Plot data in selected bins.
    """
    title = "binning 1D"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _W = _TOTAL_W / 2
    _H1 = _TOTAL_H / 2
    _H2 = 1 * _TOTAL_H / 4
    _H3 = 1 * _TOTAL_H / 4

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._count1 = Bin1dHist(1, count=True, parent=self)
        self._fom1 = Bin1dHist(1, parent=self)
        self._value1_hm = Bin1dHeatmap(1, parent=self)

        self._count2 = Bin1dHist(2, count=True, parent=self)
        self._fom2 = Bin1dHist(2, parent=self)
        self._value2_hm = Bin1dHeatmap(2, parent=self)

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
        # Left
        # -----------

        heatmap1_dock = Dock("VFOM heatmap 1", size=(self._W, self._H1))
        self._docker_area.addDock(heatmap1_dock, 'top')
        heatmap1_dock.addWidget(self._value1_hm)

        fom1_dock = Dock("FOM histogram 1", size=(self._W, self._H2))
        self._docker_area.addDock(fom1_dock, 'bottom', heatmap1_dock)
        fom1_dock.addWidget(self._fom1)

        count1_dock = Dock("Count 1", size=(self._W, self._H3))
        self._docker_area.addDock(count1_dock, 'bottom', fom1_dock)
        count1_dock.addWidget(self._count1)

        # -----------
        # Right
        # -----------

        heatmap2_dock = Dock("VFOM heatmap 2", size=(self._W, self._H1))
        self._docker_area.addDock(heatmap2_dock, 'right')
        heatmap2_dock.addWidget(self._value2_hm)

        fom2_dock = Dock("FOM histogram 2", size=(self._W, self._H2))
        self._docker_area.addDock(fom2_dock, 'bottom', heatmap2_dock)
        fom2_dock.addWidget(self._fom2)

        count2_dock = Dock("Count 2", size=(self._W, self._H3))
        self._docker_area.addDock(count2_dock, 'bottom', fom2_dock)
        count2_dock.addWidget(self._count2)


class Bin2dWindow(DockerWindow):
    """Bin2dWindow class.

    Plot data in selected bins.
    """
    title = "binning 2D"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _W = _TOTAL_W / 2
    _H = _TOTAL_H / 2

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._bin2d_count = Bin2dHeatmap(count=True, parent=self)
        self._bin2d_value = Bin2dHeatmap(count=False, parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        bin2d_count_dock = Dock("Count", size=(self._W, self._H))
        self._docker_area.addDock(bin2d_count_dock, 'right')
        bin2d_count_dock.addWidget(self._bin2d_count)

        bin2d_value_dock = Dock("FOM", size=(self._W, self._H))
        self._docker_area.addDock(bin2d_value_dock, 'top', bin2d_count_dock)
        bin2d_value_dock.addWidget(self._bin2d_value)
