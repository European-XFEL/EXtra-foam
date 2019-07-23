"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Bin1dWindow and Bin2dWindow.

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

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._count1 = Bin1dHist(1, count=True, parent=self)
        self._fom1 = Bin1dHist(1, parent=self)
        self._vfom1 = Bin1dHeatmap(1, parent=self)

        self._count2 = Bin1dHist(2, count=True, parent=self)
        self._fom2 = Bin1dHist(2, parent=self)
        self._vfom2 = Bin1dHeatmap(2, parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        w = self._TOTAL_W / 2
        h1 = self._TOTAL_H / 2
        h2 = self._TOTAL_H / 4
        h3 = self._TOTAL_H / 4

        # -----------
        # Left
        # -----------

        vfom1_dock = Dock("Heatmap 1", size=(w, h1), hideTitle=True)
        self._docker_area.addDock(vfom1_dock, 'top')
        vfom1_dock.addWidget(self._vfom1)

        fom1_dock = Dock("Histogram 1", size=(w, h2), hideTitle=True)
        self._docker_area.addDock(fom1_dock, 'bottom', vfom1_dock)
        fom1_dock.addWidget(self._fom1)

        count1_dock = Dock("Count 1", size=(w, h3), hideTitle=True)
        self._docker_area.addDock(count1_dock, 'bottom', fom1_dock)
        count1_dock.addWidget(self._count1)

        # -----------
        # Right
        # -----------

        vfom2_dock = Dock("Heatmap 2", size=(w, h1), hideTitle=True)
        self._docker_area.addDock(vfom2_dock, 'right')
        vfom2_dock.addWidget(self._vfom2)

        fom2_dock = Dock("Histogram 2", size=(w, h2), hideTitle=True)
        self._docker_area.addDock(fom2_dock, 'bottom', vfom2_dock)
        fom2_dock.addWidget(self._fom2)

        count2_dock = Dock("Count 2", size=(w, h3), hideTitle=True)
        self._docker_area.addDock(count2_dock, 'bottom', fom2_dock)
        count2_dock.addWidget(self._count2)


class Bin2dWindow(DockerWindow):
    """Bin2dWindow class.

    Plot data in selected bins.
    """
    title = "binning 2D"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

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
        w = self._TOTAL_W / 2
        h = self._TOTAL_H / 2

        bin2d_count_dock = Dock("Count", size=(w, h), hideTitle=True)
        self._docker_area.addDock(bin2d_count_dock, 'right')
        bin2d_count_dock.addWidget(self._bin2d_count)

        bin2d_value_dock = Dock("FOM", size=(w, h), hideTitle=True)
        self._docker_area.addDock(bin2d_value_dock, 'top', bin2d_count_dock)
        bin2d_value_dock.addWidget(self._bin2d_value)
