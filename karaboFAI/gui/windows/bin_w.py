"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtGui, QtCore, QtWidgets

from .base_window import PlotWindow
from ..plot_widgets import Bin1dHist, Bin1dHeatmap, Bin2dHeatmap
from ...config import config


class Bin1dWindow(PlotWindow):
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
        self._cw = QtWidgets.QSplitter()
        left_panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._cw.addWidget(left_panel)
        self._cw.addWidget(right_panel)
        self.setCentralWidget(self._cw)

        left_panel.addWidget(self._vfom1)
        left_panel.addWidget(self._fom1)
        left_panel.addWidget(self._count1)
        # A value smaller than the minimal size hint of the respective
        # widget will be replaced by the value of the hint.
        left_panel.setSizes([self._TOTAL_H, self._TOTAL_H/2, self._TOTAL_H/2])

        right_panel.addWidget(self._vfom2)
        right_panel.addWidget(self._fom2)
        right_panel.addWidget(self._count2)
        right_panel.setSizes([self._TOTAL_H, self._TOTAL_H/2, self._TOTAL_H/2])


class Bin2dWindow(PlotWindow):
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
        self._cw = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._cw.addWidget(self._bin2d_value)
        self._cw.addWidget(self._bin2d_count)
        self._cw.setSizes([1, 1])
        self.setCentralWidget(self._cw)
