"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtGui, QtCore, QtWidgets

from .base_window import _AbstractPlotWindow
from ..plot_widgets import CorrelationWidget
from ...config import config


class CorrelationWindow(_AbstractPlotWindow):
    """CorrelationWindow class.

    Plot correlations between different parameters.
    """
    title = "correlation"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._corr1 = CorrelationWidget(1, parent=self)
        self._corr2 = CorrelationWidget(2, parent=self)
        self._corr3 = CorrelationWidget(3, parent=self)
        self._corr4 = CorrelationWidget(4, parent=self)

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

        left_panel.addWidget(self._corr1)
        left_panel.addWidget(self._corr3)

        right_panel.addWidget(self._corr2)
        right_panel.addWidget(self._corr4)

    def initConnections(self):
        """Override."""
        pass

    def updateMetaData(self):
        """Override."""
        return True
