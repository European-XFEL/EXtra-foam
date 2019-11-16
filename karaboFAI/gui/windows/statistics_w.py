"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtGui, QtCore, QtWidgets

from .base_window import _AbstractPlotWindow
from ..plot_widgets import PulsesInTrainFomWidget, FomHistogramWidget
from ...config import config


class StatisticsWindow(_AbstractPlotWindow):
    """StatisticsWindow class.

    Visualize statistics.
    """
    _title = "Statistics"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']
    _TOTAL_W /= 2

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
        self._cw = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._cw.addWidget(self._pulse_fom)
        self._cw.addWidget(self._fom_historgram)
        self._cw.setSizes([1, 1])
        self.setCentralWidget(self._cw)

    def initConnections(self):
        """Override."""
        pass

    def updateMetaData(self):
        """Override."""
        return True
