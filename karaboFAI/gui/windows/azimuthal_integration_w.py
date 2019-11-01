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
from ..plot_widgets import TrainAiWidget
from ...config import config


class AzimuthalIntegrationWindow(PlotWindow):
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
        self._cw = QtWidgets.QSplitter()
        self._cw.addWidget(self._ai)
        self.setCentralWidget(self._cw)
