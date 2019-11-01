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
from ..plot_widgets import (
    PumpProbeImageView, PumpProbeOnOffWidget, PumpProbeFomWidget
)
from ...config import config


class PumpProbeWindow(PlotWindow):
    """PumpProbeWindow class."""
    title = "pump-probe"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._on_image = PumpProbeImageView(on=True, parent=self)
        self._off_image = PumpProbeImageView(on=False, parent=self)

        self._pp_fom = PumpProbeFomWidget(parent=self)
        self._pp_ai = PumpProbeOnOffWidget(parent=self)
        self._pp_diff = PumpProbeOnOffWidget(diff=True, parent=self)

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
        self._cw.setSizes([1, 1])
        self.setCentralWidget(self._cw)

        left_panel.addWidget(self._on_image)
        left_panel.addWidget(self._off_image)

        right_panel.addWidget(self._pp_ai)
        right_panel.addWidget(self._pp_diff)
        right_panel.addWidget(self._pp_fom)
