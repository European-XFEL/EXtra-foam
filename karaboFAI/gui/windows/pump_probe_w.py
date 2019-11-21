"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractPlotWindow
from ..plot_widgets import (
    PumpProbeImageView, PumpProbeOnOffWidget, PumpProbeFomWidget
)
from ...config import config


class PumpProbeWindow(_AbstractPlotWindow):
    """PumpProbeWindow class."""
    _title = "Pump-probe"

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
        self._cw = QSplitter()
        left_panel = QSplitter(Qt.Vertical)
        right_panel = QSplitter(Qt.Vertical)
        self._cw.addWidget(left_panel)
        self._cw.addWidget(right_panel)
        self._cw.setSizes([1, 1])
        self.setCentralWidget(self._cw)

        left_panel.addWidget(self._on_image)
        left_panel.addWidget(self._off_image)

        right_panel.addWidget(self._pp_ai)
        right_panel.addWidget(self._pp_diff)
        right_panel.addWidget(self._pp_fom)

    def initConnections(self):
        """Override."""
        pass
