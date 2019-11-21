"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractPlotWindow
from ..plot_widgets import RoiImageView
from ...config import config


class RoiWindow(_AbstractPlotWindow):
    """RoiWindow class."""
    _title = "ROI"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._roi1_image = RoiImageView(1, parent=self)
        self._roi2_image = RoiImageView(2, parent=self)

        self.initUI()
        self.initConnections()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        self._cw = QSplitter()
        self._cw.addWidget(self._roi1_image)
        self._cw.addWidget(self._roi2_image)
        self.setCentralWidget(self._cw)

    def initConnections(self):
        """Override."""
        pass
