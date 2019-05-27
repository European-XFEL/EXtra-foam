"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

OverviewWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..bulletin_widget import BulletinWidget
from ..plot_widgets import AssembledImageView
from ...config import config


class OverviewWindow(DockerWindow):
    """OverviewWindow class."""
    title = "overview"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._bulletin = BulletinWidget(parent=self)
        self._assembled = AssembledImageView(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""

        bulletin_dock = Dock("Bulletin",
                             size=(self._TOTAL_W, 0.1*self._TOTAL_H))
        self._docker_area.addDock(bulletin_dock)
        bulletin_dock.addWidget(self._bulletin)
        bulletin_dock.hideTitleBar()

        assembled_dock = Dock("Mean Assembled Image",
                              size=(self._TOTAL_W, 0.9*self._TOTAL_H))
        self._docker_area.addDock(assembled_dock, 'bottom', bulletin_dock)
        assembled_dock.addWidget(self._assembled)
