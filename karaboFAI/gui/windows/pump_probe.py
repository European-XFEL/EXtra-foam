"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..bulletin_widget import BulletinWidget
from ..plot_widgets import (
    AssembledImageView, LaserOnOffAiWidget, LaserOnOffDiffWidget,
    LaserOnOffFomWidget
)


class PumpProbeWindow(DockerWindow):
    """PumpProbeWindow class."""
    title = "pump-probe"

    _TOTAL_W = 1500
    _TOTAL_H = 1000

    # There are two columns of plots in the PumpProbeWindow. They are
    # numbered at 1, 2, ... from top to bottom.
    _LW = 0.4 * _TOTAL_W
    _LH1 = 0.5 * _TOTAL_H
    _LH2 = 0.25 * _TOTAL_H
    _LH3 = 0.25 * _TOTAL_H
    _RW = 0.6 * _TOTAL_W
    _RH1 = 0.5 * _TOTAL_H - 25
    _RH2 = 50
    _RH3 = 0.5 * _TOTAL_H - 25

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._on_image = AssembledImageView(parent=self)
        self._off_image = AssembledImageView(parent=self)

        self._bulletin = BulletinWidget(parent=self)
        self._bulletin.setMaximumHeight(self._RH2)

        self._on_off_fom = LaserOnOffFomWidget(parent=self)
        self._on_off_ai = LaserOnOffAiWidget(parent=self)
        self._on_off_diff = LaserOnOffDiffWidget(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        # -----------
        # left
        # -----------

        on_image_dock = Dock("Laser-on Image", size=(self._LW, self._LH1))
        self._docker_area.addDock(on_image_dock, "left")
        on_image_dock.addWidget(self._on_image)

        off_image_dock = Dock("Laser-off Image", size=(self._LW, self._LH1))
        self._docker_area.addDock(off_image_dock, 'bottom', on_image_dock)
        off_image_dock.addWidget(self._off_image)

        # -----------
        # right
        # -----------

        on_off_ai_dock = Dock("On&Off Azimuthal Integration",
                              size=(self._RW, self._RH1))
        self._docker_area.addDock(on_off_ai_dock, 'right')
        on_off_ai_dock.addWidget(self._on_off_ai)

        on_off_diff_dock = Dock("On-Off Azimuthal Integration",
                                size=(self._RW, self._RH1))
        self._docker_area.addDock(on_off_diff_dock, 'bottom', on_off_ai_dock)
        on_off_diff_dock.addWidget(self._on_off_diff)

        bulletin_dock = Dock("Bulletin", size=(self._RW, self._RH2))
        self._docker_area.addDock(bulletin_dock, 'bottom', on_off_diff_dock)
        bulletin_dock.addWidget(self._bulletin)
        bulletin_dock.hideTitleBar()

        on_off_fom_dock = Dock("FOM", size=(self._RW, self._RH3))
        self._docker_area.addDock(on_off_fom_dock, 'bottom', bulletin_dock)
        on_off_fom_dock.addWidget(self._on_off_fom)
