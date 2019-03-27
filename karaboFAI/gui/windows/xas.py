"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

XasWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..bulletin_widget import BulletinWidget
from ..misc_widgets import make_pen
from ..plot_widgets import (
    AssembledImageView, RoiImageView, XasSpectrumWidget, XasSpectrumDiffWidget
)
from ...config import config


class XasWindow(DockerWindow):
    """OverviewWindow class."""
    title = "XAS"

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

        self._assembled = AssembledImageView(parent=self)

        self._roi1_image = RoiImageView(parent=self)
        self._roi1_image.setBorder(make_pen(config["ROI_COLORS"][0]))
        self._roi2_image = RoiImageView(roi1=False, parent=self)
        self._roi2_image.setBorder(make_pen(config["ROI_COLORS"][1]))

        self._bulletin = BulletinWidget(parent=self)
        self._bulletin.setMaximumHeight(self._RH2)

        self._spectrum = XasSpectrumWidget(parent=self)
        self._spectrum_diff = XasSpectrumDiffWidget(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(800, 600)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        # -----------
        # left
        # -----------

        assembled_dock = Dock("Assembled Image", size=(self._LW, self._LH1))
        self._docker_area.addDock(assembled_dock, "left")
        assembled_dock.addWidget(self._assembled)

        roi1_image_dock = Dock("ROI1", size=(self._LW, self._LH2))
        self._docker_area.addDock(roi1_image_dock, 'bottom', assembled_dock)
        roi1_image_dock.addWidget(self._roi1_image)

        roi2_image_dock = Dock("ROI2", size=(self._LW, self._LH3))
        self._docker_area.addDock(roi2_image_dock, 'bottom', roi1_image_dock)
        roi2_image_dock.addWidget(self._roi2_image)

        # -----------
        # right
        # -----------

        spectrum_dock = Dock("Spectra",
                             size=(self._RW, self._RH1))
        self._docker_area.addDock(spectrum_dock, 'right')
        spectrum_dock.addWidget(self._spectrum)

        bulletin_dock = Dock("Bulletin", size=(self._RW, self._RH2))
        self._docker_area.addDock(bulletin_dock, 'bottom', spectrum_dock)
        bulletin_dock.addWidget(self._bulletin)
        bulletin_dock.hideTitleBar()

        spectrum_diff_dock = Dock("Difference of spectra",
                                size=(self._RW, self._RH1))
        self._docker_area.addDock(spectrum_diff_dock, 'bottom', bulletin_dock)
        spectrum_diff_dock.addWidget(self._spectrum_diff)
