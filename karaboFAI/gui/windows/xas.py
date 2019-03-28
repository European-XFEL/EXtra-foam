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
    RoiImageView, XasSpectrumWidget, XasSpectrumDiffWidget
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
    _LH = _TOTAL_H / len(config["ROI_COLORS"])
    _RW = 0.6 * _TOTAL_W
    _RH1 = 0.5 * _TOTAL_H - 25
    _RH2 = 50
    _RH3 = 0.5 * _TOTAL_H - 25

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._roi_images = []
        for i, color in enumerate(config["ROI_COLORS"], 1):
            view = RoiImageView(i, parent=self)
            view.setBorder(make_pen(color))
            self._roi_images.append(view)

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
        prev_roi_image_dock = None
        for i, image in enumerate(self._roi_images, 1):
            roi_image_dock = Dock(f"ROI{i}", size=(self._LW, self._LH))
            if prev_roi_image_dock is None:
                self._docker_area.addDock(roi_image_dock, 'left')
            else:
                self._docker_area.addDock(
                    roi_image_dock, 'bottom', prev_roi_image_dock)
            prev_roi_image_dock = roi_image_dock
            roi_image_dock.addWidget(image)

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
