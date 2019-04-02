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
    AssembledImageView, RoiImageView, XasSpectrumWidget, XasSpectrumDiffWidget,
    XasSpectrumBinCountWidget
)
from ...config import config


class XasWindow(DockerWindow):
    """OverviewWindow class."""
    title = "XAS"

    _TOTAL_W = 1600
    _TOTAL_H = 1000

    # There are two columns of plots in the PumpProbeWindow. They are
    # numbered at 1, 2, ... from top to bottom.
    _LW = 0.25 * _TOTAL_W
    _LH = _TOTAL_H / 3
    _MW = 0.25 * _TOTAL_W
    _MH1 = 0.80 * _TOTAL_H
    _MH2 = 0.20 * _TOTAL_H
    _RW = 0.50 * _TOTAL_W
    _RH = _TOTAL_H / 3

    _n_spectra = 2

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._assembled = AssembledImageView(parent=self)

        self._roi_images = []
        for i, color in enumerate(config["ROI_COLORS"][:self._n_spectra+1], 1):
            view = RoiImageView(i, parent=self)
            view.setBorder(make_pen(color))
            self._roi_images.append(view)

        self._bulletin = BulletinWidget(vertical=True, parent=self)

        self._spectrum = XasSpectrumWidget(parent=self)
        self._spectrum_diff = XasSpectrumDiffWidget(parent=self)
        self._spectrum_count = XasSpectrumBinCountWidget(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(800, 500)

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
            roi_image_dock = Dock(f"ROI{i} (I{i-1})", size=(self._LW, self._LH))
            if prev_roi_image_dock is None:
                self._docker_area.addDock(roi_image_dock, 'left')
            else:
                self._docker_area.addDock(
                    roi_image_dock, 'bottom', prev_roi_image_dock)
            prev_roi_image_dock = roi_image_dock
            roi_image_dock.addWidget(image)

        # -----------
        # middle
        # -----------

        assembled_dock = Dock("Assembled Image", size=(self._MW, self._MH1))
        self._docker_area.addDock(assembled_dock, 'right')
        assembled_dock.addWidget(self._assembled)

        bulletin_dock = Dock("Bulletin", size=(self._MW, self._MH2))
        self._docker_area.addDock(bulletin_dock, 'bottom', assembled_dock)
        bulletin_dock.addWidget(self._bulletin)
        bulletin_dock.hideTitleBar()

        # -----------
        # right
        # -----------

        spectrum_dock = Dock("Spectra", size=(self._RW, self._RH))
        self._docker_area.addDock(spectrum_dock, 'right')
        spectrum_dock.addWidget(self._spectrum)

        spectrum_diff_dock = Dock("Difference of spectra",
                                  size=(self._RW, self._RH))
        self._docker_area.addDock(spectrum_diff_dock, 'bottom', spectrum_dock)
        spectrum_diff_dock.addWidget(self._spectrum_diff)

        spectrum_count_dock = Dock("Bin count", size=(self._RW, self._RH))
        self._docker_area.addDock(
            spectrum_count_dock, 'bottom', spectrum_diff_dock)
        spectrum_count_dock.addWidget(self._spectrum_count)
