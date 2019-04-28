"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

OverviewWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..mediator import Mediator
from ..bulletin_widget import BulletinWidget
from ..misc_widgets import make_pen
from ..plot_widgets import (
    AssembledImageView, MultiPulseAiWidget, RoiImageView, RoiValueMonitor,
    PulseResolvedAiFomWidget, SinglePulseAiWidget, SinglePulseImageView
)
from ...config import config


class OverviewWindow(DockerWindow):
    """OverviewWindow class."""
    title = "overview"

    _TOTAL_W = 1500
    _TOTAL_H = 1000

    # There are two columns of plots in the OverviewWindow. They are
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

        mediator = Mediator()

        self._bulletin = BulletinWidget(parent=self)
        self._bulletin.setMaximumHeight(self._RH2)
        self._assembled = AssembledImageView(parent=self)

        if self._pulse_resolved:
            self._ai = MultiPulseAiWidget(parent=self)

            self._sample_degradation = PulseResolvedAiFomWidget(parent=self)

            self._vip1_ai_dock = None
            self._vip1_img_dock = None
            self._vip1_ai = SinglePulseAiWidget(parent=self)
            self._vip1_img = SinglePulseImageView(parent=self)

            self._vip2_ai_dock = None
            self._vip2_img_dock = None
            self._vip2_ai = SinglePulseAiWidget(parent=self)
            self._vip2_img = SinglePulseImageView(parent=self)

            mediator.vip_pulse_id1_sgn.connect(self.onPulseID1Updated)
            mediator.vip_pulse_id2_sgn.connect(self.onPulseID2Updated)
        else:
            self._ai = SinglePulseAiWidget(parent=self, plot_mean=False)

        self._roi1_image = RoiImageView(1, parent=self)
        self._roi1_image.setBorder(make_pen(config["ROI_COLORS"][0]))
        self._roi2_image = RoiImageView(2, parent=self)
        self._roi2_image.setBorder(make_pen(config["ROI_COLORS"][1]))

        self._roi_intensity = RoiValueMonitor(parent=self)
        mediator.roi_displayed_range_sgn.connect(
            self._roi_intensity.onDisplayRangeChange)

        self.initUI()

        mediator.updateVipPulseIds()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""

        # ----------
        # upper left
        # ----------

        assembled_dock = Dock("Mean Assembled Image",
                              size=(self._LW, self._LH1))
        self._docker_area.addDock(assembled_dock, 'left')
        assembled_dock.addWidget(self._assembled)

        # -----------
        # lower left
        # -----------

        roi1_image_dock = Dock("ROI1", size=(self._LW, self._LH2))
        self._docker_area.addDock(roi1_image_dock, 'bottom', assembled_dock)
        roi1_image_dock.addWidget(self._roi1_image)

        roi2_image_dock = Dock("ROI2", size=(self._LW, self._LH3))
        self._docker_area.addDock(roi2_image_dock, 'bottom', roi1_image_dock)
        roi2_image_dock.addWidget(self._roi2_image)

        if self._pulse_resolved:
            self._vip2_ai_dock = Dock("VIP pulse 0000 - AI",
                                      size=(self._LW, self._LH3))
            self._docker_area.addDock(
                self._vip2_ai_dock, 'above', roi2_image_dock)
            self._vip2_ai_dock.addWidget(self._vip2_ai)

            self._vip1_ai_dock = Dock("VIP pulse 0000 - AI",
                                      size=(self._LW, self._LH2))
            self._docker_area.addDock(
                self._vip1_ai_dock, 'above', roi1_image_dock)
            self._vip1_ai_dock.addWidget(self._vip1_ai)

            self._vip2_img_dock = Dock("VIP pulse 0000",
                                       size=(self._LW, self._LH3))
            self._docker_area.addDock(self._vip2_img_dock, 'above',
                                      self._vip2_ai_dock)
            self._vip2_img_dock.addWidget(self._vip2_img)

            self._vip1_img_dock = Dock("VIP pulse 0000",
                                       size=(self._LW, self._LH2))
            self._docker_area.addDock(self._vip1_img_dock, 'above',
                                      self._vip1_ai_dock)
            self._vip1_img_dock.addWidget(self._vip1_img)

        # -----------
        # upper right
        # -----------

        ai_dock = Dock("Normalized azimuthal Integration",
                       size=(self._RW, self._RH1))
        self._docker_area.addDock(ai_dock, 'right')
        ai_dock.addWidget(self._ai)

        # -----------
        # middle right
        # -----------

        bulletin_dock = Dock("Bulletin", size=(self._RW, self._RH2))
        self._docker_area.addDock(bulletin_dock, 'bottom', ai_dock)
        bulletin_dock.addWidget(self._bulletin)
        bulletin_dock.hideTitleBar()

        # -----------
        # lower right
        # -----------

        roi_intensity_dock = Dock("ROI intensity", size=(self._RW, self._RH3))
        self._docker_area.addDock(roi_intensity_dock, 'bottom', bulletin_dock)
        roi_intensity_dock.addWidget(self._roi_intensity)

        if self._pulse_resolved:
            sample_degradation_dock = Dock(
                "Pulse-resolved FOM", size=(self._RW, self._RH3))
            self._docker_area.addDock(
                sample_degradation_dock, 'above', roi_intensity_dock)
            sample_degradation_dock.addWidget(self._sample_degradation)

    @QtCore.pyqtSlot(int)
    def onPulseID1Updated(self, value):
        self._vip1_ai_dock.setTitle("VIP pulse {:04d} - AI".format(value))
        self._vip1_img_dock.setTitle("VIP pulse {:04d}".format(value))
        self._vip1_ai.pulse_id = value
        self._vip1_img.pulse_id = value

    @QtCore.pyqtSlot(int)
    def onPulseID2Updated(self, value):
        self._vip2_ai_dock.setTitle("VIP pulse {:04d} - AI".format(value))
        self._vip2_img_dock.setTitle("VIP pulse {:04d}".format(value))
        self._vip2_ai.pulse_id = value
        self._vip2_img.pulse_id = value
