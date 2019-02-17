"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

OverviewWindow for pulse-resolved detectors.
OverviewWindowTrainResolved for train-resolved detectors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import QtCore
from ..widgets.pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..logger import logger
from ..widgets import (
    BulletinWidget, AssembledImageView, LaserOnOffAiWidget, LaserOnOffFomWidget,
    make_pen, MultiPulseAiWidget, RoiImageView, RoiValueMonitor,
    SampleDegradationWidget, SinglePulseAiWidget, SinglePulseImageView
)
from ..config import config


class OverviewWindow(DockerWindow):
    """OverviewWindow class."""
    title = "overview"

    _ASSEMBLED_W = 600
    _ASSEMBLED_H = 500
    _AI_W = 900
    _AI_H = _ASSEMBLED_H
    _ON_OFF_AI_W = _AI_W
    _ON_OFF_AI_H = _AI_H
    _SP_W = _ASSEMBLED_W
    _SP_H = 250
    _ROI_W = _SP_W
    _ROI_H = _SP_H
    _BULLETIN_W = _AI_W
    _BULLETIN_H = 100
    _SAMPLE_W = _AI_W
    _SAMPLE_H = 2 * _SP_H - _BULLETIN_H
    _ON_OFF_W = _SAMPLE_W
    _ON_OFF_H = _SAMPLE_H
    _ROI_IT_W = _SAMPLE_W
    _ROI_IT_H = _SAMPLE_H

    _TOTAL_W = _ASSEMBLED_W + _AI_W
    _TOTAL_H = _ASSEMBLED_H + 2 * _SP_H

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._bulletin = BulletinWidget(parent=self)
        self._assembled = AssembledImageView(parent=self)

        if self._pulse_resolved:
            self._ai = MultiPulseAiWidget(parent=self)

            self._sample_degradation = SampleDegradationWidget(parent=self)

            self._vip1_ai_dock = None
            self._vip1_img_dock = None
            self._vip1_ai = SinglePulseAiWidget(parent=self)
            self._vip1_img = SinglePulseImageView(parent=self)

            self._vip2_ai_dock = None
            self._vip2_img_dock = None
            self._vip2_ai = SinglePulseAiWidget(parent=self)
            self._vip2_img = SinglePulseImageView(parent=self)

            self._mediator.vip_pulse_id1_sgn.connect(self.onPulseID1Updated)
            self._mediator.vip_pulse_id2_sgn.connect(self.onPulseID2Updated)
        else:
            self._ai = SinglePulseAiWidget(parent=self, plot_mean=False)

        self._roi1_image = RoiImageView(parent=self)
        self._roi1_image.setBorder(make_pen(config["ROI_COLORS"][0]))
        self._roi2_image = RoiImageView(roi1=False, parent=self)
        self._roi2_image.setBorder(make_pen(config["ROI_COLORS"][1]))

        self._roi_intensity = RoiValueMonitor(parent=self)
        self._mediator.roi_displayed_range_sgn.connect(
            self._roi_intensity.onDisplayRangeChange)

        self._on_off_fom = LaserOnOffFomWidget(parent=self)
        self._on_off_ai = LaserOnOffAiWidget(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_W)

        self.update()

        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""

        # ----------
        # upper left
        # ----------

        assembled_dock = Dock("Mean Assembled Image", size=(
            self._ASSEMBLED_W, self._ASSEMBLED_H))
        self._docker_area.addDock(assembled_dock, 'left')
        assembled_dock.addWidget(self._assembled)

        # -----------
        # lower left
        # -----------

        roi1_image_dock = Dock("ROI1", size=(self._ROI_W, self._ROI_H))
        self._docker_area.addDock(roi1_image_dock, 'bottom', assembled_dock)
        roi1_image_dock.addWidget(self._roi1_image)

        roi2_image_dock = Dock("ROI2", size=(self._ROI_W, self._ROI_H))
        self._docker_area.addDock(roi2_image_dock, 'bottom', roi1_image_dock)
        roi2_image_dock.addWidget(self._roi2_image)

        if self._pulse_resolved:
            #
            self._vip2_ai_dock = Dock("VIP pulse 0000 - AI",
                                      size=(self._SP_W, self._SP_H))
            self._docker_area.addDock(
                self._vip2_ai_dock, 'above', roi2_image_dock)
            self._vip2_ai_dock.addWidget(self._vip2_ai)

            #
            self._vip1_ai_dock = Dock("VIP pulse 0000 - AI",
                                      size=(self._SP_W, self._SP_H))
            self._docker_area.addDock(
                self._vip1_ai_dock, 'above', roi1_image_dock)
            self._vip1_ai_dock.addWidget(self._vip1_ai)

            #
            self._vip2_img_dock = Dock("VIP pulse 0000",
                                       size=(self._SP_W, self._SP_H))
            self._docker_area.addDock(self._vip2_img_dock, 'above',
                                      self._vip2_ai_dock)
            self._vip2_img_dock.addWidget(self._vip2_img)

            #
            self._vip1_img_dock = Dock("VIP pulse 0000",
                                       size=(self._SP_W, self._SP_H))
            self._docker_area.addDock(self._vip1_img_dock, 'above',
                                      self._vip1_ai_dock)
            self._vip1_img_dock.addWidget(self._vip1_img)

        # -----------
        # upper right
        # -----------

        #
        on_off_ai_dock = Dock("Laser On-Off Azimuthal Integration",
                              size=(self._ON_OFF_AI_W, self._ON_OFF_AI_H))
        self._docker_area.addDock(on_off_ai_dock, 'right')
        on_off_ai_dock.addWidget(self._on_off_ai)

        #
        ai_dock = Dock("Azimuthal Integration Overview",
                       size=(self._AI_W, self._AI_H))
        self._docker_area.addDock(ai_dock, 'above', on_off_ai_dock)
        ai_dock.addWidget(self._ai)

        # -----------
        # middle right
        # -----------

        #
        bulletin_dock = Dock("Bulletin",
                             size=(self._BULLETIN_W, self._BULLETIN_H))
        self._docker_area.addDock(bulletin_dock, 'bottom', ai_dock)
        bulletin_dock.addWidget(self._bulletin)
        bulletin_dock.hideTitleBar()

        # -----------
        # lower right
        # -----------

        #
        on_off_fom_dock = Dock("Laser On-Off Azimuthal Integration",
                               size=(self._ON_OFF_W, self._ON_OFF_H))
        self._docker_area.addDock(on_off_fom_dock, 'bottom', bulletin_dock)
        on_off_fom_dock.addWidget(self._on_off_fom)

        #
        roi_intensity_dock = Dock("ROI intensity",
                                  size=(self._ROI_IT_W, self._ROI_IT_H))
        self._docker_area.addDock(roi_intensity_dock, 'above', on_off_fom_dock)
        roi_intensity_dock.addWidget(self._roi_intensity)

        #
        if self._pulse_resolved:
            sample_degradation_dock = Dock(
                "Sample Degradation", size=(self._SAMPLE_W, self._SAMPLE_H))
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
