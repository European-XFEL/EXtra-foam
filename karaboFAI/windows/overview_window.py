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

from .base_window import DockerWindow, SingletonWindow
from ..logger import logger
from ..widgets import (
    BulletinWidget, ImageView, LaserOnOffAiWidget, LaserOnOffFomWidget,
    make_pen, MultiPulseAiWidget, RoiImageView, RoiValueMonitor,
    SampleDegradationWidget, SinglePulseAiWidget, SinglePulseImageView
)
from ..config import config


class OverviewWindow(DockerWindow):
    """OverviewWindow class.

    For pulse-resolved detectors.
    """
    title = "overview"

    _ASSEMBLED_IMG_W = 600
    _ASSEMBLED_IMG_H = 500
    _M_PULSE_AI_W = 900
    _M_PULSE_AI_H = _ASSEMBLED_IMG_H
    _S_PULSE_AI_W = _ASSEMBLED_IMG_W
    _S_PULSE_AI_H = 250
    _BULLETIN_W = _M_PULSE_AI_W
    _BULLETIN_H = 100
    _SAMPLE_W = _M_PULSE_AI_W
    _SAMPLE_H = 2 * _S_PULSE_AI_H - _BULLETIN_H
    _TOTAL_W = _ASSEMBLED_IMG_W + _M_PULSE_AI_W
    _TOTAL_H = _ASSEMBLED_IMG_H + 2 * _S_PULSE_AI_H

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._bulletin_widget = BulletinWidget(parent=self)
        self._assembled_image = ImageView(parent=self)
        self._multi_pulse_ai = MultiPulseAiWidget(parent=self)

        self._sample_degradation = SampleDegradationWidget(parent=self)

        self._vip_pulse1_ai_dock = None
        self._vip_pulse1_img_dock = None
        self._vip_pulse1_ai = SinglePulseAiWidget(parent=self)
        self._vip_pulse1_img = SinglePulseImageView(parent=self)

        self._vip_pulse2_ai_dock = None
        self._vip_pulse2_img_dock = None
        self._vip_pulse2_ai = SinglePulseAiWidget(parent=self)
        self._vip_pulse2_img = SinglePulseImageView(parent=self)

        self._laser_on_off_fom = LaserOnOffFomWidget(parent=self)
        self._laser_on_off_ai = LaserOnOffAiWidget(parent=self)

        self.initUI()

        parent = self.parent()

        parent.analysis_ctrl_widget.vip_pulse_id1_sgn.connect(
            self.onPulseID1Updated)
        parent.analysis_ctrl_widget.vip_pulse_id2_sgn.connect(
            self.onPulseID2Updated)

        self.resize(self._TOTAL_W, self._TOTAL_W)

        self.update()

        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        assembled_image_dock = Dock("Mean Assembled Image", size=(
            self._ASSEMBLED_IMG_W, self._ASSEMBLED_IMG_H))
        self._docker_area.addDock(assembled_image_dock, 'left')
        assembled_image_dock.addWidget(self._assembled_image)

        laser_on_off_ai_dock = Dock("Laser On-Off Azimuthal Integration",
                                    size=(self._M_PULSE_AI_W,
                                          self._M_PULSE_AI_H))
        self._docker_area.addDock(laser_on_off_ai_dock, 'right')
        laser_on_off_ai_dock.addWidget(self._laser_on_off_ai)

        multi_pulse_ai_dock = Dock("Multi-pulse Azimuthal Integration",
                                   size=(self._M_PULSE_AI_W,
                                         self._M_PULSE_AI_H))
        self._docker_area.addDock(
            multi_pulse_ai_dock, 'above', laser_on_off_ai_dock)
        multi_pulse_ai_dock.addWidget(self._multi_pulse_ai)

        bulletin_docker = Dock("Bulletin",
                               size=(self._BULLETIN_W, self._BULLETIN_H))
        self._docker_area.addDock(bulletin_docker, 'bottom',
                                  "Multi-pulse Azimuthal Integration")
        bulletin_docker.addWidget(self._bulletin_widget)
        bulletin_docker.hideTitleBar()

        laser_on_off_fom_dock = Dock("Laser On-Off Azimuthal Integration",
                                     size=(self._SAMPLE_W,
                                           self._SAMPLE_H))
        self._docker_area.addDock(laser_on_off_fom_dock, 'bottom', 'Bulletin')
        laser_on_off_fom_dock.addWidget(self._laser_on_off_fom)

        sample_degradation_dock = Dock("Sample Degradation",
                                       size=(self._SAMPLE_W, self._SAMPLE_H))
        self._docker_area.addDock(
            sample_degradation_dock, 'above', laser_on_off_fom_dock)
        sample_degradation_dock.addWidget(self._sample_degradation)

        self._vip_pulse2_ai_dock = Dock("VIP pulse 0000 - AI",
                                        size=(self._S_PULSE_AI_W,
                                              self._S_PULSE_AI_H))
        self._docker_area.addDock(self._vip_pulse2_ai_dock, 'bottom',
                                  "Mean Assembled Image")
        self._vip_pulse2_ai_dock.addWidget(self._vip_pulse2_ai)

        self._vip_pulse1_ai_dock = Dock("VIP pulse 0000 - AI",
                                        size=(self._S_PULSE_AI_W,
                                              self._S_PULSE_AI_H))
        self._docker_area.addDock(self._vip_pulse1_ai_dock, 'bottom',
                                  "Mean Assembled Image")
        self._vip_pulse1_ai_dock.addWidget(self._vip_pulse1_ai)

        self._vip_pulse2_img_dock = Dock("VIP pulse 0000",
                                         size=(self._S_PULSE_AI_W,
                                               self._S_PULSE_AI_H))
        self._docker_area.addDock(self._vip_pulse2_img_dock, 'above',
                                  self._vip_pulse2_ai_dock)
        self._vip_pulse2_img_dock.addWidget(self._vip_pulse2_img)

        self._vip_pulse1_img_dock = Dock("VIP pulse 0000",
                                         size=(self._S_PULSE_AI_W,
                                               self._S_PULSE_AI_H))
        self._docker_area.addDock(self._vip_pulse1_img_dock, 'above',
                                  self._vip_pulse1_ai_dock)
        self._vip_pulse1_img_dock.addWidget(self._vip_pulse1_img)

    @QtCore.pyqtSlot(int)
    def onPulseID1Updated(self, value):
        self._vip_pulse1_ai_dock.setTitle("VIP pulse {:04d} - AI".format(value))
        self._vip_pulse1_img_dock.setTitle("VIP pulse {:04d}".format(value))
        self._vip_pulse1_ai.pulse_id = value
        self._vip_pulse1_img.pulse_id = value

    @QtCore.pyqtSlot(int)
    def onPulseID2Updated(self, value):
        self._vip_pulse2_ai_dock.setTitle("VIP pulse {:04d} - AI".format(value))
        self._vip_pulse2_img_dock.setTitle("VIP pulse {:04d}".format(value))
        self._vip_pulse2_ai.pulse_id = value
        self._vip_pulse2_img.pulse_id = value


class OverviewWindowTrainResolved(DockerWindow):
    """OverviewWindow class.

    For train-resolved detectors.
    """

    title = "overview"

    _ASSEMBLED_IMG_W = 600
    _ASSEMBLED_IMG_H = 500
    _AI_W = 900
    _AI_H = _ASSEMBLED_IMG_H
    _BULLETIN_W = _AI_W
    _BULLETIN_H = 100
    _ROI_W = 600
    _ROI_H = 250
    _ROI_INTENSITY_W = _AI_W
    _ROI_INTENSITY_H = 2 * _ROI_H - _BULLETIN_H
    _TOTAL_W = _ASSEMBLED_IMG_W + _AI_W
    _TOTAL_H = _ASSEMBLED_IMG_H + 2 * _ROI_H

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._bulletin_widget = BulletinWidget(pulse_resolved=False,
                                               parent=self)
        self._assembled_image = ImageView(parent=self)
        self._ai = SinglePulseAiWidget(parent=self, plot_mean=False)

        self._roi1_image = RoiImageView(parent=self)
        self._roi1_image.setBorder(make_pen(config["ROI_COLORS"][0]))
        self._roi2_image = RoiImageView(roi1=False, parent=self)
        self._roi2_image.setBorder(make_pen(config["ROI_COLORS"][1]))

        self._roi_intensity = RoiValueMonitor(parent=self)
        self._mediator.roi_displayed_range_sgn.connect(
            self._roi_intensity.onDisplayRangeChange)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.update()

        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        assembled_image_dock = Dock("Assembled Image", size=(
            self._ASSEMBLED_IMG_W, self._ASSEMBLED_IMG_H))
        self._docker_area.addDock(assembled_image_dock, 'left')
        assembled_image_dock.addWidget(self._assembled_image)

        ai_dock = Dock("Azimuthal Integration", size=(self._AI_W, self._AI_H))
        self._docker_area.addDock(ai_dock, 'right')
        ai_dock.addWidget(self._ai)

        bulletin_docker = Dock("Bulletin",
                               size=(self._BULLETIN_W, self._BULLETIN_H))
        self._docker_area.addDock(bulletin_docker, 'bottom',
                                  "Azimuthal Integration")
        bulletin_docker.addWidget(self._bulletin_widget)
        bulletin_docker.hideTitleBar()

        roi1_image_dock = Dock("ROI1", size=(self._ROI_W, self._ROI_H))
        self._docker_area.addDock(roi1_image_dock, 'bottom',
                                  assembled_image_dock)
        roi1_image_dock.addWidget(self._roi1_image)

        roi2_image_dock = Dock("ROI2", size=(self._ROI_W, self._ROI_H))
        self._docker_area.addDock(roi2_image_dock, 'bottom',
                                  roi1_image_dock)
        roi2_image_dock.addWidget(self._roi2_image)

        roi_intensity_dock = Dock("ROI intensity", size=(
            self._ROI_INTENSITY_W, self._ROI_INTENSITY_H))
        self._docker_area.addDock(roi_intensity_dock, 'bottom',
                                  bulletin_docker)
        roi_intensity_dock.addWidget(self._roi_intensity)
