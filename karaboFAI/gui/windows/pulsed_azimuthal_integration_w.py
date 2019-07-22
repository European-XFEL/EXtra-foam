"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PulsedAzimuthalIntegrationWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..plot_widgets import TrainAiWidget, SinglePulseImageView
from ...config import config


class PulsedAzimuthalIntegrationWindow(DockerWindow):
    """PulsedAzimuthalIntegrationWindow class."""
    title = "pulsed-azimuthal-integration"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _LW = 0.5 * _TOTAL_W
    _LH = 0.5 * _TOTAL_H
    _RW = 0.5 * _TOTAL_W
    _RH1 = 0.3 * _TOTAL_H
    _RH2 = 0.4 * _TOTAL_H

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        if self._pulse_resolved:
            self._ai = TrainAiWidget(parent=self)

            self._vip1_img_dock = None
            self._vip1_img = SinglePulseImageView(parent=self)

            self._vip2_img_dock = None
            self._vip2_img = SinglePulseImageView(parent=self)

            self.initUI()

            self.resize(self._TOTAL_W, self._TOTAL_H)
        else:
            self._ai = TrainAiWidget(parent=self)

            self.initUI()
            self.resize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.initConnections()
        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""

        if self._pulse_resolved:

            # left column

            self._vip1_img_dock = Dock("VIP pulse 0000",
                                       size=(self._LW, self._LH))
            self._docker_area.addDock(self._vip1_img_dock)
            self._vip1_img_dock.addWidget(self._vip1_img)

            self._vip2_img_dock = Dock("VIP pulse 0000",
                                       size=(self._LW, self._LH))
            self._docker_area.addDock(
                self._vip2_img_dock, 'bottom', self._vip1_img_dock)
            self._vip2_img_dock.addWidget(self._vip2_img)

            # right column

            ai_dock = Dock("Normalized azimuthal Integration",
                           size=(self._RW, self._RH2))
            self._docker_area.addDock(ai_dock, 'right')
            ai_dock.addWidget(self._ai)

        else:
            ai_dock = Dock("Normalized azimuthal Integration",
                           size=(self._TOTAL_W, self._TOTAL_H))
            self._docker_area.addDock(ai_dock)
            ai_dock.addWidget(self._ai)

    def initConnections(self):
        """Override."""
        if self._pulse_resolved:
            mediator = self._mediator
            mediator.poi_index1_sgn.connect(self.onPulseID1Updated)
            mediator.poi_index2_sgn.connect(self.onPulseID2Updated)
            mediator.poi_indices_connected_sgn.emit()

    @QtCore.pyqtSlot(int)
    def onPulseID1Updated(self, value):
        self._vip1_img_dock.setTitle("VIP pulse {:04d}".format(value))
        self._vip1_img.pulse_index = value

    @QtCore.pyqtSlot(int)
    def onPulseID2Updated(self, value):
        self._vip2_img_dock.setTitle("VIP pulse {:04d}".format(value))
        self._vip2_img.pulse_index = value
