"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PulseOfInterestWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..plot_widgets import SinglePulseImageView
from ...config import config


class PulseOfInterestWindow(DockerWindow):
    """PulseOfInterestWindow class."""
    title = "pulse-of-interest"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    _MW = _TOTAL_W
    _MH = 0.5 * _TOTAL_H

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._poi1_img_dock = None
        self._poi1_img = SinglePulseImageView(parent=self)

        self._poi2_img_dock = None
        self._poi2_img = SinglePulseImageView(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.initConnections()
        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        # upper row

        self._poi1_img_dock = Dock("POI pulse 0000", size=(self._MW, self._MH))
        self._docker_area.addDock(self._poi1_img_dock)
        self._poi1_img_dock.addWidget(self._poi1_img)

        # lower row

        self._poi2_img_dock = Dock("POI pulse 0000", size=(self._MW, self._MH))
        self._docker_area.addDock(
            self._poi2_img_dock, 'bottom', self._poi1_img_dock)
        self._poi2_img_dock.addWidget(self._poi2_img)

    def initConnections(self):
        """Override."""
        if self._pulse_resolved:
            mediator = self._mediator
            mediator.poi_index1_sgn.connect(self.onPulseID1Updated)
            mediator.poi_index2_sgn.connect(self.onPulseID2Updated)
            mediator.poi_indices_connected_sgn.emit()

    @QtCore.pyqtSlot(int)
    def onPulseID1Updated(self, value):
        self._poi1_img_dock.setTitle("POI pulse {:04d}".format(value))
        self._poi1_img.pulse_index = value

    @QtCore.pyqtSlot(int)
    def onPulseID2Updated(self, value):
        self._poi2_img_dock.setTitle("POI pulse {:04d}".format(value))
        self._poi2_img.pulse_index = value
