"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

RoiWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph.dockarea import Dock

from .base_window import DockerWindow
from ..mediator import Mediator
from ..misc_widgets import make_pen
from ..plot_widgets import RoiImageView, RoiValueMonitor
from ...config import config


class RoiWindow(DockerWindow):
    """RoiWindow class."""
    title = "ROI"

    _TOTAL_W = 800
    _TOTAL_H = 1000

    _LW1 = 0.5 * _TOTAL_W
    _LW2 = _TOTAL_W
    _LH1 = 0.3 * _TOTAL_H
    _LH2 = 0.4 * _TOTAL_H

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        mediator = Mediator()

        self._roi1_image = RoiImageView(1, parent=self)
        self._roi1_image.setBorder(make_pen(config["ROI_COLORS"][0]))
        self._roi2_image = RoiImageView(2, parent=self)
        self._roi2_image.setBorder(make_pen(config["ROI_COLORS"][1]))
        self._roi3_image = RoiImageView(3, parent=self)
        self._roi3_image.setBorder(make_pen(config["ROI_COLORS"][2]))
        self._roi4_image = RoiImageView(4, parent=self)
        self._roi4_image.setBorder(make_pen(config["ROI_COLORS"][3]))

        self._roi_intensity = RoiValueMonitor(parent=self)
        mediator.roi_displayed_range_sgn.connect(
            self._roi_intensity.onDisplayRangeChange)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        roi1_image_dock = Dock("ROI1", size=(self._LW1, self._LH1))
        self._docker_area.addDock(roi1_image_dock)
        roi1_image_dock.addWidget(self._roi1_image)

        roi2_image_dock = Dock("ROI2", size=(self._LW1, self._LH1))
        self._docker_area.addDock(roi2_image_dock, 'bottom', roi1_image_dock)
        roi2_image_dock.addWidget(self._roi2_image)

        roi4_image_dock = Dock("ROI4", size=(self._LW1, self._LH1))
        self._docker_area.addDock(roi4_image_dock, 'right')
        roi4_image_dock.addWidget(self._roi4_image)

        roi3_image_dock = Dock("ROI3", size=(self._LW1, self._LH1))
        self._docker_area.addDock(roi3_image_dock, 'top', roi4_image_dock)
        roi3_image_dock.addWidget(self._roi3_image)

        roi_intensity_dock = Dock("ROI intensity", size=(self._LW2, self._LH2))
        self._docker_area.addDock(roi_intensity_dock, 'bottom')
        roi_intensity_dock.addWidget(self._roi_intensity)
