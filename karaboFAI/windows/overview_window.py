"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

OverviewWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph.dockarea import Dock

from .base_window import DockerWindow, SingletonWindow
from ..logger import logger
from ..widgets import (
    BulletinWidget, ImageAnalysisWidget, MultiPulseAiWidget,
    SampleDegradationWidget, SinglePulseAiWidget, SinglePulseImageWidget
)


@SingletonWindow
class OverviewWindow(DockerWindow):
    """OverviewWindow class."""

    title = "overview"

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)
        parent.registerPlotWindow(self)

        self._bulletin_widget = BulletinWidget(parent=self)
        self._assembled_image = ImageAnalysisWidget(parent=self)
        self._multi_pulse_ai = MultiPulseAiWidget(parent=self)
        self._sample_degradation = SampleDegradationWidget(parent=self)

        self._single_pulse_ai1 = SinglePulseAiWidget(parent=self)
        parent.data_ctrl_widget.vip_pulse_id1_sgn.connect(
            self._single_pulse_ai1.onPulseIDUpdated)
        self._single_pulse_img1 = SinglePulseImageWidget(
            parent=self, pulse_id=0)

        self._single_pulse_ai2 = SinglePulseAiWidget(parent=self)
        parent.data_ctrl_widget.vip_pulse_id2_sgn.connect(
            self._single_pulse_ai2.onPulseIDUpdated)
        self._single_pulse_img2 = SinglePulseImageWidget(
            parent=self, pulse_id=1)

        # tell MainGUI to emit signals in order to update shared parameters
        # Note: must be called after all the Widgets which own shared
        # parameters have been initialized
        parent.updateSharedParameters()

        self.initUI()
        self.update()

        self.resize(1500, 1000)
        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        assembled_image_dock = Dock("Mean Assembled Image", size=(600, 500))
        self._docker_area.addDock(assembled_image_dock, 'left')
        assembled_image_dock.addWidget(self._assembled_image)

        multi_pulse_ai_dock = Dock("Multi-pulse Azimuthal Integration",
                                   size=(900, 500))
        self._docker_area.addDock(multi_pulse_ai_dock, 'right')
        multi_pulse_ai_dock.addWidget(self._multi_pulse_ai)

        bulletin_docker = Dock("Bulletin", size=(900, 100))
        self._docker_area.addDock(bulletin_docker, 'bottom',
                                  "Multi-pulse Azimuthal Integration")
        bulletin_docker.addWidget(self._bulletin_widget)
        bulletin_docker.hideTitleBar()

        sample_degradation_dock = Dock("Sample Degradation", size=(900, 400))
        self._docker_area.addDock(sample_degradation_dock, 'bottom', "Bulletin")
        sample_degradation_dock.addWidget(self._sample_degradation)

        single_pulse_ai2_dock = Dock("Single pulse AI 2", size=(600, 250))
        self._docker_area.addDock(single_pulse_ai2_dock, 'bottom',
                                  "Mean Assembled Image")
        single_pulse_ai2_dock.addWidget(self._single_pulse_ai2)

        single_pulse_ai1_dock = Dock("Single pulse AI 1", size=(600, 250))
        self._docker_area.addDock(single_pulse_ai1_dock, 'bottom',
                                  "Mean Assembled Image")
        single_pulse_ai1_dock.addWidget(self._single_pulse_ai1)

        single_pulse_img2_dock = Dock("Single pulse image 2", size=(600, 250))
        self._docker_area.addDock(single_pulse_img2_dock, 'above',
                                  "Single pulse AI 2")
        single_pulse_img2_dock.addWidget(self._single_pulse_img2)

        single_pulse_img1_dock = Dock("Single pulse image 1", size=(600, 250))
        self._docker_area.addDock(single_pulse_img1_dock, 'above',
                                  "Single pulse AI 1")
        single_pulse_img1_dock.addWidget(self._single_pulse_img1)
