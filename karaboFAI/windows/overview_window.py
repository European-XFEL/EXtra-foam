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
    BulletinWidget, ImageAnalysisWidget, MultiPulseAiWidget,
    SampleDegradationWidget, SinglePulseAiWidget, SinglePulseImageWidget
)


@SingletonWindow
class OverviewWindow(DockerWindow):
    """OverviewWindow class.

    For pulse-resolved detectors.
    """
    title = "overview"

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)
        parent.registerPlotWindow(self)

        self._bulletin_widget = BulletinWidget(parent=self)
        self._assembled_image = ImageAnalysisWidget(parent=self)
        self._multi_pulse_ai = MultiPulseAiWidget(parent=self)

        self._sample_degradation = SampleDegradationWidget(parent=self)

        self._vip_pulse1_ai_dock = None
        self._vip_pulse1_img_dock = None
        self._vip_pulse1_ai = SinglePulseAiWidget(parent=self)
        self._vip_pulse1_img = SinglePulseImageWidget(parent=self)

        self._vip_pulse2_ai_dock = None
        self._vip_pulse2_img_dock = None
        self._vip_pulse2_ai = SinglePulseAiWidget(parent=self)
        self._vip_pulse2_img = SinglePulseImageWidget(parent=self)

        self.initUI()

        parent.data_ctrl_widget.vip_pulse_id1_sgn.connect(
            self.onPulseID1Updated)
        parent.data_ctrl_widget.vip_pulse_id2_sgn.connect(
            self.onPulseID2Updated)

        self.resize(1500, 1000)

        # tell MainGUI to emit signals in order to update shared parameters
        # Note: must be called after all the Widgets which own shared
        # parameters have been initialized
        parent.updateSharedParameters()

        self.update()

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

        self._vip_pulse2_ai_dock = Dock("VIP pulse 0000 - AI", size=(600, 250))
        self._docker_area.addDock(self._vip_pulse2_ai_dock, 'bottom',
                                  "Mean Assembled Image")
        self._vip_pulse2_ai_dock.addWidget(self._vip_pulse2_ai)

        self._vip_pulse1_ai_dock = Dock("VIP pulse 0000 - AI", size=(600, 250))
        self._docker_area.addDock(self._vip_pulse1_ai_dock, 'bottom',
                                  "Mean Assembled Image")
        self._vip_pulse1_ai_dock.addWidget(self._vip_pulse1_ai)

        self._vip_pulse2_img_dock = Dock("VIP pulse 0000", size=(600, 250))
        self._docker_area.addDock(self._vip_pulse2_img_dock, 'above',
                                  self._vip_pulse2_ai_dock)
        self._vip_pulse2_img_dock.addWidget(self._vip_pulse2_img)

        self._vip_pulse1_img_dock = Dock("VIP pulse 0000", size=(600, 250))
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


@SingletonWindow
class OverviewWindowTrainResolved(DockerWindow):
    """OverviewWindow class.

    For train-resolved detectors.
    """

    title = "overview"

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)
        parent.registerPlotWindow(self)

        self._bulletin_widget = BulletinWidget(parent=self,
                                               pulse_resolved=False)
        self._assembled_image = ImageAnalysisWidget(parent=self)
        self._ai = MultiPulseAiWidget(parent=self)

        self.initUI()

        self.resize(1200, 500)

        # tell MainGUI to emit signals in order to update shared parameters
        # Note: must be called after all the Widgets which own shared
        # parameters have been initialized
        parent.updateSharedParameters()

        self.update()

        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        super().initUI()

    def initPlotUI(self):
        """Override."""
        assembled_image_dock = Dock("Assembled Image", size=(600, 400))
        self._docker_area.addDock(assembled_image_dock, 'left')
        assembled_image_dock.addWidget(self._assembled_image)

        ai_dock = Dock("Azimuthal Integration", size=(600, 400))
        self._docker_area.addDock(ai_dock, 'right')
        ai_dock.addWidget(self._ai)

        bulletin_docker = Dock("Bulletin", size=(600, 100))
        self._docker_area.addDock(bulletin_docker, 'top',
                                  "Azimuthal Integration")
        bulletin_docker.addWidget(self._bulletin_widget)
        bulletin_docker.hideTitleBar()
