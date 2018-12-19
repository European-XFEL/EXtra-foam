"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

OverviewWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph.dockarea import Dock, DockArea
from ..widgets.pyqtgraph import LayoutWidget, QtCore, QtGui

from .base_window import AbstractWindow, SingletonWindow
from ..logger import logger
from ..widgets import (
    ImageAnalysisWidget, SinglePulseAiWidget, MultiPulseAiWidget,
    SampleDegradationWidget, SinglePulseImageWidget
)


@SingletonWindow
class OverviewWindow(AbstractWindow):
    """OverviewWindow class."""

    title = "overview"

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)
        parent.registerPlotWidget(self)

        self.mask_range_sp = None
        parent.mask_range_sgn.connect(self.onMaskRangeChanged)

        self.normalization_range_sp = None
        parent.normalization_range_sgn.connect(
            self.onNormalizationRangeChanged)

        self.diff_integration_range_sp = None
        parent.diff_integration_range_sgn.connect(
            self.onDiffIntegrationRangeChanged)

        # tell MainGUI to emit signals in order to update shared parameters
        parent.updateSharedParameters()

        self._title_lb = QtGui.QLabel("")

        self._docker_area = DockArea()

        self._title_widget = LayoutWidget(parent=self)
        self._assembled_image = ImageAnalysisWidget(parent=self)
        self._multi_pulse_ai = MultiPulseAiWidget(parent=self)
        self._sample_degradation = SampleDegradationWidget(parent=self)
        self._single_pulse_ai1 = SinglePulseAiWidget(parent=self)
        self._single_pulse_ai2 = SinglePulseAiWidget(parent=self)
        self._single_pulse_img1 = SinglePulseImageWidget(parent=self)
        self._single_pulse_img2 = SinglePulseImageWidget(parent=self)

        self.initUI()
        self.updatePlots()

        self.resize(1500, 1000)
        logger.info("Open {}".format(self.__class__.__name__))

    def initUI(self):
        """Override."""
        self._title_lb.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
        self._title_widget.addWidget(self._title_lb)

        self.initPlotUI()
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._docker_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cw.setLayout(layout)

    def initPlotUI(self):
        """Override."""
        assembled_image_dock = Dock("Mean Assembled Image", size=(600, 500))
        self._docker_area.addDock(assembled_image_dock, 'left')
        assembled_image_dock.addWidget(self._assembled_image)

        multi_pulse_ai_dock = Dock("Multi-pulse Azimuthal Integration",
                                   size=(900, 580))
        self._docker_area.addDock(multi_pulse_ai_dock, 'right')
        multi_pulse_ai_dock.addWidget(self._multi_pulse_ai)

        title_docker = Dock("Title", size=(900, 20))
        self._docker_area.addDock(title_docker, 'top',
                                  "Multi-pulse Azimuthal Integration")
        title_docker.addWidget(self._title_widget)
        title_docker.hideTitleBar()

        sample_degradation_dock = Dock("Sample Degradation", size=(900, 500))
        self._docker_area.addDock(sample_degradation_dock, 'bottom',
                                  "Multi-pulse Azimuthal Integration")
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

    def clearPlots(self):
        """Clear plots.

        This method is called by the main GUI.
        """
        self._assembled_image.clear()
        self._multi_pulse_ai.clear()
        self._sample_degradation.clear()
        self._single_pulse_ai1.clear()
        self._single_pulse_ai2.clear()
        self._single_pulse_img1.clear()
        self._single_pulse_img2.clear()

    def updatePlots(self):
        """Update plots.

        This method is called by the main GUI.
        """
        data = self._data.get()
        if data.empty():
            return

        self._title_lb.setText("Train ID: {}{}Number of pulses per train: {}".
                               format(data.tid, ' '*8, len(data.intensity)))

        self._assembled_image.update(data)

        self._multi_pulse_ai.update(data)

        self._sample_degradation.update(
            data, self.normalization_range_sp, self.diff_integration_range_sp)

        self._single_pulse_ai1.update(data, 0)
        self._single_pulse_ai2.update(data, 1)

        self._single_pulse_img1.update(data, 0, self.mask_range_sp)
        self._single_pulse_img2.update(data, 1, self.mask_range_sp)

    @QtCore.pyqtSlot(float, float)
    def onNormalizationRangeChanged(self, lb, ub):
        self.normalization_range_sp = (lb, ub)

    @QtCore.pyqtSlot(float, float)
    def onDiffIntegrationRangeChanged(self, lb, ub):
        self.diff_integration_range_sp = (lb, ub)

    @QtCore.pyqtSlot(float, float)
    def onMaskRangeChanged(self, lb, ub):
        self.mask_range_sp = (lb, ub)
