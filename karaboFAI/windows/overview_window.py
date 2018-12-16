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
from ..widgets.pyqtgraph import QtCore, QtGui

from .base_window import AbstractWindow
from ..logger import logger
from ..widgets import (
    ImageAnalysisWidget, SinglePulseAiWidget, MultiPulseAiWidget,
    SampleDegradationWidget
)


class OverviewWindow(AbstractWindow):
    """OverviewWindow class."""

    title = "overview"

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)
        self.parent().registerPlotWidget(self)

        self.normalization_range_sp = None
        self.diff_integration_range_sp = None
        self.parent().normalization_range_sgn.connect(
            self.onNormalizationRangeChanged)
        self.parent().diff_integration_range_sgn.connect(
            self.onDiffIntegrationRangeChanged)

        self._docker_area = DockArea()

        self._assembled_image = ImageAnalysisWidget(parent=self)
        self._multi_pulse_ai = MultiPulseAiWidget(parent=self)
        self._sample_degradation = SampleDegradationWidget(parent=self)
        self._single_pulse_ai1 = SinglePulseAiWidget(parent=self)
        self._single_pulse_ai2 = SinglePulseAiWidget(parent=self)

        self.initUI()
        self.updatePlots()

        self.resize(1500, 1000)
        logger.info("Open {}".format(self.__class__.__name__))

        # tell MainGUI to emit signals in order to update shared parameters
        self.parent().updateSharedParameters()

    def initUI(self):
        """Override."""
        self.initPlotUI()
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._docker_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cw.setLayout(layout)

    def initPlotUI(self):
        """Override."""
        assembled_image_dock = Dock("Assembled Image", size=(600, 600))
        self._docker_area.addDock(assembled_image_dock, 'left')
        assembled_image_dock.addWidget(self._assembled_image)

        multi_pulse_ai_dock = Dock("Multi-pulse Azimuthal Integration",
                                   size=(900, 600))
        self._docker_area.addDock(multi_pulse_ai_dock, 'right')
        multi_pulse_ai_dock.addWidget(self._multi_pulse_ai)

        sample_degradation_dock = Dock("Sample Degradation", size=(900, 400))
        self._docker_area.addDock(sample_degradation_dock, 'bottom',
                                  "Multi-pulse Azimuthal Integration")
        sample_degradation_dock.addWidget(self._sample_degradation)

        individual_pulse2_dock = Dock("Individual pulse", size=(600, 200))
        self._docker_area.addDock(individual_pulse2_dock, 'bottom',
                                  "Assembled Image")
        individual_pulse2_dock.addWidget(self._single_pulse_ai2)

        individual_pulse1_dock = Dock("Individual pulse", size=(600, 200))
        self._docker_area.addDock(individual_pulse1_dock, 'bottom',
                                  "Assembled Image")
        individual_pulse1_dock.addWidget(self._single_pulse_ai1)

    def clearPlots(self):
        """Clear plots.

        This method is called by the main GUI.
        """
        self._assembled_image.clear()
        self._multi_pulse_ai.clear()
        self._sample_degradation.clear()
        self._single_pulse_ai1.clear()
        self._single_pulse_ai2.clear()

    def updatePlots(self):
        """Update plots.

        This method is called by the main GUI.
        """
        data = self._data.get()
        if data.empty():
            return

        self._assembled_image.update(data)

        self._multi_pulse_ai.update(data)

        self._sample_degradation.update(
            data, self.normalization_range_sp, self.diff_integration_range_sp)

        self._single_pulse_ai1.update(data, 0)
        self._single_pulse_ai2.update(data, 1)

    @QtCore.pyqtSlot(float, float)
    def onNormalizationRangeChanged(self, lb, ub):
        self.normalization_range_sp = (lb, ub)

    @QtCore.pyqtSlot(float, float)
    def onDiffIntegrationRangeChanged(self, lb, ub):
        self.diff_integration_range_sp = (lb, ub)
