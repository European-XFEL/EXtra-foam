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
from ..widgets.pyqtgraph import (
    ImageView, intColor, mkPen, PlotWidget, QtCore, QtGui
)


from .base_window import AbstractWindow
from ..config import config
from ..logger import logger
from ..widgets import colorMapFactory, SampleDegradationWidget


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
        self._assembled = ImageView()
        self._multiline = PlotWidget()
        self._sample_degradation = SampleDegradationWidget()

        self.initUI()
        self._is_initialized = False
        self.updatePlots()

        self.resize(1500, 1000)
        logger.info("Open {}".format(self.__class__.__name__))

        # tell MainGUI to emit signals in order to update shared parameters
        self.parent().updateSharedParameters()

        print(self.normalization_range_sp)

    def initUI(self):
        """Override."""
        self.initPlotUI()
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._docker_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cw.setLayout(layout)

    def initPlotUI(self):
        """Override."""
        assembled_dock = Dock("Assembled Detector Image", size=(600, 600))
        self._docker_area.addDock(assembled_dock, 'left')
        assembled_dock.addWidget(self._assembled)

        imageview_dock = Dock("Azimuthal Integration", size=(900, 600))
        self._docker_area.addDock(imageview_dock, 'right')
        imageview_dock.addWidget(self._multiline)

        sample_degradation_dock = Dock("Sample Degradation", size=(600, 400))
        self._docker_area.addDock(
            sample_degradation_dock, 'bottom', "Assembled Detector Image")
        sample_degradation_dock.addWidget(self._sample_degradation)

        self._assembled.setColorMap(colorMapFactory[config["COLOR_MAP"]])

        self._multiline.setTitle("")
        self._multiline.setLabel('bottom', "Momentum transfer (1/A)")
        self._multiline.setLabel('left', "Scattering signal (arb. u.)")

    def clearPlots(self):
        """Clear plots.

        This method is called by the main GUI.
        """
        self._multiline.clear()
        self._assembled.clear()
        self._sample_degradation.clear()

    def updatePlots(self):
        """Update plots.

        This method is called by the main GUI.
        """
        data = self._data.get()
        if data.empty():
            return

        self._assembled.setImage(data.image_mean, autoRange=False,
                                 autoLevels=(not self._is_initialized))
        if not self._is_initialized:
            self._is_initialized = True

        momentum = data.momentum
        line = self._multiline
        for i, intensity in enumerate(data.intensity):
            line.plot(momentum, intensity,
                      pen=mkPen(intColor(i, hues=9, values=5), width=2))
            line.setTitle("Train ID: {}, number of pulses: {}".
                          format(data.tid, len(data.intensity)))

        self._sample_degradation.updatePlots(
            data, self.normalization_range_sp, self.diff_integration_range_sp)

    @QtCore.pyqtSlot(float, float)
    def onNormalizationRangeChanged(self, lb, ub):
        self.normalization_range_sp = (lb, ub)
        # then update the parameter tree
        try:
            self._pro_params.child('Normalization range').setValue(
                '{}, {}'.format(lb, ub))
        except KeyError:
            pass

    @QtCore.pyqtSlot(float, float)
    def onDiffIntegrationRangeChanged(self, lb, ub):
        self.diff_integration_range_sp = (lb, ub)
        # then update the parameter tree
        try:
            self._pro_params.child("Diff integration range").setValue(
                '{}, {}'.format(lb, ub))
        except KeyError:
            pass
