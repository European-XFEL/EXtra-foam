"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

LaserOnOffWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from ..widgets.pyqtgraph import BarGraphItem, QtGui
from ..widgets.pyqtgraph import parametertree as ptree

from .base_window import PlotWindow, SingletonWindow
from ..logger import logger
from ..data_processing.proc_utils import normalize_curve, slice_curve


@SingletonWindow
class SampleDegradationWindow(PlotWindow):
    """SampleDegradationWindow class.

    A window which allows users to monitor the degradation of the sample
    within a train.
    """
    plot_w = 800
    plot_h = 450

    title = "sample degradation monitor"

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)

        self.initUI()
        self.updatePlots()

        logger.info("Open SampleDegradationMonitor")

    def initPlotUI(self):
        """Override."""
        self._gl_widget.setFixedSize(self.plot_w, self.plot_h)

        self._gl_widget.nextRow()

        p = self._gl_widget.addPlot()
        self._plot_items.append(p)
        p.setLabel('left', "Integrated difference (arb.)")
        p.setLabel('bottom', "Pulse ID")
        p.setTitle('Integrated absolute difference with respect to '
                   'the first pulse')

    def initCtrlUI(self):
        """Override."""
        self._ctrl_widget = QtGui.QWidget()
        self._ctrl_widget.setMinimumWidth(300)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._ptree)
        self._ctrl_widget.setLayout(layout)

    def updatePlots(self):
        """Override."""
        data = self._data.get()
        if data.empty():
            return

        momentum = data.momentum

        # normalize azimuthal integration curves for each pulse
        normalized_pulse_intensities = []
        for pulse_intensity in data.intensity:
            normalized = normalize_curve(
                pulse_intensity, momentum, *self.normalization_range_sp)
            normalized_pulse_intensities.append(normalized)

        # calculate the different between each pulse and the first one
        diffs = [p - normalized_pulse_intensities[0]
                 for p in normalized_pulse_intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self.diff_integration_range_sp)[0]
            foms.append(np.sum(np.abs(fom)))

        bar = BarGraphItem(
            x=range(len(foms)), height=foms, width=0.6, brush='b')

        p = self._plot_items[0]
        p.addItem(bar)
        p.plot()

    def updateParameterTree(self):
        """Override."""
        self._pro_params.addChildren([
            self.normalization_range_param,
            self.diff_integration_range_param,
        ])

        params = ptree.Parameter.create(name='params', type='group',
                                        children=[self._pro_params])

        self._ptree.setParameters(params, showTop=False)
