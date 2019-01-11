"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

SampleDegradationWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .pyqtgraph import BarGraphItem, QtCore

from ..data_processing.proc_utils import normalize_curve, slice_curve
from .plot_widget import PlotWidget


class SampleDegradationWidget(PlotWidget):
    """SampleDegradationWindow class.

    A widget which allows users to monitor the degradation of the sample
    within a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._normalization_range_sp = None
        parent.parent().ana_setup_widget.normalization_range_sgn.connect(
            self.onNormalizationRangeChanged)

        self._diff_integration_range_sp = None
        parent.parent().ana_setup_widget.diff_integration_range_sgn.connect(
            self.onDiffIntegrationRangeChanged)

        self.setLabel('left', "Integrated difference (arb.)")
        self.setLabel('bottom', "Pulse ID")
        self.setTitle('FOM with respect to the first pulse')

    def update(self, data):
        """Override."""
        # TODO: since this becomes a mandatory widget, we can consider to
        # TODO: move the calculation outside of the update method.
        momentum = data.momentum

        # normalize azimuthal integration curves for each pulse
        normalized_pulse_intensities = []
        for pulse_intensity in data.intensity:
            normalized = normalize_curve(
                pulse_intensity, momentum, *self._normalization_range_sp)
            normalized_pulse_intensities.append(normalized)

        # calculate the different between each pulse and the first one
        diffs = [p - normalized_pulse_intensities[0]
                 for p in normalized_pulse_intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self._diff_integration_range_sp)[0]
            foms.append(np.sum(np.abs(fom)))

        self.addItem(BarGraphItem(
            x=range(len(foms)), height=foms, width=0.6, brush='b'))

    @QtCore.pyqtSlot(float, float)
    def onNormalizationRangeChanged(self, lb, ub):
        self._normalization_range_sp = (lb, ub)

    @QtCore.pyqtSlot(float, float)
    def onDiffIntegrationRangeChanged(self, lb, ub):
        self._diff_integration_range_sp = (lb, ub)
