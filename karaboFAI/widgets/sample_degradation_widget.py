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

from .pyqtgraph import BarGraphItem, PlotWidget

from ..data_processing.proc_utils import normalize_curve, slice_curve


class SampleDegradationWidget(PlotWidget):
    """SampleDegradationWindow class.

    A widget which allows users to monitor the degradation of the sample
    within a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._p = self.plot()
        self.setLabel('left', "Integrated difference (arb.)")
        self.setLabel('bottom', "Pulse ID")
        self.setTitle('FOM with respect to the first pulse')

    def updatePlots(self, data, normalization_range, diff_integration_range):
        """Override."""
        momentum = data.momentum

        # normalize azimuthal integration curves for each pulse
        normalized_pulse_intensities = []
        for pulse_intensity in data.intensity:
            normalized = normalize_curve(
                pulse_intensity, momentum, *normalization_range)
            normalized_pulse_intensities.append(normalized)

        # calculate the different between each pulse and the first one
        diffs = [p - normalized_pulse_intensities[0]
                 for p in normalized_pulse_intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *diff_integration_range)[0]
            foms.append(np.sum(np.abs(fom)))

        bar = BarGraphItem(
            x=range(len(foms)), height=foms, width=0.6, brush='b')

        self.addItem(bar)
