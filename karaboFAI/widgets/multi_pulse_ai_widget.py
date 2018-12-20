"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

MultiPulseAiWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import intColor, mkPen
from .plot_widget import PlotWidget


class MultiPulseAiWidget(PlotWidget):
    """MultiPulseAiWidget class.

    Widget used for displaying azimuthal integration result for all
    the pulses in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._n_pulses = 0

        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setLabel('left', "Scattering signal (arb. u.)")

    def clear(self):
        """Override."""
        self.reset()

    def reset(self):
        """Override."""
        for item in self.plotItem.items:
            item.setData([], [])
        self._n_pulses = 0

    def update(self, data):
        """Override."""
        momentum = data.momentum
        intensities = data.intensity

        n_pulses = len(intensities)
        if n_pulses != self._n_pulses:
            self._n_pulses = n_pulses
            # re-plot if number of pulses change
            self.clear()
            for i, intensity in enumerate(intensities):
                self.plot(momentum, intensity,
                          pen=mkPen(intColor(i, hues=9, values=5), width=2))
        else:
            for item, intensity in zip(self.plotItem.items, intensities):
                item.setData(momentum, intensity)
