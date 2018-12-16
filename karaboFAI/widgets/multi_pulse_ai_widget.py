"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

MultiPulseAiWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import intColor, mkPen, PlotWidget


class MultiPulseAiWidget(PlotWidget):
    """MultiPulseAiWidget class.

    Widget used for displaying azimuthal integration result for all
    the pulses in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("")
        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setLabel('left', "Scattering signal (arb. u.)")

    def update(self, data):
        momentum = data.momentum
        for i, intensity in enumerate(data.intensity):
            # TODO: use setData, but take of pulse number changes
            self.plot(momentum, intensity,
                      pen=mkPen(intColor(i, hues=9, values=5), width=2))

        self.setTitle("Train ID: {}, number of pulses: {}".
                      format(data.tid, len(data.intensity)))
