"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

SinglePulseAiWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import PlotWidget

from ..logger import logger
from ..widgets.misc_widgets import PenFactory


class SinglePulseAiWidget(PlotWidget):
    """SinglePulseAiWidget class.

    A widget which allows user to visualize the the azimuthal integration
    result of individual pulses. The azimuthal integration result is also
    compared with the average azimuthal integration of all the pulses.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.addLegend(offset=(-40, 20))

    def update(self, data, pulse_id):
        if pulse_id >= data.intensity.shape[0]:
            logger.error("Pulse ID {} out of range (0 - {})!".
                         format(pulse_id, data.intensity.shape[0] - 1))
            return

        self.plot(data.momentum, data.intensity[pulse_id],
                  name="Pulse {}".format(pulse_id),
                  pen=PenFactory.yellow)
        self.plot(data.momentum, data.intensity_mean,
                  name="mean",
                  pen=PenFactory.cyan)
