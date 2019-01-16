"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

SinglePulseAiWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .plot_widget import PlotWidget
from ..logger import logger
from ..widgets.misc_widgets import PenFactory


class SinglePulseAiWidget(PlotWidget):
    """SinglePulseAiWidget class.

    A widget which allows user to visualize the the azimuthal integration
    result of individual pulses. The azimuthal integration result is also
    compared with the average azimuthal integration of all the pulses.
    """
    def __init__(self, *, parent=None, pulse_id=0):
        """Initialization.

        :param int pulse_id: the ID of the pulse to be displayed.
        """
        super().__init__(parent=parent)

        self.pulse_id = pulse_id

        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.addLegend(offset=(-40, 20))

        self._pulse_plot = self.plot(name="pulse_plot", pen=PenFactory.yellow)
        self._mean_plot = self.plot(name="mean", pen=PenFactory.cyan)

    def clear(self):
        """Override."""
        self.reset()

    def reset(self):
        """Override."""
        self._pulse_plot.setData([], [])
        self._mean_plot.setData([], [])

    def update(self, data):
        """Override."""

        if self.pulse_id >= data.intensity.shape[0]:
            logger.error("Pulse ID {} out of range (0 - {})!".
                         format(self.pulse_id, data.intensity.shape[0] - 1))
            return

        self._pulse_plot.setData(data.momentum, data.intensity[self.pulse_id])
        self._mean_plot.setData(data.momentum, data.intensity_mean)
