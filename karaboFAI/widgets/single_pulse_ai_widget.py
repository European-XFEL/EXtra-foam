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
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.pulse_id = 0

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
        try:
            self._pulse_plot.setData(data.momentum, data.intensity[self.pulse_id])
        except IndexError as e:
            logger.error("<VIP pulse ID>: " + str(e))
            return

        self._mean_plot.setData(data.momentum, data.intensity_mean)
