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
    def __init__(self, *, plot_mean=True, parent=None):
        """Initialization.

        :param bool plot_mean: whether to plot the mean AI of all pulses
            if the data is pulse resolved.
        """
        super().__init__(parent=parent)

        self.pulse_id = 0

        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setLabel('bottom', "Momentum transfer (1/A)")

        self._pulse_plot = self.plot(name="pulse_plot", pen=PenFactory.yellow)

        if plot_mean:
            self._mean_plot = self.plot(name="mean", pen=PenFactory.cyan)
            self.addLegend(offset=(-40, 20))
        else:
            self._mean_plot = None

    def clear(self):
        """Override."""
        self.reset()

    def reset(self):
        """Override."""
        self._pulse_plot.setData([], [])
        if self._mean_plot is not None:
            self._mean_plot.setData([], [])

    def update(self, data):
        """Override."""
        # pulse resolved data
        if data.intensity.ndim == 2:
            max_id = len(data.intensity) - 1
            if self.pulse_id <= max_id:
                self._pulse_plot.setData(data.momentum,
                                         data.intensity[self.pulse_id])
            else:
                logger.error("<VIP pulse ID>: VIP pulse ID ({}) > Maximum "
                             "pulse ID ({})".format(self.pulse_id, max_id))
                return
        else:
            self._pulse_plot.setData(data.momentum, data.intensity)

        if self._mean_plot is not None:
            self._mean_plot.setData(data.momentum, data.intensity_mean)
