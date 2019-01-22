"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

SinglePulseAiWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import QtCore

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

        self._pulse_id = 0

        self.setTitle('')
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
        if self._pulse_id >= data.intensity.shape[0]:
            logger.error("Out of range: valid range of VIP pulse ID is 0 - {}!".
                         format(data.intensity.shape[0] - 1))
            self.setTitle('')
            return

        self.setTitle('Pulse ID: {:04d}'.format(self._pulse_id))
        self._pulse_plot.setData(data.momentum, data.intensity[self._pulse_id])
        self._mean_plot.setData(data.momentum, data.intensity_mean)

    @QtCore.pyqtSlot(int)
    def onPulseIDUpdated(self, value):
        self._pulse_id = value
