"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Concrete PlotWidgets.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_plot_widget import PlotWidget

from ..pyqtgraph import QtCore

from ..misc_widgets import make_brush, make_pen
from ...logger import logger
from ...config import config


class SinglePulseAiWidget(PlotWidget):
    """SinglePulseAiWidget class.

    A widget which allows user to visualize the the azimuthal integration
    result of a single pulse. The azimuthal integration result is also
    compared with the average azimuthal integration of all the pulses.
    """
    def __init__(self, *, pulse_id=0, plot_mean=True, parent=None):
        """Initialization.

        :param int pulse_id: ID of the pulse to be plotted.
        :param bool plot_mean: whether to plot the mean AI of all pulses
            if the data is pulse resolved.
        """
        super().__init__(parent=parent)

        self.pulse_id = pulse_id

        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setLabel('bottom', "Momentum transfer (1/A)")

        if plot_mean:
            self.addLegend(offset=(-40, 20))

        self._pulse_plot = self.plotCurve(name="pulse_plot", pen=make_pen("y"))

        if plot_mean:
            self._mean_plot = self.plotCurve(name="mean", pen=make_pen("b"))
        else:
            self._mean_plot = None

    def update(self, data):
        """Override."""
        momentum = data.ai.momentum
        intensities = data.ai.intensities

        if intensities is None:
            return

        if intensities.ndim == 2:
            # pulse resolved data
            max_id = data.n_pulses - 1
            if self.pulse_id <= max_id:
                self._pulse_plot.setData(momentum,
                                         intensities[self.pulse_id])
            else:
                logger.error("<VIP pulse ID>: VIP pulse ID ({}) > Maximum "
                             "pulse ID ({})".format(self.pulse_id, max_id))
                return
        else:
            self._pulse_plot.setData(momentum, intensities)

        if self._mean_plot is not None:
            self._mean_plot.setData(momentum, data.ai.intensity_mean)


class MultiPulseAiWidget(PlotWidget):
    """MultiPulseAiWidget class.

    Widget for displaying azimuthal integration result for all
    the pulses in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._n_pulses = 0

        self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setLabel('left', "Scattering signal (arb. u.)")

    def update(self, data):
        """Override."""
        momentum = data.ai.momentum
        intensities = data.ai.intensities

        if intensities is None:
            return

        n_pulses = len(intensities)
        if n_pulses != self._n_pulses:
            self._n_pulses = n_pulses
            # re-plot if number of pulses change
            self.clear()
            for i, intensity in enumerate(intensities):
                self.plotCurve(momentum, intensity,
                               pen=make_pen(i, hues=9, values=5))
        else:
            for item, intensity in zip(self.plotItem.items, intensities):
                item.setData(momentum, intensity)


class PulsedFOMWidget(PlotWidget):
    """PulsedFOMWidget class.

    A widget which allows users to monitor the azimuthal integration FOM
    of each pulse with respect to the first pulse in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._plot = self.plotBar(width=0.6, brush=make_brush('b'))

        self.setLabel('left', "Integrated difference (arb.)")
        self.setLabel('bottom', "Pulse ID")
        self.setTitle('FOM with respect to the first pulse')

    def update(self, data):
        """Override."""
        foms = data.ai.pulse_fom
        if foms is None:
            return

        self._plot.setData(range(len(foms)), foms)


class RoiValueMonitor(PlotWidget):
    """RoiValueMonitor class.

    Widget for displaying the evolution of the value (integration, median,
    mean) of ROIs.
    """
    def __init__(self, *, window=600, parent=None):
        """Initialization.

        :param int window: window size, i.e. maximum number of trains to
            display. Default = 600.
        """
        super().__init__(parent=parent)

        self._window = window

        self.setLabel('bottom', "Train ID")
        self.setLabel('left', "Intensity (arb. u.)")
        self.addLegend(offset=(-40, 20))

        self._plots = []
        for i, c in enumerate(config["ROI_COLORS"], 1):
            self._plots.append(self.plotCurve(name=f"ROI {i}", pen=make_pen(c)))

    def update(self, data):
        """Override."""
        for i, plot in enumerate(self._plots, 1):
            tids, roi_hist, _ = getattr(data.roi, f"roi{i}_hist")
            plot.setData(tids[-self._window:], roi_hist[-self._window:])

    @QtCore.pyqtSlot(int)
    def onDisplayRangeChange(self, v):
        self._window = v


class CorrelationWidget(PlotWidget):
    """CorrelationWidget class.

    Widget for displaying correlations between FOM and different parameters.
    """
    _colors = ['c', 'b', 'o', 'y']
    _brushes = {
        0: make_brush(_colors[0], 120),
        1: make_brush(_colors[1], 120),
        2: make_brush(_colors[2], 120),
        3: make_brush(_colors[3], 120)
    }
    _opaque_brushes = {
        0: make_brush(_colors[0]),
        1: make_brush(_colors[1]),
        2: make_brush(_colors[2]),
        3: make_brush(_colors[3])
    }

    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._idx = idx

        self.setLabel('left', "FOM (arb. u.)")
        self.setLabel('bottom', "Correlator (arb. u.)")

        self._bar = self.plotErrorBar()
        self._plot = self.plotScatter(brush=self._brushes[self._idx])

        self._device_id = None
        self._ppt = None
        self._resolution = 0.0

    def update(self, data):
        """Override."""
        try:
            correlator, foms, info = getattr(data.correlation,
                                             f'param{self._idx}')
        except AttributeError:
            return

        device_id = info['device_id']
        ppt = info['property']
        if self._device_id != device_id or self._ppt != ppt:
            self.setLabel('bottom', f"{device_id + ' | ' + ppt} (arb. u.)")
            self._device_id = device_id
            self._ppt = ppt

        if isinstance(foms, np.ndarray):
            # PairData
            if self._resolution != 0.0:
                self._resolution = 0.0
                self._bar.setData([], [], beam=0.0)
                self._plot.setBrush(self._brushes[self._idx])

            self._plot.setData(correlator, foms)
            # make auto-range of the viewbox work correctly
            self._bar.setData(correlator, foms)
        else:
            # AccumulatedPairData
            resolution = info['resolution']

            if self._resolution != resolution:
                self._resolution = resolution
                self._bar.setData([], [], beam=resolution)
                self._plot.setBrush(self._opaque_brushes[self._idx])

            self._bar.setData(x=correlator,
                              y=foms.avg, y_min=foms.min, y_max=foms.max)
            self._plot.setData(correlator, foms.avg)


class PumpProbeOnOffWidget(PlotWidget):
    """PumpProbeOnOffWidget class.

    Widget for displaying the pump and probe signal or their difference.
    """
    def __init__(self, diff=False, *, parent=None):
        """Initialization.

        :param bool diff: True for displaying on-off while False for
            displaying on and off
        """
        super().__init__(parent=parent)

        # self.setLabel('left', "Scattering signal (arb. u.)")
        # self.setLabel('bottom', "Momentum transfer (1/A)")
        self.setLabel('left', "y (arb. u.)")
        self.setLabel('bottom', "x (arb. u.)")
        self.addLegend(offset=(-40, 20))

        self._is_diff = diff
        if diff:
            self._on_off_pulse = self.plotCurve(name="On - Off", pen=make_pen("y"))
        else:
            self._on_pulse = self.plotCurve(name="On", pen=make_pen("d"))
            self._off_pulse = self.plotCurve(name="Off", pen=make_pen("g"))

    def update(self, data):
        """Override."""
        x, _, _ = data.pp.data
        on = data.pp.norm_on_ma
        off = data.pp.norm_off_ma
        on_off = data.pp.norm_on_off_ma

        if on is None or off is None:
            return

        if isinstance(on, np.ndarray) and on.ndim > 1:
            # call reset() to reset() plots from other analysis types
            self.reset()
            return

        if self._is_diff:
            self._on_off_pulse.setData(x, on_off)
        else:
            self._on_pulse.setData(x, on)
            self._off_pulse.setData(x, off)


class PumpProbeFomWidget(PlotWidget):
    """PumpProbeFomWidget class.

    Widget for displaying the evolution of FOM in pump-probe analysis.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Train ID")
        self.setLabel('left', "FOM (arb. u.)")

        self._plot = self.plotScatter(brush=make_brush('o'))

    def update(self, data):
        """Override."""
        tids, foms, _ = data.pp.fom
        self._plot.setData(tids, foms)


class XasSpectrumWidget(PlotWidget):
    """XasSpectrumWidget class.

    Widget for displaying the XAS spectra.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Energy (eV)")
        self.setLabel('left', "Absorption")

        self._spectrum1 = self.plotScatter(
            name="ROI2/ROI1", brush=make_brush('p'), size=12)
        self._spectrum2 = self.plotScatter(
            name="ROI3/ROI1", brush=make_brush('g'), size=12)

        self.addLegend(offset=(-40, 20))

    def update(self, data):
        """Override."""
        bin_center = data.xas.bin_center
        absorptions = data.xas.absorptions

        self._spectrum1.setData(bin_center, absorptions[0])
        self._spectrum2.setData(bin_center, absorptions[1])


class XasSpectrumDiffWidget(PlotWidget):
    """XasSpectrumDiffWidget class.

    Widget for displaying the difference of two XAS spectra.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Energy (eV)")
        self.setLabel('left', "Absorption")

        self._plot = self.plotScatter(brush=make_brush('b'), size=12)

    def update(self, data):
        """Override."""
        bin_center = data.xas.bin_center
        absorptions = data.xas.absorptions

        self._plot.setData(bin_center, absorptions[1] - absorptions[0])


class XasSpectrumBinCountWidget(PlotWidget):
    """XasSpectrumBinCountWidget class.

    Widget for displaying the number of data points in each energy bins.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Energy (eV)")
        self.setLabel('left', "Count")

        self._plot = self.plotBar(width=0.8)

    def update(self, data):
        """Override."""
        bin_center = data.xas.bin_center
        bin_count = data.xas.bin_count

        self._plot.setData(bin_center, bin_count)
