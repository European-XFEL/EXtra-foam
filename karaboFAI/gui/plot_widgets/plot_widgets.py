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

from ..misc_widgets import make_brush, make_pen, SequentialColors
from ...logger import logger
from ...config import config


class SinglePulseAiWidget(PlotWidget):
    """SinglePulseAiWidget class.

    A widget which allows user to visualize the the azimuthal integration
    result of a single pulse. The azimuthal integration result is also
    compared with the average azimuthal integration of all the pulses.
    """
    def __init__(self, *, pulse_index=0, plot_mean=True, parent=None):
        """Initialization.

        :param int pulse_index: ID of the pulse to be plotted.
        :param bool plot_mean: whether to plot the mean AI of all pulses
            if the data is pulse resolved.
        """
        super().__init__(parent=parent)

        self.pulse_index = pulse_index

        self.setLabel('left', "Scattering signal (arb. u.)")
        self.setLabel('bottom', "Momentum transfer (1/A)")

        if plot_mean:
            self.addLegend(offset=(-40, 20))

        if plot_mean:
            self._mean_plot = self.plotCurve(name="mean", pen=make_pen("g"))
        else:
            self._mean_plot = None

        self._pulse_plot = self.plotCurve(name="pulse_plot", pen=make_pen("p"))

    def update(self, data):
        """Override."""
        momentum = data.ai.momentum
        intensities = data.ai.intensities

        if intensities is None:
            return

        max_id = data.n_pulses - 1
        if self.pulse_index <= max_id:
            self._pulse_plot.setData(momentum, intensities[self.pulse_index])
        else:
            logger.error("<VIP pulse index>: VIP pulse index ({}) > Maximum "
                         "pulse index ({})".format(self.pulse_index, max_id))
            return

        if self._mean_plot is not None:
            self._mean_plot.setData(momentum, data.ai.intensity)


class TrainAiWidget(PlotWidget):
    """TrainAiWidget class.

    Widget for displaying azimuthal integration result for all
    the pulse(s) in a train.
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

        if data.pulse_resolved:
            intensities = data.ai.intensities
            if intensities is None:
                return

            n_pulses = len(intensities)
            if self._n_pulses != n_pulses:
                self.clear()

                colors = SequentialColors().s1(n_pulses)
                for i, intensity in enumerate(intensities):
                    self.plotCurve(momentum, intensity, pen=make_pen(colors[i]))
            else:
                for item, intensity in zip(self.plotItem.items, intensities):
                    item.setData(momentum, intensity)

        else:
            intensity = data.ai.intensity
            if intensity is None:
                return

            if self._n_pulses == 0:
                # initialize
                self.plotCurve(momentum, intensity,
                               pen=make_pen(SequentialColors().r[0]))
                self._n_pulses = 1
            else:
                self.plotItem.items[0].setData(momentum, intensity)


class PulsedFOMWidget(PlotWidget):
    """PulsedFOMWidget class.

    A widget which allows users to monitor the azimuthal integration FOM
    of each pulse with respect to the first pulse in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._plot = self.plotBar()

        self.setLabel('left', "Integrated difference (arb.)")
        self.setLabel('bottom', "Pulse index")
        self.setTitle('FOM with respect to the first pulse')

    def update(self, data):
        """Override."""
        foms = data.ai.intensities_foms
        if foms is None:
            return

        self._plot.setData(range(len(foms)), foms)


class RoiValueMonitor(PlotWidget):
    """RoiValueMonitor class.

    Widget for displaying the evolution of the value (integration, median,
    mean) of ROIs.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

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
            plot.setData(tids, roi_hist)


class CorrelationWidget(PlotWidget):
    """CorrelationWidget class.

    Widget for displaying correlations between FOM and different parameters.
    """
    _colors = config["CORRELATION_COLORS"]
    _pens = [make_pen(color) for color in _colors]
    _brushes = [make_brush(color, 120) for color in _colors]
    _opaque_brushes = [make_brush(color) for color in _colors]

    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._idx = idx # start from 1

        self.setLabel('left', "FOM (arb. u.)")
        self.setLabel('bottom', "Correlator (arb. u.)")

        self._bar = self.plotErrorBar(pen=self._pens[self._idx-1])
        self._plot = self.plotScatter(brush=self._brushes[self._idx-1])

        self._device_id = None
        self._ppt = None
        self._resolution = 0.0

    def update(self, data):
        """Override."""
        try:
            correlator_hist, fom_hist, info = getattr(
                data.correlation, f'correlation{self._idx}')
        except AttributeError:
            return

        device_id = info['device_id']
        ppt = info['property']
        if self._device_id != device_id or self._ppt != ppt:
            self.setLabel('bottom', f"{device_id + ' | ' + ppt} (arb. u.)")
            self._device_id = device_id
            self._ppt = ppt

        if isinstance(fom_hist, np.ndarray):
            # PairData
            if self._resolution != 0.0:
                self._resolution = 0.0
                self._bar.setData([], [], beam=0.0)
                self._plot.setBrush(self._brushes[self._idx-1])

            self._plot.setData(correlator_hist, fom_hist)
            # make auto-range of the viewbox work correctly
            self._bar.setData(correlator_hist, fom_hist)
        else:
            # AccumulatedPairData
            resolution = info['resolution']

            if self._resolution != resolution:
                self._resolution = resolution
                self._bar.setData([], [], beam=resolution)
                self._plot.setBrush(self._opaque_brushes[self._idx-1])

            self._bar.setData(x=correlator_hist,
                              y=fom_hist.avg,
                              y_min=fom_hist.min,
                              y_max=fom_hist.max)
            self._plot.setData(correlator_hist, fom_hist.avg)


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
            self._on_off_pulse = self.plotCurve(name="On - Off", pen=make_pen("p"))
        else:
            self._on_pulse = self.plotCurve(name="On", pen=make_pen("r"))
            self._off_pulse = self.plotCurve(name="Off", pen=make_pen("b"))

    def update(self, data):
        """Override."""
        x = data.pp.x
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

        self._plot = self.plotScatter(brush=make_brush('g'))

    def update(self, data):
        """Override."""
        tids, fom_hist, _ = data.pp.fom_hist
        self._plot.setData(tids, fom_hist)


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
            name="ROI2/ROI1", brush=make_brush('r'), size=12)
        self._spectrum2 = self.plotScatter(
            name="ROI3/ROI1", brush=make_brush('b'), size=12)

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

        self._plot = self.plotBar()

    def update(self, data):
        """Override."""
        bin_center = data.xas.bin_center
        bin_count = data.xas.bin_count

        self._plot.setData(bin_center, bin_count)


class BinWidget(PlotWidget):
    """BinWidget class.

    Widget for displaying the pump and probe signal or their difference.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._n_bins = 0

        self.setLabel('left', "y (arb. u.)")
        self.setLabel('bottom', "x (arb. u.)")
        self.addLegend(offset=(-40, 20))

    def update(self, data):
        """Override."""
        bin_center = data.bin.center_x
        data_x = data.bin.data_x
        x = data.bin.x

        if data_x is None:
            return

        n_bins = len(bin_center)
        if self._n_bins != n_bins:
            self.clear()

            self._n_bins = n_bins

            colors = SequentialColors().s1(n_bins)

            bin_width = bin_center[1] - bin_center[0]
            for i, v in enumerate(data_x):
                start = bin_center[i] - bin_width/2.
                end = bin_center[i] + bin_width/2.
                self.plotCurve(x, v,
                               name=f"{start:>8.2e}, {end:>8.2e}",
                               pen=make_pen(colors[i]))
        else:
            for item, v in zip(self.plotItem.items, data_x):
                item.setData(x, v)


class BinCountWidget(PlotWidget):
    """BinCountWidget class.

    Widget for displaying the number of data points in each bins.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "x")
        self.setLabel('left', "Count")

        self._plot = self.plotBar()

    def update(self, data):
        """Override."""
        center = data.bin.center_x
        count = data.bin.count_x

        if count is None:
            return

        self._plot.setData(center, count)
