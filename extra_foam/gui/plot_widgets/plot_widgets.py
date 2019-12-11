"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .plot_widget_base import PlotWidgetF

from ..misc_widgets import make_brush, make_pen
from ...config import config


class PulsesInTrainFomWidget(PlotWidgetF):
    """PulsesInTrainFomWidget class.

    A widget which allows users to monitor the FOM of each pulse in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._plot = self.plotBar()

        self.setLabel('left', "FOM")
        self.setLabel('bottom', "Pulse index")
        self.setTitle('Pulse resolved FOM in a train')

    def updateF(self, data):
        """Override."""
        fom_hist = data.st.fom_hist
        if fom_hist is None:
            self.reset()
            return
        self._plot.setData(range(len(fom_hist)), fom_hist)


class FomHistogramWidget(PlotWidgetF):
    """StatisticsWidget class

    Plot statistics of accumulated FOMs from different analysis
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setTitle("FOM Histogram")
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'FOM')
        self._plot = self.plotBar(pen=make_pen('g'), brush=make_brush('b'))

    def updateF(self, data):
        center = data.st.fom_bin_center
        counts = data.st.fom_count
        if center is None:
            self.reset()
            return
        self._plot.setData(center, counts)


class CorrelationWidget(PlotWidgetF):
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

        self._idx = idx  # start from 1

        self.setTitle('')
        self.setLabel('left', "FOM (arb. u.)")
        self.setLabel('bottom', "Correlator (arb. u.)")

        self._bar = self.plotErrorBar(pen=self._pens[self._idx-1])
        self._plot = self.plotScatter(brush=self._brushes[self._idx-1])

        self._device_id = None
        self._ppt = None
        self._resolution = 0.0

    def updateF(self, data):
        """Override."""
        correlator_hist, fom_hist, info = getattr(
            data.corr, f'correlation{self._idx}').hist

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


class PumpProbeOnOffWidget(PlotWidgetF):
    """PumpProbeOnOffWidget class.

    Widget for displaying the pump and probe signal or their difference.
    """
    def __init__(self, diff=False, *, parent=None):
        """Initialization.

        :param bool diff: True for displaying on-off while False for
            displaying on and off
        """
        super().__init__(parent=parent)

        self.setLabel('left', "y (arb. u.)")
        self.setLabel('bottom', "x (arb. u.)")
        self.addLegend(offset=(-40, 20))

        self._is_diff = diff
        if diff:
            self._on_off_pulse = self.plotCurve(name="On - Off", pen=make_pen("p"))
        else:
            self._on_pulse = self.plotCurve(name="On", pen=make_pen("r"))
            self._off_pulse = self.plotCurve(name="Off", pen=make_pen("b"))

    def updateF(self, data):
        """Override."""
        x = data.pp.x
        y_on = data.pp.y_on
        y_off = data.pp.y_off
        y = data.pp.y

        if y_on is None or y_off is None:
            return

        if isinstance(y_on, np.ndarray) and y_off.ndim > 1:
            # call reset() to reset() plots from other analysis types
            self.reset()
            return

        if self._is_diff:
            self._on_off_pulse.setData(x, y)
        else:
            self._on_pulse.setData(x, y_on)
            self._off_pulse.setData(x, y_off)


class PumpProbeFomWidget(PlotWidgetF):
    """PumpProbeFomWidget class.

    Widget for displaying the evolution of FOM in pump-probe analysis.
    """

    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Train ID")
        self.setLabel('left', "FOM (arb. u.)")

        self._plot = self.plotScatter(brush=make_brush('g'))

    def updateF(self, data):
        """Override."""
        tids, fom_hist, _ = data.pp.fom_hist
        self._plot.setData(tids, fom_hist)


class Bin1dHist(PlotWidgetF):
    """Bin1dHist class.

    Widget for visualizing histogram of count for 1D-binning.
    """
    def __init__(self, idx, *, count=False, parent=None):
        """Initialization.

        :param int idx: index of the binning parameter (must be 1 or 2).
        :param bool count: True for count plot and False for FOM plot.
        """
        super().__init__(parent=parent)

        self._idx = idx
        self._count = count

        self.setTitle('')
        self.setLabel('bottom', "Bin center")
        if count:
            self.setLabel('left', "Count")
            self._plot = self.plotBar(pen=make_pen('g'), brush=make_brush('b'))
        else:
            self.setLabel('left', "FOM")
            self._plot = self.plotScatter(brush=make_brush('p'))

    def updateF(self, data):
        """Override."""
        bin = getattr(data.bin, f"bin{self._idx}")
        if not bin.updated:
            return

        if self._count:
            hist = bin.count_hist
        else:
            hist = bin.fom_hist

        if hist is not None:
            self.setLabel('bottom', bin.label)
            self._plot.setData(bin.center, hist)
