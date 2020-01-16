"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractPlotWindow
from ..misc_widgets import make_brush, make_pen
from ..plot_widgets import PlotWidgetF, TimedPlotWidgetF
from ...config import config


class InTrainFomPlot(PlotWidgetF):
    """InTrainFomPlot class.

    A widget which allows users to monitor the FOM of each pulse in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._plot = self.plotScatter()

        self.setLabel('left', "FOM")
        self.setLabel('bottom', "Pulse index")
        self.setTitle('Pulse resolved FOMs in a train')

    def updateF(self, data):
        """Override."""
        foms = data.hist.pulse_foms
        if foms is None:
            self.reset()
        else:
            self._plot.setData(range(len(foms)), foms)


class FomHist(TimedPlotWidgetF):
    """FomHist class

    Plot statistics of accumulated FOMs from different analysis.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setTitle("FOM Histogram")
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'FOM')
        self._plot = self.plotBar()

    def refresh(self):
        """Override."""
        item = self._data.hist
        hist, bin_centers = item.hist, item.bin_centers

        if bin_centers is None:
            self.reset()
        else:
            self._plot.setData(bin_centers, hist)


class CorrelationPlot(TimedPlotWidgetF):
    """CorrelationPlot class.

    Widget for displaying correlations between FOM and different parameters.
    """
    _colors = config["CORRELATION_COLORS"]
    _pens = [make_pen(color) for color in _colors]
    _brushes = [make_brush(color, 120) for color in _colors]
    _opaque_brushes = [make_brush(color) for color in _colors]

    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._idx = idx

        self.setTitle(f'Correlation {idx+1}')
        self._default_x_label = "Correlator (arb. u.)"
        self._default_y_label = "FOM (arb. u.)"

        self._device_id = ""
        self._ppt = ""
        self._resolution = 0.0

        self.updateLabel(self._device_id, self._ppt)

        self._newScatterPlot()

    def refresh(self):
        """Override."""
        item = self._data.corr[self._idx]

        device_id = item.device_id
        ppt = item.property

        if device_id != self._device_id or ppt != self._ppt:
            self.updateLabel(device_id, ppt)
            self._device_id = device_id
            self._ppt = ppt

        resolution = item.resolution
        y = item.y
        if resolution == 0:
            # SimplePairSequence
            if self._resolution != 0:
                self._newScatterPlot()
                self._resolution = 0

            self._plot.setData(item.x, y)
        else:
            # OneWayAccuPairSequence
            if self._resolution == 0:
                self._newErrorBarPlot(resolution)
                self._resolution = resolution
            self._plot.setData(item.x, y.avg, y_min=y.min, y_max=y.max)

    def updateLabel(self, device_id, ppt):
        if device_id and ppt:
            new_label = f"{device_id + ' | ' + ppt} (arb. u.)"
        else:
            new_label = self._default_x_label
        self.setLabel('bottom', new_label)

        self.setLabel('left', self._default_y_label)

    def _newScatterPlot(self):
        self.clear()
        self._plot = self.plotScatter(brush=self._brushes[self._idx-1])

    def _newErrorBarPlot(self, resolution):
        self.clear()
        self._plot = self.plotErrorBar(beam=resolution,
                                       pen=self._pens[self._idx-1])


class StatisticsWindow(_AbstractPlotWindow):
    """StatisticsWindow class.

    Visualize statistics.
    """
    _title = "Statistics"

    _TOTAL_W, _TOTAL_H = config['GUI']['PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._pulse_fom = InTrainFomPlot(parent=self)
        self._fom_hist = FomHist(parent=self)

        self._corr1 = CorrelationPlot(0, parent=self)
        self._corr2 = CorrelationPlot(1, parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        self._cw = QSplitter()
        left_panel = QSplitter(Qt.Vertical)
        right_panel = QSplitter(Qt.Vertical)
        self._cw.addWidget(left_panel)
        self._cw.addWidget(right_panel)
        self._cw.setSizes([1, 1])
        self.setCentralWidget(self._cw)

        left_panel.addWidget(self._pulse_fom)
        left_panel.addWidget(self._fom_hist)
        left_panel.setSizes([1, 1])

        right_panel.addWidget(self._corr1)
        right_panel.addWidget(self._corr2)
        left_panel.setSizes([1, 1])

    def initConnections(self):
        """Override."""
        pass
