"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from string import Template

from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractPlotWindow
from ..plot_widgets import HistMixin, PlotWidgetF, TimedPlotWidgetF
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
        self.setTitle('Pulse-resolved FOMs in a train')

    def updateF(self, data):
        """Override."""
        foms = data.pulse.hist.pulse_foms
        if foms is None:
            self.reset()
        else:
            self._plot.setData(range(len(foms)), foms)


class FomHist(HistMixin, TimedPlotWidgetF):
    """FomHist class

    Plot statistics of accumulated FOMs from different analysis.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self._plot = self.plotBar()

        self._title_template = Template(
            f"FOM Histogram (mean: $mean, median: $median, std: $std)")
        self.updateTitle()
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'FOM')

    def refresh(self):
        """Override."""
        hist = self._data.hist
        bin_centers = hist.bin_centers
        if bin_centers is None:
            self.reset()
        else:
            self._plot.setData(bin_centers, hist.hist)
            self.updateTitle(hist.mean, hist.median, hist.std)


class HistogramWindow(_AbstractPlotWindow):
    """HistogramWindow class.

    Visualize histogram.
    """
    _title = "Histogram"

    _TOTAL_W, _TOTAL_H = config['GUI_PLOT_WINDOW_SIZE']
    _TOTAL_H /= 2

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._pulse_fom = InTrainFomPlot(parent=self)
        self._fom_hist = FomHist(parent=self)

        self.initUI()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        self._cw = QSplitter()
        self._cw.addWidget(self._pulse_fom)
        self._cw.addWidget(self._fom_hist)
        self._cw.setSizes([1, 1])

        self.setCentralWidget(self._cw)

    def initConnections(self):
        """Override."""
        pass
