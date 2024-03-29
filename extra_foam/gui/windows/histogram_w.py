"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from string import Template

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QSplitter, QVBoxLayout

from .base_window import _AbstractPlotWindow
from ..ctrl_widgets import HistogramCtrlWidget
from ..misc_widgets import FColor
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
        foms = data["processed"].pulse.hist.pulse_foms
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
        self._fitted = self.plotCurve(pen=FColor.mkPen('r'))

        self._title_template = Template(
            f"FOM Histogram (mean: $mean, median: $median, std: $std)")
        self.updateTitle()
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'FOM')

    def refresh(self):
        """Override."""
        hist = self._data["processed"].hist
        bin_centers = hist.bin_centers
        if bin_centers is None:
            self.reset()
        else:
            self._plot.setData(bin_centers, hist.hist)
            self.updateTitle(hist.mean, hist.median, hist.std)

    def data(self):
        return self._plot.data()

    def setFitted(self, x, y):
        self._fitted.setData(x, y)


class HistogramWindow(_AbstractPlotWindow):
    """HistogramWindow class.

    Visualize histogram.
    """
    _title = "Histogram"

    _TOTAL_W, _TOTAL_H = config['GUI_PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._ctrl_widget = self.createCtrlWidget(HistogramCtrlWidget)

        self._pulse_fom = InTrainFomPlot(parent=self)
        self._fom_hist = FomHist(parent=self)

        self.initUI()
        self.initConnections()
        self.loadMetaData()
        self.updateMetaData()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(int(0.6 * self._TOTAL_W), int(0.6 * self._TOTAL_H))

        self.update()

    def initUI(self):
        """Override."""
        plots = QSplitter(Qt.Horizontal)
        plots.addWidget(self._pulse_fom)
        plots.addWidget(self._fom_hist)
        plots.setSizes([1, 1])

        self._cw = QFrame()
        layout = QVBoxLayout()
        layout.addWidget(plots)
        layout.addWidget(self._ctrl_widget)
        self._ctrl_widget.setFixedHeight(
            self._ctrl_widget.minimumSizeHint().height())
        self._cw.setLayout(layout)
        self.setCentralWidget(self._cw)

    def initConnections(self):
        """Override."""
        self._ctrl_widget.fit_curve_sgn.connect(self._onCurveFit)
        self._ctrl_widget.clear_fitting_sgn.connect(self._onClearFitting)

    def _onCurveFit(self):
        data = self._fom_hist.data()

        x, y = self._ctrl_widget.fit_curve(*data)
        self._fom_hist.setFitted(x, y)

    def _onClearFitting(self):
        self._fom_hist.setFitted([], [])

    def closeEvent(self, QCloseEvent):
        self._ctrl_widget.resetAnalysisType()
        super().closeEvent(QCloseEvent)
