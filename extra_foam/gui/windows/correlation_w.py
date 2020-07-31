"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QFrame, QSplitter, QVBoxLayout

from .base_window import _AbstractPlotWindow
from ..ctrl_widgets import CorrelationCtrlWidget
from ..misc_widgets import FColor
from ..plot_widgets import TimedPlotWidgetF
from ...config import config


class CorrelationPlot(TimedPlotWidgetF):
    """CorrelationPlot class.

    Widget for displaying correlations between FOM and different parameters.
    """
    _colors = config["GUI_CORRELATION_COLORS"]
    _pens = [(FColor.mkPen(pair[0]), FColor.mkPen(pair[1])) for pair in _colors]
    _brushes = [(FColor.mkBrush(pair[0], alpha=120),
                 FColor.mkBrush(pair[1], alpha=120)) for pair in _colors]

    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._idx = idx

        self.setTitle(f'Correlation {idx+1}')
        self._default_x_label = "Correlator (arb. u.)"
        self._default_y_label = "FOM (arb. u.)"
        self.addLegend(offset=(-40, 20))
        self.hideLegend()

        self._source = ""
        self._resolution = 0.0

        self.updateLabel()

        self._newScatterPlot()

    def refresh(self):
        """Override."""
        item = self._data.corr[self._idx]

        src = item.source
        if src != self._source:
            self._source = src
            self.updateLabel()

        resolution = item.resolution
        y = item.y
        y_slave = item.y_slave
        if resolution == 0:
            if self._resolution != 0:
                # bar -> scatter plot
                self._removeBothPlots()
                self._newScatterPlot()
                self._resolution = 0

            self._plot.setData(item.x, y)
            # The following code looks awkward but it is by far
            # the best solution. The user could choose to deactivate
            # the master-slave mode and keep the slave data there.
            # In this case, a legend is still required. Therefore,
            # we cannot toggle the legend by the signal from the
            # master-slave checkbox in the GUI.
            if y_slave is not None:
                self._plot_slave.setData(item.x_slave, y_slave)
                if len(y_slave) > 0:
                    self.showLegend()
                else:
                    self.hideLegend()
            else:
                self.hideLegend()
        else:
            if resolution != self._resolution:
                if self._resolution == 0:
                    # scatter -> bar plot
                    self._removeBothPlots()
                    self._newStatisticsBarPlot(resolution)
                else:
                    # update beam
                    self._plot.setBeam(resolution)
                    self._plot_slave.setBeam(resolution)
                self._resolution = resolution

            self._plot.setData(item.x, y.avg, y_min=y.min, y_max=y.max)
            if y_slave is not None:
                self._plot_slave.setData(
                    item.x_slave, y_slave.avg,
                    y_min=y_slave.min, y_max=y_slave.max)
                if len(y_slave.avg) > 0:
                    self.showLegend()
                else:
                    self.hideLegend()
            else:
                self.hideLegend()

    def updateLabel(self):
        src = self._source
        if src:
            new_label = f"{src} (arb. u.)"
        else:
            new_label = self._default_x_label
        self.setLabel('bottom', new_label)

        self.setLabel('left', self._default_y_label)

    def _newScatterPlot(self):
        brush_pair = self._brushes[self._idx]
        pen_pair = self._pens[self._idx]

        self._plot = self.plotScatter(brush=brush_pair[0], name="master")
        self._plot_slave = self.plotScatter(brush=brush_pair[1], name="slave")

        self._fitted = self.plotCurve(pen=pen_pair[0])

    def _newStatisticsBarPlot(self, resolution):
        pen_pair = self._pens[self._idx]

        self._plot = self.plotStatisticsBar(
            beam=resolution, pen=pen_pair[0], name="master")
        self._plot_slave = self.plotStatisticsBar(
            beam=resolution, pen=pen_pair[1], name="slave")

        self._fitted = self.plotCurve(pen=pen_pair[0])

    def _removeBothPlots(self):
        self.removeItem(self._fitted)
        self.removeItem(self._plot)
        self.removeItem(self._plot_slave)

    def data(self):
        return self._plot.data()[:2], self._plot_slave.data()[:2]

    def setFitted(self, x, y):
        self._fitted.setData(x, y)


class CorrelationWindow(_AbstractPlotWindow):
    """CorrelationWindow class.

    Visualize correlation.
    """
    _title = "Correlation"

    _TOTAL_W, _TOTAL_H = config['GUI_PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._ctrl_widget = self.createCtrlWidget(CorrelationCtrlWidget)

        self._corr1 = CorrelationPlot(0, parent=self)
        self._corr2 = CorrelationPlot(1, parent=self)

        self.initUI()
        self.initConnections()
        self.loadMetaData()
        self.updateMetaData()

        self.resize(self._TOTAL_W, self._TOTAL_H)
        self.setMinimumSize(0.6*self._TOTAL_W, 0.6*self._TOTAL_H)

        self.update()

    def initUI(self):
        """Override."""
        plots = QSplitter()
        plots.addWidget(self._corr1)
        plots.addWidget(self._corr2)
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

    def _onCurveFit(self, is_corr1):
        corr = self._corr1 if is_corr1 else self._corr2

        data, _ = corr.data()

        x, y = self._ctrl_widget.fit_curve(*data)
        corr.setFitted(x, y)

    def _onClearFitting(self, is_corr1):
        corr = self._corr1 if is_corr1 else self._corr2
        corr.setFitted([], [])

    def closeEvent(self, QCloseEvent):
        self._ctrl_widget.resetAnalysisType()
        super().closeEvent(QCloseEvent)
