"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from enum import IntEnum
from collections import deque

import numpy as np

from PyQt5.QtCore import QTimer

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets import PlotWidgetF

app = mkQApp()


class PlotType(IntEnum):
    Line = 0
    Bar = 1
    StatisticsBar = 2
    Scatter = 3


class BenchmarkPlotItemSpeed:
    def __init__(self, plot_type=PlotType.Line):
        self._dt = deque(maxlen=60)

        self._timer = QTimer()
        self._timer.timeout.connect(self.update)
        # self._timer.singleShot(1, self.update)

        self._widget = PlotWidgetF()
        self._widget.addLegend()

        if plot_type == PlotType.Line:
            self._graph = self._widget.plotCurve(name='line')
            n_pts = 5000
        elif plot_type == PlotType.Bar:
            self._graph = self._widget.plotBar(name='bar')
            n_pts = 300
        elif plot_type == PlotType.StatisticsBar:
            self._graph = self._widget.plotStatisticsBar(name='statistics bar')
            self._graph.setBeam(1)
            n_pts = 500
        elif plot_type == PlotType.Scatter:
            self._graph = self._widget.plotScatter(name='scatter')
            n_pts = 5000
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        self._x = np.arange(n_pts)
        self._data = 100 * np.random.normal(size=(50, n_pts))
        if plot_type == PlotType.StatisticsBar:
            self._y_min = self._data - 20
            self._y_max = self._data + 20
        self._plot_type = plot_type

        self._prev_t = None
        self._count = 0

        self._widget.show()

    def start(self):
        self._prev_t = time.time()
        self._timer.start(0)

    def update(self):
        idx = self._count % 10
        if self._plot_type == PlotType.StatisticsBar:
            self._graph.setData(self._x, self._data[idx],
                                y_min=self._y_min[idx], y_max=self._y_max[idx])
        else:
            self._graph.setData(self._x, self._data[idx])

        self._count += 1

        now = time.time()
        self._dt.append(now - self._prev_t)
        self._prev_t = now
        fps = len(self._dt) / sum(self._dt)

        self._widget.setTitle(f"{fps:.2f} fps")

        app.processEvents()  # force complete redraw for every plot


if __name__ == '__main__':
    bench = BenchmarkPlotItemSpeed(PlotType.Scatter)
    bench.start()
    app.exec_()
