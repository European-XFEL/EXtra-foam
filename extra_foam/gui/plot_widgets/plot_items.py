"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from PyQt5.QtGui import QPainter, QPainterPath, QPicture
from PyQt5.QtCore import QRectF

from .. import pyqtgraph as pg

from ..misc_widgets import FColor


class CurvePlotItem(pg.PlotItem):
    """CurvePlotItem."""

    def __init__(self, x=None, y=None, *, pen=None,
                 name=None, parent=None):
        """Initialization."""
        super().__init__(name=name, parent=parent)

        self._x = None
        self._y = None

        self._pen = FColor.mkPen('g') if pen is None else pen

        self.setData(x, y)

    def setData(self, x, y):
        """Override."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")

        self.updateGraph()

    def data(self):
        """Override."""
        return self._x, self._y

    def _prepareGraph(self):
        """Override."""
        p = QPainterPath()

        # TODO: use QDataStream to improve performance
        x, y = self._transformData()
        if len(x) >= 2:
            p.moveTo(x[0], y[0])
            for px, py in zip(x[1:], y[1:]):
                p.lineTo(px, py)

        self._graph = p

    def drawSample(self, p):
        """Override."""
        p.setPen(self._pen)
        # Legend sample has a bounding box of (0, 0, 20, 20)
        p.drawLine(0, 11, 20, 11)


class BarGraphItem(pg.PlotItem):
    """BarGraphItem"""
    def __init__(self, x=None, y=None, *, width=1.0, pen=None, brush=None,
                 name=None, parent=None):
        """Initialization."""
        super().__init__(name=name, parent=parent)

        self._x = None
        self._y = None

        self._name = name

        if width > 1.0 or width <= 0:
            width = 1.0
        self._width = width

        if pen is None and brush is None:
            self._pen = FColor.mkPen(None)
            self._brush = FColor.mkBrush('b')
        else:
            self._pen = FColor.mkPen(None) if pen is None else pen
            self._brush = FColor.mkBrush(None) if brush is None else brush

        self.setData(x, y)

    def setData(self, x, y):
        """Override."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")

        self.updateGraph()

    def data(self):
        """Override."""
        return self._x, self._y

    def _prepareGraph(self):
        """Override."""
        self._graph = QPicture()
        p = QPainter(self._graph)
        p.setPen(self._pen)
        p.setBrush(self._brush)

        x, y = self._transformData()
        # Now it works for bar plot with equalized gaps
        # TODO: extend it
        if len(x) > 1:
            width = self._width * (x[1] - x[0])
        else:
            width = self._width

        for px, py in zip(x, y):
            p.drawRect(QRectF(px - width/2, 0, width, py))

        p.end()

    def paint(self, painter, *args):
        """Override."""
        if self._graph is None:
            self.drawPicture()
        self._graph.play(painter)

    def boundingRect(self):
        """Override."""
        return QRectF(super().boundingRect())

    def drawSample(self, p):
        """Override."""
        p.setBrush(self._brush)
        p.setPen(self._pen)
        # Legend sample has a bounding box of (0, 0, 20, 20)
        p.drawRect(QRectF(2, 2, 18, 18))


class StatisticsBarItem(pg.PlotItem):
    """StatisticsBarItem."""

    def __init__(self, x=None, y=None, *, y_min=None, y_max=None, beam=None,
                 line=False, pen=None,
                 name=None, parent=None):
        """Initialization.

        Note: y is not used for now.
        """
        super().__init__(name=name, parent=parent)

        self._x = None
        self._y = None

        self._name = name

        self._y_min = None
        self._y_max = None

        self._beam = 0.0 if beam is None else beam
        self._line = line
        self._pen = FColor.mkPen('p') if pen is None else pen

        self.setData(x, y, y_min=y_min, y_max=y_max)

    def setData(self, x, y, y_min=None, y_max=None, beam=None):
        """Override."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        self._y_min = self._y if y_min is None else y_min
        self._y_max = self._y if y_max is None else y_max
        if beam is not None:
            # keep the default beam if not specified
            self._beam = beam

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")
        if not len(self._y) == len(self._y_min) == len(self._y_max):
            raise ValueError(
                "'y_min' and 'y_max' data have different lengths!")

        self.updateGraph()

    def data(self):
        """Override."""
        return self._x, self._y, self._y_min, self._y_max

    def setBeam(self, w):
        self._beam = w

    def _prepareGraph(self):
        p = QPainterPath()

        x, y, y_min, y_max = self._transformData()
        beam = self._beam
        for px, u, l in zip(x, y_min, y_max):
            # plot the lower horizontal lines
            p.moveTo(px - beam / 2., l)
            p.lineTo(px + beam / 2., l)

            # plot the vertical line
            p.moveTo(px, l)
            p.lineTo(px, u)

            # plot the upper horizontal line
            p.moveTo(px - beam / 2., u)
            p.lineTo(px + beam / 2., u)

        if self._line and len(x) > 2:
            p.moveTo(x[-1], y[-1])
            for px, py in zip(reversed(x[:-1]), reversed(y[:-1])):
                p.lineTo(px, py)

        self._graph = p

    def drawSample(self, p):
        """Override."""
        p.setPen(self._pen)

        # Legend sample has a bounding box of (0, 0, 20, 20)
        p.drawLine(2, 2, 8, 2)  # lower horizontal line
        p.drawLine(5, 2, 5, 18)  # vertical line
        p.drawLine(2, 18, 8, 18)  # upper horizontal line

    def _transformData(self):
        """Override."""
        y_min = self.toLogScale(self._y_min) if self._log_y_mode else self._y_min
        y_max = self.toLogScale(self._y_max) if self._log_y_mode else self._y_max
        return super()._transformData() + (y_min, y_max)
