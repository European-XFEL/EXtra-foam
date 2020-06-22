"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtGui import QPainter, QPainterPath, QPicture
from PyQt5.QtCore import QRectF

from .. import pyqtgraph as pg

from ..misc_widgets import FColor


class CurvePlotItem(pg.PlotItem):
    """CurvePlotItem."""

    def __init__(self, x=None, y=None, *, pen=None):
        """Initialization."""
        super().__init__()

        self._path = None

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

        self._path = None
        # Schedules a redraw of the area covered by rect in this item.
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    def preparePath(self):
        p = QPainterPath()

        x, y = self._x, self._y
        if len(x) >= 2:
            p.moveTo(x[0], y[0])
            for px, py in zip(x[1:], y[1:]):
                p.lineTo(px, py)

        self._path = p

    def paint(self, painter, *args):
        """Override."""
        if self._path is None:
            self.preparePath()

        painter.setPen(self._pen)
        painter.drawPath(self._path)

    def boundingRect(self):
        """Override."""
        if self._path is None:
            self.preparePath()
        return self._path.boundingRect()


class BarGraphItem(pg.PlotItem):
    """BarGraphItem"""
    def __init__(self, x=None, y=None, *, width=1.0, pen=None, brush=None):
        """Initialization."""
        super().__init__()

        self._picture = None

        self._x = None
        self._y = None

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

        self._picture = None
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    def drawPicture(self):
        self._picture = QPicture()
        p = QPainter(self._picture)
        p.setPen(self._pen)
        p.setBrush(self._brush)

        # Now it works for bar plot with equalized gaps
        # TODO: extend it
        if len(self._x) > 1:
            width = self._width * (self._x[1] - self._x[0])
        else:
            width = self._width

        for x, y in zip(self._x, self._y):
            p.drawRect(QRectF(x - width/2, 0, width, y))

        p.end()

    def paint(self, painter, *args):
        """Override."""
        if self._picture is None:
            self.drawPicture()
        self._picture.play(painter)

    def boundingRect(self):
        """Override."""
        if self._picture is None:
            self.drawPicture()
        return QRectF(self._picture.boundingRect())


class StatisticsBarItem(pg.PlotItem):
    """StatisticsBarItem."""

    def __init__(self, x=None, y=None, *, y_min=None, y_max=None, beam=None,
                 line=False, pen=None):
        """Initialization.

        Note: y is not used for now.
        """
        super().__init__()

        self._path = None

        self._x = None
        self._y = None
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
        self._path = None
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    def setBeam(self, w):
        self._beam = w

    def isEmptyGraph(self):
        return not bool(len(self._x))

    def preparePath(self):
        p = QPainterPath()

        beam = self._beam
        for x, u, l in zip(self._x, self._y_min, self._y_max):
            # plot the lower horizontal lines
            p.moveTo(x - beam / 2., l)
            p.lineTo(x + beam / 2., l)

            # plot the vertical line
            p.moveTo(x, l)
            p.lineTo(x, u)

            # plot the upper horizontal line
            p.moveTo(x - beam / 2., u)
            p.lineTo(x + beam / 2., u)

        if self._line and len(self._x) > 2:
            p.moveTo(self._x[-1], self._y[-1])
            for x, y in zip(reversed(self._x[:-1]), reversed(self._y[:-1])):
                p.lineTo(x, y)

        self._path = p

    def paint(self, painter, *args):
        """Override."""
        if self._path is None:
            self.preparePath()

        painter.setPen(self._pen)
        painter.drawPath(self._path)

    def boundingRect(self):
        """Override."""
        if self._path is None:
            self.preparePath()
        return self._path.boundingRect()
