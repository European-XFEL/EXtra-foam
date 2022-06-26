"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPainter, QPainterPath, QPicture

from ..misc_widgets import FColor


class FoamPlotDataItem(pg.PlotDataItem):
    def setData(self, *args, **kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            if args[0] is None and args[1] is None:
                return

        super().setData(*args, **kwargs)

    def data(self):
        return self.getData()

    def setLogX(self, state: bool):
        self.setLogMode(state, self.opts["logMode"][1])

    def setLogY(self, state: bool):
        self.setLogMode(self.opts["logMode"][0], state)

    @property
    def _log_x_mode(self):
        return self.opts["logMode"][0]

    @property
    def _log_y_mode(self):
        return self.opts["logMode"][1]


class CurvePlotItem(FoamPlotDataItem):
    def __init__(self, x=None, y=None, pen=None, name=None, parent=None):
        pen = FColor.mkPen('g') if pen is None else pen

        super().__init__(x=x, y=y, pen=pen, name=name, parent=parent,
                         connect="all", symbol=None)


class ScatterPlotItem(FoamPlotDataItem):
    def __init__(self, x=None, y=None, symbol="o", size=8,
                 pen=None, brush=None, name=None, parent=None):
        # Set reasonable defaults for the pen and brush
        if pen is None and brush is None:
            pen = FColor.mkPen(None)
            brush = FColor.mkBrush('b')
        else:
            pen = FColor.mkPen(None) if pen is None else pen
            brush = FColor.mkBrush(None) if brush is None else brush

        super().__init__(x=x, y=y, symbol=symbol, symbolSize=size, symbolPen=pen,
                         symbolBrush=brush, name=name, parent=None,
                         pen=None)


class FoamPlotItem(pg.GraphicsObject):
    def __init__(self, name=None, parent=None):
        super().__init__(parent)

        self._graph = None
        self.opts = { }
        self._name = "" if name is None else name

        self._log_x_mode = False
        self._log_y_mode = False

    @abc.abstractmethod
    def setData(self, *args, **kwargs):
        raise NotImplementedError

    def _parseInputData(self, x, y, **kwargs):
        """Convert input to np.array and apply shape check."""
        if isinstance(x, list):
            x = np.array(x)
        elif x is None:
            x = np.array([])

        if isinstance(y, list):
            y = np.array(y)
        elif y is None:
            y = np.array([])

        if len(x) != len(y):
            raise ValueError("'x' and 'y' data have different lengths!")

        # do not set data unless they pass the sanity check!
        self._x, self._y = x, y

    @abc.abstractmethod
    def data(self):
        raise NotImplementedError

    def updateGraph(self):
        self._graph = None
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    @abc.abstractmethod
    def _prepareGraph(self):
        raise NotImplementedError

    def paint(self, p, *args):
        """Override."""
        if self._graph is None:
            self._prepareGraph()
        p.setPen(self._pen)
        p.drawPath(self._graph)

    def boundingRect(self):
        """Override."""
        if self._graph is None:
            self._prepareGraph()
        return self._graph.boundingRect()

    def setLogX(self, state):
        """Set log mode for x axis."""
        self._log_x_mode = state
        self.updateGraph()

    def setLogY(self, state):
        """Set log mode for y axis."""
        self._log_y_mode = state
        self.updateGraph()

    def transformedData(self):
        """Transform and return the internal data to log scale if requested.

        Child class should re-implement this method if it has a
        different internal data structure.
        """
        return (self.toLogScale(self._x) if self._log_x_mode else self._x,
                self.toLogScale(self._y) if self._log_y_mode else self._y)

    @staticmethod
    def toLogScale(arr, policy=None):
        """Convert array result to logarithmic scale."""
        ret = np.nan_to_num(arr)
        ret[ret < 0] = 0
        return np.log10(ret + 1)

    def name(self):
        """An identity of the PlotItem.

        Used in LegendItem.
        """
        return self._name

    def drawSample(self, p):
        """Draw a sample used in LegendItem."""
        pass


class StatisticsBarItem(FoamPlotItem):
    def __init__(self, x=None, y=None, *, y_min=None, y_max=None, beam=None,
                 line=False, pen=None,
                 name=None, parent=None):
        """Initialization.

        Note: y is not used for now.
        """
        super().__init__(name=name, parent=parent)

        self._x = None
        self._y = None
        self._y_min = None
        self._y_max = None

        self._beam = 0.0 if beam is None else beam
        self._line = line
        self.opts["pen"] = FColor.mkPen('p') if pen is None else pen
        self._pen = self.opts["pen"]

        self.setData(x, y, y_min=y_min, y_max=y_max)

    def setData(self, x, y, y_min=None, y_max=None, beam=None):
        """Override."""
        self._parseInputData(x, y, y_min=y_min, y_max=y_max)

        if beam is not None:
            # keep the default beam if not specified
            self._beam = beam

        self.updateGraph()

    def _parseInputData(self, x, y, **kwargs):
        """Override."""
        if isinstance(x, list):
            x = np.array(x)
        elif x is None:
            x = np.array([])

        if isinstance(y, list):
            y = np.array(y)
        elif y is None:
            y = np.array([])

        if len(x) != len(y):
            raise ValueError("'x' and 'y' data have different lengths!")

        y_min = kwargs.get('y_min', None)
        if isinstance(y_min, list):
            y_min = np.array(y_min)
        elif y_min is None:
            y_min = y

        y_max = kwargs.get('y_max', None)
        if isinstance(y_max, list):
            y_max = np.array(y_max)
        elif y_max is None:
            y_max = y

        if not len(y) == len(y_min) == len(y_max):
            raise ValueError(
                "'y_min' and 'y_max' data have different lengths!")

        # do not set data unless they pass the sanity check!
        self._x, self._y = x, y
        self._y_min, self._y_max = y_min, y_max

    def data(self):
        """Override."""
        return self._x, self._y, self._y_min, self._y_max

    def setBeam(self, w):
        self._beam = w

    def _prepareGraph(self):
        p = QPainterPath()

        x, y, y_min, y_max = self.transformedData()
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

    def transformedData(self):
        """Override."""
        y_min = self.toLogScale(self._y_min) if self._log_y_mode else self._y_min
        y_max = self.toLogScale(self._y_max) if self._log_y_mode else self._y_max
        return super().transformedData() + (y_min, y_max)


class BarGraphItem(FoamPlotItem):
    """BarGraphItem"""
    def __init__(self, x=None, y=None, *, width=1.0, pen=None, brush=None,
                 name=None, parent=None):
        """Initialization."""
        super().__init__(name=name, parent=parent)

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

        # This field is for compatibility with pyqtgraph
        self.opts = { "pen": self.pen }

        self.setData(x, y)

    @property
    def pen(self):
        return self._pen

    def setData(self, x, y):
        """Override."""
        self._parseInputData(x, y)
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

        x, y = self.transformedData()
        # Now it works for bar plot with equalized gaps
        # TODO: extend it
        if len(x) > 1:
            width = self._width * (x[1] - x[0])
        else:
            width = self._width

        for px, py in zip(x, y):
            p.drawRect(QRectF(px - width/2, 0, width, py))

        p.end()

    def paint(self, p, *args):
        """Override."""
        if self._graph is None:
            self._prepareGraph()
        self._graph.play(p)

    def boundingRect(self):
        """Override."""
        return QRectF(super().boundingRect())

    def drawSample(self, p):
        """Override."""
        p.setBrush(self._brush)
        p.setPen(self._pen)
        # Legend sample has a bounding box of (0, 0, 20, 20)
        p.drawRect(QRectF(2, 2, 18, 18))
