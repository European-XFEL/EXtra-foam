"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import struct
from collections import OrderedDict

import numpy as np

from PyQt5.QtGui import (
    QImage, QPainter, QPainterPath, QPicture, QPixmap, QTransform, QFont
)
from PyQt5.QtCore import pyqtSignal, QByteArray, QDataStream, QRectF, Qt, QPointF

from .. import pyqtgraph as pg
from ..pyqtgraph import functions as fn
from ..misc_widgets import FColor
from ...utils import LinearROI as MetroLinearROI


class LinearROI(pg.LinearRegionItem):
    def __init__(self, plot_widget, *args, label="", pen=FColor.mkPen("g"), **kwargs):
        super().__init__(*args, pen=pen, **kwargs)

        self._plot_widget = plot_widget
        self._label = None

        self._label_item = pg.TextItem(color=pen.color(), angle=90)
        self._label_item.setParentItem(self)
        self.sigRegionChanged.connect(self._updateLabelPos)
        self._plot_widget.getViewBox().sigRangeChanged.connect(self._updateLabelPos)

        font = QFont()
        font.setPointSizeF(15)
        self._label_item.setFont(font)
        self.setLabel(label)

    def _updateLabelPos(self):
        plot_height = self._plot_widget.range.height()
        y_pos = self._plot_widget.getViewBox().mapSceneToView(QPointF(0, 0.5 * plot_height)).y()

        x_min, _ = self.getRegion()
        self._label_item.setPos(x_min, y_pos)

    def label(self):
        return self._label

    def setLabel(self, label):
        self._label_item.setText(label)

        if len(label) == 0:
            self._label_item.hide()
        else:
            self._label_item.show()

        self._label = label

    def configureFromMetroROI(self, metro_roi: MetroLinearROI):
        if not isinstance(metro_roi, MetroLinearROI):
            raise RuntimeError(f"Input must be a MetroLinearROI, not {type(metro_roi)}")

        self.setRegion((metro_roi.lower_bound, metro_roi.upper_bound))


class CurvePlotItem(pg.PlotItem):
    """CurvePlotItem."""

    def __init__(self, x=None, y=None, *,
                 pen=None, name=None, check_finite=True, parent=None):
        """Initialization."""
        super().__init__(name=name, parent=parent)

        self._x = None
        self._y = None

        self._pen = FColor.mkPen('g') if pen is None else pen

        self._check_finite = check_finite

        self.setData(x, y)

    def setData(self, x, y):
        """Override."""
        self._parseInputData(x, y)
        self.updateGraph()

    def data(self):
        """Override."""
        return self._x, self._y

    def transformedData(self):
        """Override."""
        if not self._check_finite:
            return super().transformedData()

        # inf/nans completely prevent the plot from being displayed starting on
        # Qt version 5.12.3
        # we do not expect to have nan in x
        return (self.toLogScale(self._x) if self._log_x_mode else self._x,
                self.toLogScale(self._y)
                if self._log_y_mode else np.nan_to_num(self._y))

    def _prepareGraph(self):
        """Override."""
        x, y = self.transformedData()
        self._graph = self.array2Path(x, y)

    @staticmethod
    def array2Path(x, y):
        """Convert array to QPainterPath."""
        path = QPainterPath()
        if len(x) >= 2:
            # see: https://github.com/qt/qtbase/blob/dev/src/gui/painting/qpainterpath.cpp
            n = len(x)
            buf = np.empty(n+2, dtype=[('c', '>i4'), ('x', '>f8'), ('y', '>f8')])
            byteview = buf.view(dtype=np.ubyte)
            # header (size)
            byteview[:16] = 0
            byteview.data[16:20] = struct.pack('>i', n)
            # data
            data = buf[1:-1]
            data['c'], data['x'], data['y'] = 1, x, y
            data['c'][0] = 0
            # tail (cStart, fillRule)
            byteview.data[-20:-16] = struct.pack('>i', 0)
            byteview.data[-16:-12] = struct.pack('>i', 0)

            # take the pointer without copy
            arr = QByteArray.fromRawData(byteview.data[16:-12])
            QDataStream(arr) >> path
        return path

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
        self._y_min = None
        self._y_max = None

        self._beam = 0.0 if beam is None else beam
        self._line = line
        self._pen = FColor.mkPen('p') if pen is None else pen

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


class ScatterPlotItem(pg.PlotItem):
    """ScatterPlotItem.

    Implemented based on pyqtgraph.ScatterPlotItem.
    """
    @staticmethod
    def createSymbols():
        _names = ['o', 's', 't', 't1', 't2', 't3',
                  'd', '+', 'x', 'p', 'h', 'star',
                  'arrow_up', 'arrow_right', 'arrow_down', 'arrow_left']
        symbols = OrderedDict([(name, QPainterPath()) for name in _names])
        symbols['o'].addEllipse(QRectF(-0.5, -0.5, 1., 1.))
        symbols['s'].addRect(QRectF(-0.5, -0.5, 1., 1.))
        _coords = {
            't': [(-0.5, -0.5), (0, 0.5), (0.5, -0.5)],
            't1': [(-0.5, 0.5), (0, -0.5), (0.5, 0.5)],
            't2': [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0)],
            't3': [(0.5, 0.5), (0.5, -0.5), (-0.5, 0)],
            'd': [(0., -0.5), (-0.4, 0.), (0, 0.5), (0.4, 0)],
            '+': [(-0.5, -0.05), (-0.5, 0.05), (-0.05, 0.05), (-0.05, 0.5),
                  (0.05, 0.5), (0.05, 0.05), (0.5, 0.05), (0.5, -0.05),
                  (0.05, -0.05), (0.05, -0.5), (-0.05, -0.5), (-0.05, -0.05)],
            'p': [(0, -0.5), (-0.4755, -0.1545), (-0.2939, 0.4045),
                  (0.2939, 0.4045), (0.4755, -0.1545)],
            'h': [(0.433, 0.25), (0., 0.5), (-0.433, 0.25), (-0.433, -0.25),
                  (0, -0.5), (0.433, -0.25)],
            'star': [(0, -0.5), (-0.1123, -0.1545), (-0.4755, -0.1545),
                     (-0.1816, 0.059), (-0.2939, 0.4045), (0, 0.1910),
                     (0.2939, 0.4045), (0.1816, 0.059), (0.4755, -0.1545),
                     (0.1123, -0.1545)],
            'arrow_down': [
                (-0.125, 0.125), (0, 0), (0.125, 0.125),
                (0.05, 0.125), (0.05, 0.5), (-0.05, 0.5), (-0.05, 0.125)
            ]
        }

        for k, c in _coords.items():
            symbols[k].moveTo(*c[0])
            for x, y in c[1:]:
                symbols[k].lineTo(x, y)
            symbols[k].closeSubpath()

        del _coords

        tr = QTransform()
        tr.rotate(45)
        symbols['x'] = tr.map(symbols['+'])
        tr.rotate(45)
        symbols['arrow_right'] = tr.map(symbols['arrow_down'])
        symbols['arrow_up'] = tr.map(symbols['arrow_right'])
        symbols['arrow_left'] = tr.map(symbols['arrow_up'])
        return symbols

    _symbol_map = createSymbols.__func__()

    def __init__(self, x=None, y=None, *, symbol='o', size=8,
                 pen=None, brush=None, name=None, parent=None):
        """Initialization."""
        super().__init__(name=name, parent=parent)

        self._x = None
        self._y = None

        self._bounds = [None, None]

        if pen is None and brush is None:
            self._pen = FColor.mkPen(None)
            self._brush = FColor.mkBrush('b')
        else:
            self._pen = FColor.mkPen(None) if pen is None else pen
            self._brush = FColor.mkBrush(None) if brush is None else brush

        self._size = size

        self._symbol_path = self._symbol_map[symbol]
        self._symbol_fragment = None
        self._symbol_width = None
        self._buildFragment()

        self.setData(x, y)

    def updateGraph(self):
        """Override."""
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self._bounds = [None, None]

    def setData(self, x, y):
        """Override."""
        self._parseInputData(x, y)
        self.updateGraph()

    def data(self):
        """Override."""
        return self._x, self._y

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if frac >= 1.0 and orthoRange is None and self._bounds[ax] is not None:
            return self._bounds[ax]

        if len(self._y) == 0:
            return None, None

        x, y = self.transformedData()
        if ax == 0:
            d = x
            d2 = y
        elif ax == 1:
            d = y
            d2 = x

        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            d = d[mask]
            d2 = d2[mask]

            if d.size == 0:
                return None, None

        if frac >= 1.0:
            self._bounds[ax] = (np.nanmin(d), np.nanmax(d))
            return self._bounds[ax]
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            mask = np.isfinite(d)
            d = d[mask]
            return np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])

    def pixelPadding(self):
        return 0.7072 * self._symbol_width

    def boundingRect(self):
        """Override."""
        xmn, xmx = self.dataBounds(ax=0)
        ymn, ymx = self.dataBounds(ax=1)
        if xmn is None or xmx is None:
            xmn = 0
            xmx = 0
        if ymn is None or ymx is None:
            ymn = 0
            ymx = 0

        px = py = 0.0
        pad = self.pixelPadding()
        if pad > 0:
            # determine length of pixel in local x, y directions
            px, py = self.pixelVectors()
            try:
                px = 0 if px is None else px.length()
            except OverflowError:
                px = 0
            try:
                py = 0 if py is None else py.length()
            except OverflowError:
                py = 0

            # return bounds expanded by pixel size
            px *= pad
            py *= pad
        return QRectF(xmn - px, ymn - py, 2*px + xmx - xmn, 2*py + ymx - ymn)

    def viewTransformChanged(self):
        # FIXME: I am not sure whether this is needed
        self.prepareGeometryChange()
        self._bounds = [None, None]

    def mapPointsToDevice(self, pts):
        tr = self.deviceTransform()
        if tr is None:
            return None

        pts = fn.transformCoordinates(tr, pts)
        pts -= 0.5 * self._symbol_width
        # prevent sluggish GUI and possibly Qt segmentation fault.
        pts = np.clip(pts, -2**30, 2**30)

        return pts

    def getViewMask(self, pts):
        vb = self.getViewBox()
        if vb is None:
            return None

        rect = vb.mapRectToDevice(vb.boundingRect())
        w = 0.5 * self._symbol_width

        mask = ((pts[0] + w > rect.left()) &
                (pts[0] - w < rect.right()) &
                (pts[1] + w > rect.top()) &
                (pts[1] - w < rect.bottom()))

        return mask

    def paint(self, p, *args):
        """Override."""
        p.resetTransform()

        x, y = self.transformedData()

        pts = np.vstack([x, y])
        pts = self.mapPointsToDevice(pts)
        if pts is None:
            return

        masked_pts = pts[:, self.getViewMask(pts)]
        width = self._symbol_width
        source_rect = QRectF(self._symbol_fragment.rect())
        for px, py in zip(masked_pts[0, :], masked_pts[1, :]):
            p.drawPixmap(QRectF(px, py, width, width),
                         self._symbol_fragment,
                         source_rect)

    def drawSample(self, p):
        """Override."""
        p.translate(10, 10)
        self.drawSymbol(p)

    def drawSymbol(self, p):
        p.scale(self._size, self._size)
        p.setPen(self._pen)
        p.setBrush(self._brush)
        p.drawPath(self._symbol_path)

    def _buildFragment(self):
        pen = self._pen
        size = int(self._size + max(np.ceil(pen.widthF()), 1))
        image = QImage(size, size, QImage.Format_ARGB32)
        image.fill(0)
        p = QPainter(image)
        try:
            # default is QPainter.TextAntialiasing
            p.setRenderHint(QPainter.Antialiasing)
            center = 0.5 * size
            p.translate(center, center)
            self.drawSymbol(p)
        finally:
            p.end()

        self._symbol_fragment = QPixmap(image)
        self._symbol_width = self._symbol_fragment.width()
