"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Base PlotWidget and related PlotItems.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .. import pyqtgraph as pg

from ..pyqtgraph import GraphicsView, PlotItem, QtCore, QtGui

from ..misc_widgets import make_brush, make_pen


class _BarPlotItem(pg.GraphicsObject):
    """_BarPlotItem"""
    def __init__(self, x=None, y=None, width=1.0, pen=None, brush=None):
        """Initialization."""
        super().__init__()

        self._picture = None

        self._x = None
        self._y = None

        self._width = None
        self.width = width

        self._pen = make_pen('e') if pen is None else pen
        self._brush = make_brush('e') if brush is None else brush

        self.setData(x, y)

    def setData(self, x, y, width=None, pen=None):
        """PlotItem interface."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")

        self.width = width

        if pen is not None:
            self._pen = pen

        self._picture = None
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value is None:
            return

        if value > 1.0 or value <= 0:
            value = 1.0  # set to default
        self._width = value

    def drawPicture(self):
        self._picture = QtGui.QPicture()
        p = QtGui.QPainter(self._picture)

        p.setPen(self._pen)
        p.setBrush(self._brush)

        # Now it works for bar plot with equalized gaps
        # TODO: extend it
        if len(self._x) > 1:
            width = self._width * (self._x[1] - self._x[0])
        else:
            width = self._width

        for x, y in zip(self._x, self._y):
            rect = QtCore.QRectF(x - width/2, 0, width, y)
            p.drawRect(rect)

        p.end()
        self.prepareGeometryChange()

    def paint(self, p, *args):
        if self._picture is None:
            self.drawPicture()
        self._picture.play(p)

    def boundingRect(self):
        if self._picture is None:
            self.drawPicture()
        return QtCore.QRectF(self._picture.boundingRect())


class _ErrorBarItem(pg.GraphicsObject):
    """ErrorBarItem."""

    def __init__(self, x=None, y=None, y_min=None, y_max=None, beam=None, pen=None):
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
        self._pen = make_pen('e') if pen is None else pen

        self.setData(x, y, y_min=y_min, y_max=y_max)

    def setData(self, x, y, y_min=None, y_max=None, beam=None, pen=None):
        """PlotItem interface."""
        self._x = [] if x is None else x
        self._y = [] if y is None else y

        self._y_min = self._y if y_min is None else y_min
        self._y_max = self._y if y_max is None else y_max

        if len(self._x) != len(self._y):
            raise ValueError("'x' and 'y' data have different lengths!")
        if not len(self._y) == len(self._y_min) == len(self._y_max):
            raise ValueError(
                "'y_min' and 'y_max' data have different lengths!")

        if beam is not None and beam >= 0.0:
            self._beam = beam
        if pen is not None:
            self._pen = pen

        self._path = None
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    def drawPath(self):
        p = QtGui.QPainterPath()

        x = self._x

        for i in range(len(x)):
            # plot the lower horizontal lines
            p.moveTo(x[i] - self._beam / 2., self._y_min[i])
            p.lineTo(x[i] + self._beam / 2., self._y_min[i])

            # plot the vertical line
            p.moveTo(x[i], self._y_min[i])
            p.lineTo(x[i], self._y_max[i])

            # plot the upper horizontal line
            p.moveTo(x[i] - self._beam / 2., self._y_max[i])
            p.lineTo(x[i] + self._beam / 2., self._y_max[i])

        self._path = p
        self.prepareGeometryChange()

    def paint(self, p, *args):
        if self._path is None:
            self.drawPath()

        p.setPen(self._pen)
        p.drawPath(self._path)

    def boundingRect(self):
        if self._path is None:
            self.drawPath()
        return self._path.boundingRect()


class PlotWidget(GraphicsView):
    """GraphicsView widget displaying a single PlotItem.

    Note: it is different from the PlotWidget in pyqtgraph.

    This base class should be used to display plots except image in
    karaboFAI. For image, please refer to ImageView class.
    """
    # signals wrapped from PlotItem / ViewBox
    sigRangeChanged = QtCore.Signal(object, object)
    sigTransformChanged = QtCore.Signal(object)

    _pen = make_pen(None)
    _brush_size = 10

    def __init__(self, parent=None, background='default', **kargs):
        """Initialization."""
        super().__init__(parent, background=background)
        if parent is not None:
            parent.registerPlotWidget(self)

        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)
        self.enableMouse(False)
        self.plotItem = PlotItem(**kargs)
        self.setCentralItem(self.plotItem)

        self.plotItem.sigRangeChanged.connect(self.viewRangeChanged)

    def clear(self):
        """Remove all the items in the PlotItem object."""
        plot_item = self.plotItem
        for i in plot_item.items[:]:
            plot_item.removeItem(i)

    def reset(self):
        """Clear the data of all the items in the PlotItem object."""
        for item in self.plotItem.items:
            item.setData([], [])

    def update(self, data):
        raise NotImplemented

    def close(self):
        self.plotItem.close()
        self.plotItem = None
        self.setParent(None)
        super().close()

    def addItem(self, *args, **kwargs):
        """Explicitly call PlotItem.addItem.

        This method must be here to override the addItem method in
        GraphicsView. Otherwise, people may misuse the addItem method.
        """
        self.plotItem.addItem(*args, **kwargs)

    def removeItem(self, *args, **kwargs):
        self.plotItem.removeItem(*args, **kwargs)

    def plotCurve(self, *args, **kwargs):
        """Add and return a new curve plot."""
        item = pg.PlotCurveItem(*args, **kwargs)
        self.plotItem.addItem(item)
        return item

    def plotScatter(self, *args, pen=None, size=None, **kwargs):
        """Add and return a new scatter plot."""
        if pen is None:
            pen = self._pen
        if size is None:
            size = self._brush_size

        item = pg.ScatterPlotItem(*args, pen=pen, size=size, **kwargs)
        self.plotItem.addItem(item)
        return item

    def plotBar(self, x=None, y=None, width=1.0, **kwargs):
        """Add and return a new bar plot."""
        item = _BarPlotItem(x=x, y=y, width=width, **kwargs)
        self.plotItem.addItem(item)
        return item

    def plotErrorBar(self, x=None, y=None, y_min=None, y_max=None, beam=None):
        item = _ErrorBarItem(x=x, y=y, y_min=y_min, y_max=y_max, beam=beam)
        self.plotItem.addItem(item)
        return item

    def plotImage(self, *args, **kargs):
        """Add and return a image item."""
        # TODO: this will be done when another branch is merged
        raise NotImplemented

    def setAspectLocked(self, *args, **kwargs):
        self.plotItem.setAspectLocked(*args, **kwargs)

    def setLabel(self, *args, **kwargs):
        self.plotItem.setLabel(*args, **kwargs)

    def setTitle(self, *args, **kwargs):
        self.plotItem.setTitle(*args, **kwargs)

    def addLegend(self, *args, **kwargs):
        self.plotItem.addLegend(*args, **kwargs)

    def hideAxis(self):
        for v in ["left", 'bottom']:
            self.plotItem.hideAxis(v)

    def showAxis(self):
        for v in ["left", 'bottom']:
            self.plotItem.showAxis(v)

    def viewRangeChanged(self, view, range):
        self.sigRangeChanged.emit(self, range)

    def saveState(self):
        return self.plotItem.saveState()

    def restoreState(self, state):
        return self.plotItem.restoreState(state)

    def closeEvent(self, QCloseEvent):
        parent = self.parent()
        if parent is not None:
            parent.unregisterPlotWidget(self)
        super().closeEvent(QCloseEvent)
