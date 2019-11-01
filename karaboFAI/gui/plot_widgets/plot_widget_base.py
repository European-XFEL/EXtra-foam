"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtCore, QtGui

from .. import pyqtgraph as pg
from ..pyqtgraph import GraphicsView, PlotItem

from .plot_items import BarPlotItem, ErrorBarItem
from ..misc_widgets import make_pen


class PlotWidgetF(GraphicsView):
    """PlotWidget base class.

    GraphicsView widget displaying a single PlotItem.

    Note: it is different from the PlotWidget in pyqtgraph.

    This base class should be used to display plots except image in
    karaboFAI. For image, please refer to ImageViewF class.
    """
    # signals wrapped from PlotItem / ViewBox
    sigRangeChanged = QtCore.Signal(object, object)
    sigTransformChanged = QtCore.Signal(object)

    _pen = make_pen(None)
    _brush_size = 8

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
        item = BarPlotItem(x=x, y=y, width=width, **kwargs)
        self.plotItem.addItem(item)
        return item

    def plotErrorBar(self,
                     x=None,
                     y=None,
                     y_min=None,
                     y_max=None,
                     beam=None,
                     **kwargs):
        item = ErrorBarItem(x=x,
                            y=y,
                            y_min=y_min,
                            y_max=y_max,
                            beam=beam,
                            **kwargs)
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
