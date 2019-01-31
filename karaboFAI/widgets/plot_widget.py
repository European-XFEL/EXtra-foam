"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PlotWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .pyqtgraph import GraphicsView, PlotItem, QtCore, QtGui


class PlotWidget(GraphicsView):
    """GraphicsView widget displaying a single PlotItem.

    This is a reimplementation of the PlotWidget in pyqtgraph.

    :class:`GraphicsView <pyqtgraph.GraphicsView>` widget with a single
    :class:`PlotItem <pyqtgraph.PlotItem>` inside.
    """
    # signals wrapped from PlotItem / ViewBox
    sigRangeChanged = QtCore.Signal(object, object)
    sigTransformChanged = QtCore.Signal(object)

    # The following methods are wrapped directly from PlotItem:
    __plotitem_methods = [
        'plot', 'removeItem', 'addLegend',
        'setXRange', 'setYRange', 'setRange', 'autoRange',
        'enableAutoRange', 'disableAutoRange',
        'setXLink', 'setYLink', 'setLabel', 'setTitle', 'setLimits',
        'viewRect', 'setMouseEnabled', 'register', 'unregister'
    ]

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
        pass

    def update(self, data):
        raise NotImplemented

    def close(self):
        self.plotItem.close()
        self.plotItem = None
        self.setParent(None)
        super().close()

    def __getattr__(self, attr):
        """Forward methods from plotItem."""
        if attr in self.__plotitem_methods:
            return getattr(self.plotItem, attr)
        raise AttributeError(attr)

    def addItem(self, *args, **kwargs):
        """Explicitly call the addItem in PlotItem.

        GraphicsView also has the addItem method.
        """
        self.plotItem.addItem(*args, **kwargs)

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
