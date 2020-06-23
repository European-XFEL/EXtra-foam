import abc

import numpy as np

from ..Qt import QtGui, QtCore, QT_LIB
if QT_LIB in ['PyQt4', 'PyQt5']:
    import sip
from .GraphicsItem import GraphicsItem

__all__ = ['GraphicsObject', 'PlotItem']


class GraphicsObject(GraphicsItem, QtGui.QGraphicsObject):
    """
    **Bases:** :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`, :class:`QtGui.QGraphicsObject`

    Extension of QGraphicsObject with some useful methods (provided by :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`)
    """
    _qtBaseClass = QtGui.QGraphicsObject

    def __init__(self, *args, **kwargs):
        self.__inform_view_on_changes = True
        QtGui.QGraphicsObject.__init__(self, *args, **kwargs)
        self.setFlag(self.ItemSendsGeometryChanges)
        GraphicsItem.__init__(self)
        
    def itemChange(self, change, value):
        ret = QtGui.QGraphicsObject.itemChange(self, change, value)
        if change in [self.ItemParentHasChanged, self.ItemSceneHasChanged]:
            self.parentChanged()
        try:
            inform_view_on_change = self.__inform_view_on_changes
        except AttributeError:
            # It's possible that the attribute was already collected when the itemChange happened
            # (if it was triggered during the gc of the object).
            pass
        else:
            if inform_view_on_change and change in [self.ItemPositionHasChanged, self.ItemTransformHasChanged]:
                self.informViewBoundsChanged()
            
        ## workaround for pyqt bug:
        ## http://www.riverbankcomputing.com/pipermail/pyqt/2012-August/031818.html
        if QT_LIB in ['PyQt4', 'PyQt5'] and change == self.ItemParentChange and isinstance(ret, QtGui.QGraphicsItem):
            ret = sip.cast(ret, QtGui.QGraphicsItem)

        return ret


class PlotItem(GraphicsObject):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._graph = None

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
