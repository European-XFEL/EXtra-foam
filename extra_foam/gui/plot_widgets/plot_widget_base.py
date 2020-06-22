"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc

import numpy as np

from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import QSizePolicy

from .. import pyqtgraph as pg

from .graphics_widgets import PlotArea
from .plot_items import (
    BarGraphItem, CurvePlotItem, StatisticsBarItem
)
from ..misc_widgets import FColor
from ...config import config
from ...typing import final


class PlotWidgetF(pg.GraphicsView):
    """PlotWidget base class.

    GraphicsView widget displaying a single PlotItem.

    Note: it is different from the PlotWidget in pyqtgraph.

    This base class should be used to display plots except images.
    For displaying images, please refer to ImageViewF class.
    """
    # signals wrapped from PlotArea / ViewBox
    sigRangeChanged = pyqtSignal(object, object)
    sigTransformChanged = pyqtSignal(object)

    def __init__(self, parent=None, *,
                 background='default', show_indicator=False, **kargs):
        """Initialization."""
        super().__init__(parent, background=background)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.enableMouse(False)

        self._title = ""

        self._plot_area = PlotArea(**kargs)
        self.setCentralWidget(self._plot_area)

        # TODO: improve indicator implementation
        # Move Indicator to PlotArea
        self._show_indicator = show_indicator
        if show_indicator:
            self._v_line = pg.InfiniteLine(angle=90, movable=False)
            self._h_line = pg.InfiniteLine(angle=0, movable=False)
            self._indicator = pg.TextItem(color=FColor.k)
            self._v_line.hide()
            self._h_line.hide()
            self._indicator.hide()
            self._plot_area.addItem(self._v_line, ignore_bounds=True)
            self._plot_area.addItem(self._h_line, ignore_bounds=True)
            self._plot_area.scene().addItem(self._indicator)

            # rateLimit should be fast enough to be able to capture
            # the leaveEvent
            self._proxy = pg.SignalProxy(self._plot_area.scene().sigMouseMoved,
                                         rateLimit=60, slot=self.onMouseMoved)

        self._vb2 = None  # ViewBox for y2 axis

        self._plot_area.range_changed_sgn.connect(self.viewRangeChanged)

        if parent is not None and hasattr(parent, 'registerPlotWidget'):
            parent.registerPlotWidget(self)

    def reset(self):
        """Clear the data of all the items in the PlotArea object."""
        self._plot_area.clearAllPlotItems()

        # TODO: improve vb2 implementation
        if self._vb2 is not None:
            for item in self._vb2.addedItems:
                try:
                    item.setData([], [])
                except TypeError:
                    pass

    @abc.abstractmethod
    def updateF(self, data):
        """This method is called by the parent window."""
        raise NotImplementedError

    def close(self):
        self._plot_area.close()
        self._plot_area = None
        self.setParent(None)
        super().close()

    def addItem(self, *args, **kwargs):
        """Explicitly call PlotArea.addItem.

        This method must be here to override the addItem method in
        GraphicsView. Otherwise, people may misuse the addItem method.
        """
        self._plot_area.addItem(*args, **kwargs)

    def removeItem(self, *args, **kwargs):
        self._plot_area.removeItem(*args, **kwargs)

    def plotCurve(self, *args, y2=False, **kwargs):
        """Add and return a new curve plot."""
        item = CurvePlotItem(*args, **kwargs)

        if y2:
            if self._vb2 is None:
                self.createY2()
            self._vb2.addItem(item)
        else:
            self._plot_area.addItem(item)

        return item

    def plotScatter(self, *args, **kwargs):
        """Add and return a new scatter plot."""
        if 'pen' not in kwargs:
            kwargs['pen'] = FColor.mkPen(None)
        item = pg.ScatterPlotItem(*args, **kwargs)
        self._plot_area.addItem(item)
        return item

    def plotBar(self, x=None, y=None, width=1.0, y2=False, **kwargs):
        """Add and return a new bar plot."""
        item = BarGraphItem(x=x, y=y, width=width, **kwargs)

        if y2:
            if self._vb2 is None:
                self.createY2()
            self._vb2.addItem(item)
        else:
            self._plot_area.addItem(item)

        return item

    def plotStatisticsBar(self, x=None, y=None, y_min=None, y_max=None,
                          beam=None, y2=False, **kwargs):
        item = StatisticsBarItem(x=x, y=y, y_min=y_min, y_max=y_max,
                                 beam=beam, **kwargs)
        if y2:
            if self._vb2 is None:
                self.createY2()
            self._vb2.addItem(item)
        else:
            self._plot_area.addItem(item)

        return item

    def plotImage(self, *args, **kargs):
        """Add and return a image item."""
        # TODO: this will be done when another branch is merged
        raise NotImplemented

    def createY2(self):
        vb = pg.ViewBox()
        plot_item = self._plot_area
        plot_item.scene().addItem(vb)
        plot_item.getAxis('right').linkToView(vb)
        # TODO: improve
        vb.setXLink(self._plot_area._vb)
        self._plot_area._vb.sigResized.connect(self.updateY2View)
        self._vb2 = vb

    def updateY2View(self):
        vb = self._vb2
        if vb is None:
            return
        # update ViewBox-y2 to match ViewBox-y
        # TODO: improve
        vb.setGeometry(self._plot_area._vb.sceneBoundingRect())
        # not sure this is required
        # vb.linkedViewChanged(self._plot_area.vb, vb.XAxis)

    def removeAllItems(self):
        """Remove all the items in the PlotArea object."""
        self._plot_area.removeAllItems()

    def setAspectLocked(self, *args, **kwargs):
        self._plot_area.setAspectLocked(*args, **kwargs)

    def setLabel(self, *args, **kwargs):
        self._plot_area.setLabel(*args, **kwargs)

    def setTitle(self, *args, **kwargs):
        self._plot_area.setTitle(*args, **kwargs)

    def addLegend(self, *args, **kwargs):
        self._plot_area.addLegend(*args, **kwargs)

    def hideAxis(self):
        """Hide x and y axis."""
        for v in ["left", 'bottom']:
            self._plot_area.showAxis(v, False)

    def showAxis(self):
        """Show x and y axis."""
        for v in ["left", 'bottom']:
            self._plot_area.showAxis(v, True)

    def hideLegend(self):
        """Hide legend."""
        self._plot_area.showLegend(False)

    def showLegend(self):
        """Show legend."""
        self._plot_area.showLegend(True)

    def disableLogMenu(self):
        """Disable log X/Y context menu."""
        self._plot_area.enableLogMenu(False)

    def viewRangeChanged(self, view, range):
        self.sigRangeChanged.emit(self, range)

    def onMouseMoved(self, ev):
        pos = ev[0]
        y_shift = 45  # move text to the top of the mouse cursor
        self._indicator.setPos(pos.x(), pos.y() - y_shift)

        # TODO: improve
        m_pos = self._plot_area._vb.mapSceneToView(pos)
        x, y = m_pos.x(), m_pos.y()
        self._v_line.setPos(x)
        self._h_line.setPos(y)

        self._indicator.setText(f"x={x}\ny={y}")

    def enterEvent(self, ev):
        """Override."""
        if self._show_indicator:
            self._v_line.show()
            self._h_line.show()
            self._indicator.show()

    def leaveEvent(self, ev):
        """Override."""
        if self._show_indicator:
            self._v_line.hide()
            self._h_line.hide()
            self._indicator.hide()

    def closeEvent(self, event):
        """Override."""
        parent = self.parent()
        if parent is not None:
            parent.unregisterPlotWidget(self)
        super().closeEvent(event)


class TimedPlotWidgetF(PlotWidgetF):
    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._data = None

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_imp)
        self._timer.start(config["GUI_PLOT_WITH_STATE_UPDATE_TIMER"])

    @abc.abstractmethod
    def refresh(self):
        pass

    def _refresh_imp(self):
        if self._data is not None:
            self.refresh()

    @final
    def updateF(self, data):
        """Override."""
        self._data = data


class HistMixin:
    def updateTitle(self, mean=np.nan, median=np.nan, std=np.nan):
        self.setTitle(self._title_template.substitute(
            mean=f"{mean:.2e}", median=f"{median:.2e}", std=f"{std:.2e}"))

    def reset(self):
        super().reset()
        self.updateTitle()
