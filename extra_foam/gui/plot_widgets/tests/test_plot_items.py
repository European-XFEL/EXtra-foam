import pytest

import numpy as np

from PyQt5.QtCore import QByteArray, QDataStream, QIODevice, QPointF, QRectF

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.plot_widget_base import PlotWidgetF
from extra_foam.gui.plot_widgets.plot_items import (
    CurvePlotItem, BarGraphItem, ScatterPlotItem, StatisticsBarItem
)
from extra_foam.logger import logger

from . import _display

app = mkQApp()

logger.setLevel("CRITICAL")


class TestPlotItems:
    @classmethod
    def setup_class(cls):
        cls._widget = PlotWidgetF()
        if _display():
            cls._widget.show()

    def teardown_method(self):
        self._widget.removeAllItems()

    def testBarGraphItem(self, dtype=np.float32):
        x = np.arange(10).astype(dtype)
        y = x * 1.5

        # x and y are lists
        item = BarGraphItem(x.tolist(), y.tolist(), name='bar')
        self._widget.addItem(item)
        self._widget.addLegend()
        assert isinstance(item._x, np.ndarray)
        assert isinstance(item._y, np.ndarray)

        # x and y are numpy.arrays
        item.setData(x, y)
        _display()

        # test different lengths
        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2), np.arange(3))

        # test log mode
        self._widget._plot_area._onLogXChanged(True)
        _display()
        assert item.boundingRect() == QRectF(-1.0, 0, 3.0, 14.0)
        self._widget._plot_area._onLogYChanged(True)
        _display()
        assert item.boundingRect() == QRectF(-1.0, 0, 3.0, 2.0)

        # clear data
        item.setData([], [])
        assert isinstance(item._x, np.ndarray)
        assert isinstance(item._y, np.ndarray)
        _display()

    def testStatisticsBarItem(self, dtype=np.float32):
        x = np.arange(10).astype(dtype)
        y = np.arange(10).astype(dtype)

        # x and y are lists
        item = StatisticsBarItem(x.tolist(), y.tolist(), name='statistics bar')
        self._widget.addItem(item)
        self._widget.addLegend()
        assert isinstance(item._x, np.ndarray)
        assert isinstance(item._y, np.ndarray)
        assert isinstance(item._y_min, np.ndarray)
        assert isinstance(item._y_max, np.ndarray)

        # x and y are numpy.arrays
        y_min = y - 1
        y_max = y + 1
        item.setBeam(1)
        item.setData(x, y, y_min=y_min, y_max=y_max)
        _display()

        # test different lengths
        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2), np.arange(3))

        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2), np.arange(2), y_min=np.arange(3), y_max=np.arange(2))

        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2), np.arange(2), y_min=np.arange(2), y_max=np.arange(3))

        # test log mode
        self._widget._plot_area._onLogXChanged(True)
        _display()
        assert item.boundingRect() == QRectF(-0.5, -1.0, 2.0, 11.0)
        self._widget._plot_area._onLogYChanged(True)
        _display()
        assert item.boundingRect().topLeft() == QPointF(-0.5, 0.0)
        assert 1.5, item.boundingRect().bottomRight().x()
        assert 1.0 < item.boundingRect().bottomRight().y() < 1.1

        # clear data
        item.setData([], [])
        assert isinstance(item._x, np.ndarray)
        assert isinstance(item._y, np.ndarray)
        _display()
