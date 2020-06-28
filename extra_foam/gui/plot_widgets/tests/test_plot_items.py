import pytest
from unittest.mock import MagicMock

import numpy as np

from PyQt5.QtCore import QByteArray, QDataStream, QIODevice, QPointF, QRectF
from PyQt5.QtGui import QPainterPath

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.plot_widget_base import PlotWidgetF
from extra_foam.gui.plot_widgets.plot_items import (
    CurvePlotItem, BarGraphItem, StatisticsBarItem
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

    def tearDown(self):
        self._widget.removeAllItems()

    def testCurvePlotItemArray2Path(self):
        item = CurvePlotItem()
        self._widget.addItem(item)

        # create a path from arrays
        size = 5
        x = np.arange(size)
        y = 2 * np.arange(size)
        item.setData(x, y)
        p = item._graph

        # stream path
        arr = QByteArray()
        buf = QDataStream(arr, QIODevice.ReadWrite)
        buf << p
        buf.device().reset()

        # test protocol
        assert arr.size() == 4 + size * 20 + 8
        assert buf.readInt32() == size
        for i in range(5):
            if i == 0:
                assert buf.readInt32() == 0
            else:
                assert buf.readInt32() == 1
            assert buf.readDouble() == x[i]
            assert buf.readDouble() == y[i]
        assert buf.readInt32() == 0
        assert buf.readInt32() == 0

    @pytest.mark.parametrize("dtype", [np.float, np.int64, np.uint16])
    def testCurvePlotItem(self, dtype):
        item = CurvePlotItem()
        self._widget.addItem(item)

        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2).astype(dtype), np.arange(3).astype(dtype))

        # x and y are lists
        item.setData(np.arange(10).tolist(), np.arange(10).astype(dtype).tolist())

        # x and y are numpy.arrays
        item.setData(np.arange(10).astype(dtype), np.arange(10).astype(dtype))
        if dtype == np.float:
            _display()

        # test log mode
        item.setLogX(True)
        if dtype == np.float:
            _display()
        assert item.boundingRect() == QRectF(0, 0, 1.0, 9.0)
        item.setLogY(True)
        if dtype == np.float:
            _display()
        assert item.boundingRect() == QRectF(0, 0, 1.0, 1.0)

        # clear data
        item.setData([], [])
        if dtype == np.float:
            _display()

    def testBarGraphItem(self):
        item = BarGraphItem()
        self._widget.addItem(item)

        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2), np.arange(3))

        item.setData(np.arange(10), np.arange(10))
        _display()

        # test log mode
        item.setLogX(True)
        _display()
        assert item.boundingRect() == QRectF(-1.0, 0, 3.0, 9.0)
        item.setLogY(True)
        _display()
        assert item.boundingRect() == QRectF(-1.0, 0, 3.0, 1.0)

        # clear data
        item.setData([], [])
        _display()

    def testStatisticsBarItem(self):
        item = StatisticsBarItem()
        self._widget.addItem(item)

        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2), np.arange(3))

        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2), np.arange(2), y_min=np.arange(3), y_max=np.arange(2))

        with pytest.raises(ValueError, match="different lengths"):
            item.setData(np.arange(2), np.arange(2), y_min=np.arange(2), y_max=np.arange(3))

        y = np.arange(10)
        y_min = y - 1
        y_max = y + 1
        item.setBeam(1)
        item.setData(np.arange(10), y, y_min=y_min, y_max=y_max)
        _display()

        # test log mode
        item.setLogX(True)
        _display()
        assert item.boundingRect() == QRectF(-0.5, -1.0, 2.0, 11.0)
        item.setLogY(True)
        _display()
        assert item.boundingRect().topLeft() == QPointF(-0.5, 0.0)
        assert 1.5, item.boundingRect().bottomRight().x()
        assert 1.0 < item.boundingRect().bottomRight().y() < 1.1

        # clear data
        item.setData([], [])
        _display()
