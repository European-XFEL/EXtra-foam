import unittest
from unittest.mock import MagicMock

import numpy as np

from PyQt5.QtCore import QPointF, QRectF

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.plot_widget_base import PlotWidgetF
from extra_foam.gui.plot_widgets.plot_items import (
    CurvePlotItem, BarGraphItem, StatisticsBarItem
)
from extra_foam.logger import logger

from . import _display

app = mkQApp()

logger.setLevel("CRITICAL")


class TestPlotItems(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._widget = PlotWidgetF()
        if _display():
            cls._widget.show()

    def tearDown(self):
        self._widget.removeAllItems()

    def testCurvePlotItem(self):
        item = CurvePlotItem()
        self._widget.addItem(item)

        with self.assertRaisesRegex(ValueError, "different lengths"):
            item.setData(np.arange(2), np.arange(3))

        item.setData(np.arange(10), np.arange(10))
        _display()

        # test log mode
        item.setLogX(True)
        _display()
        self.assertEqual(QRectF(0, 0, 1.0, 9.0), item.boundingRect())
        item.setLogY(True)
        _display()
        self.assertEqual(QRectF(0, 0, 1.0, 1.0), item.boundingRect())

        # clear data
        item.setData([], [])
        _display()

    def testBarGraphItem(self):
        item = BarGraphItem()
        self._widget.addItem(item)

        with self.assertRaisesRegex(ValueError, "different lengths"):
            item.setData(np.arange(2), np.arange(3))

        item.setData(np.arange(10), np.arange(10))
        _display()

        # test log mode
        item.setLogX(True)
        _display()
        self.assertEqual(QRectF(-1.0, 0, 3.0, 9.0), item.boundingRect())
        item.setLogY(True)
        _display()
        self.assertEqual(QRectF(-1.0, 0, 3.0, 1.0), item.boundingRect())

        # clear data
        item.setData([], [])
        _display()

    def testStatisticsBarItem(self):
        item = StatisticsBarItem()
        self._widget.addItem(item)

        with self.assertRaisesRegex(ValueError, "different lengths"):
            item.setData(np.arange(2), np.arange(3))

        with self.assertRaisesRegex(ValueError, "different lengths"):
            item.setData(np.arange(2), np.arange(2), y_min=np.arange(3), y_max=np.arange(2))

        with self.assertRaisesRegex(ValueError, "different lengths"):
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
        self.assertEqual(QRectF(-0.5, -1.0, 2.0, 11.0), item.boundingRect())
        item.setLogY(True)
        _display()
        self.assertEqual(QPointF(-0.5, 0.0), item.boundingRect().topLeft())
        self.assertEqual(1.5, item.boundingRect().bottomRight().x())
        self.assertGreater(1.1, item.boundingRect().bottomRight().y())
        self.assertLess(1.0, item.boundingRect().bottomRight().y())

        # clear data
        item.setData([], [])
        _display()
