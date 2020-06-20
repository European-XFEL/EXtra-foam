import unittest
from unittest.mock import MagicMock

import numpy as np

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

        item.setData([], [])
        _display()

    def testBarGraphItem(self):
        item = BarGraphItem()
        self._widget.addItem(item)

        with self.assertRaisesRegex(ValueError, "different lengths"):
            item.setData(np.arange(2), np.arange(3))

        item.setData(np.arange(10), np.arange(10))
        _display()

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

        item.setData([], [])
        _display()
