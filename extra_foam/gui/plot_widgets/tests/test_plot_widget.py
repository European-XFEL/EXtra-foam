import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from PyQt5.QtTest import QTest
from PyQt5.QtCore import QPoint

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.plot_widget_base import PlotWidgetF, TimedPlotWidgetF
from extra_foam.logger import logger


app = mkQApp()

logger.setLevel("CRITICAL")


class TestPlotWidget(unittest.TestCase):
    def setUp(self):
        self._widget = PlotWidgetF()

    def testAddPlots(self):
        widget = self._widget

        self.assertEqual(len(self._widget._plot_area._items), 2)

        # test Legend
        widget.addLegend()
        widget.plotCurve(name="curve")
        widget.plotScatter(name="scatter")
        widget.plotBar(name="bar")
        widget.plotStatisticsBar(name="statistics")

        self.assertEqual(len(self._widget._plot_area._items), 6)

    def testForwardMethod(self):
        widget = self._widget

        for method in ["removeAllItems", "setAspectLocked", "setLabel", "setTitle",
                       "setAnnotationList", "addLegend", "invertX", "invertY", "autoRange"]:
            with patch.object(widget._plot_area, method) as mocked:
                getattr(widget, method)()
                mocked.assert_called_once()

    def testShowHideAxisLegend(self):
        widget = self._widget

        widget.showAxis()
        self.assertTrue(widget._plot_area.getAxis("left").isVisible())
        self.assertTrue(widget._plot_area.getAxis("left").isVisible())
        widget.hideAxis()
        self.assertFalse(widget._plot_area.getAxis("left").isVisible())
        self.assertFalse(widget._plot_area.getAxis("left").isVisible())

        widget.addLegend()
        self.assertTrue(widget._plot_area._legend.isVisible())
        widget.hideLegend()
        self.assertFalse(widget._plot_area._legend.isVisible())
        widget.showLegend()
        self.assertTrue(widget._plot_area._legend.isVisible())

    def testCurvePlot(self):
        plot = self._widget.plotCurve(np.arange(3), np.arange(1, 4, 1))
        app.processEvents()

        # test set empty data
        plot.setData([], [])
        app.processEvents()

        plot.setData([1], [1])
        app.processEvents()

        # test if x and y have different lengths
        with self.assertRaises(Exception):
            plot.setData([1, 2, 3], [])

    def testBarPlot(self):
        # set any valid number
        plot = self._widget.plotBar([1, 2], [3, 4])
        app.processEvents()

        # test set empty data
        plot.setData([], [])
        app.processEvents()

        # test if x and y have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [])

    def testErrorBarPlot(self):
        # set any valid number
        plot = self._widget.plotStatisticsBar([1, 2], [3, 4])
        app.processEvents()

        # set x, y, y_min and y_max together
        plot.setData([1, 2], [1, 2], y_min=[0, 0], y_max=[2, 2])
        app.processEvents()

        # test set empty data
        plot.setData([], [])
        app.processEvents()

        # test if x and y have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [], y_min=[0, 0, 0], y_max=[2, 2, 2])

        # test if y_min/ymax have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [1, 2, 3], y_min=[0, 0, 0], y_max=[2, 2])

    def testCrossCursor(self):
        widget = self._widget
        self.assertFalse(widget._v_line.isVisible())
        self.assertFalse(widget._h_line.isVisible())
        widget._plot_area._show_cross_cb.setChecked(True)
        self.assertTrue(widget._v_line.isVisible())
        self.assertTrue(widget._h_line.isVisible())

        # TODO: test mouse move


class TestTimedPlotWidgetF(unittest.TestCase):
    def testUpdate(self):
        widget = TimedPlotWidgetF()
        widget.refresh = MagicMock()

        self.assertIsNone(widget._data)
        widget._refresh_imp()
        widget.refresh.assert_not_called()

        widget.updateF(1)
        widget._refresh_imp()
        widget.refresh.assert_called_once()
