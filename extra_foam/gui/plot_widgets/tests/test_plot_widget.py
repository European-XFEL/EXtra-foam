import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.plot_widget_base import PlotWidgetF, TimedPlotWidgetF
from extra_foam.gui.plot_widgets.plot_items import StatisticsBarItem
from extra_foam.logger import logger

from . import _display


app = mkQApp()

logger.setLevel("CRITICAL")


class TestPlotWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # test addLegend before adding plot items
        widget = PlotWidgetF()
        widget.addLegend()
        cls._curve1 = widget.plotCurve(name="curve1")
        cls._scatter1 = widget.plotScatter(name="scatter1")
        cls._bar2 = widget.plotBar(name="bar2", y2=True)
        cls._statistics2 = widget.plotStatisticsBar(name="statistics2", y2=True)
        cls._widget1 = widget
        if _display():
            cls._widget1.show()

        # test addLegend after adding plot items
        widget = PlotWidgetF()
        cls._bar1 = widget.plotBar(name="bar1")
        cls._statistics1 = widget.plotStatisticsBar(name="statistics1")
        cls._curve2 = widget.plotCurve(name="curve2", y2=True)
        cls._scatter2 = widget.plotScatter(name="scatter2", y2=True)
        widget.addLegend()
        cls._widget2 = widget
        if _display():
            cls._widget2.show()

        cls._plot_items1 = [cls._curve1, cls._scatter1, cls._bar2, cls._statistics2]
        cls._plot_items2 = [cls._bar1, cls._statistics1, cls._curve2, cls._scatter2]

    def testAddPlots(self):
        self.assertEqual(len(self._widget1._plot_area._items), 6)
        self.assertEqual(len(self._widget2._plot_area._items), 6)

    def testForwardMethod(self):
        widget = self._widget1

        for method in ["removeAllItems", "setAspectLocked", "setLabel", "setTitle",
                       "setAnnotationList", "addLegend", "invertX", "invertY", "autoRange"]:
            with patch.object(widget._plot_area, method) as mocked:
                getattr(widget, method)()
                mocked.assert_called_once()

    def testShowHideAxisLegend(self):
        widget = self._widget1

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

    def testPlot1(self):
        # widget1

        for i, plot in enumerate(self._plot_items1):
            x = np.arange(20)
            y = np.random.rand(20)
            y[-i-1:-1] = np.nan
            if isinstance(plot, StatisticsBarItem):
                plot.setData(x, y, y - 0.1, y + 0.1)
            else:
                plot.setData(x, y)
            _display()

        self._widget1._plot_area._onLogXChanged(True)
        _display()
        self._widget1._plot_area._onLogYChanged(True)
        _display()

        for plot in self._plot_items1:
            plot.setData([], [])
            _display()

        # widget2

        for i, plot in enumerate(self._plot_items2):
            x = np.arange(20)
            y = np.random.rand(20)
            y[-i-1:-1] = np.nan
            if isinstance(plot, StatisticsBarItem):
                plot.setData(x, y, y - 0.1, y + 0.1)
            else:
                plot.setData(x, y)
            _display()

        self._widget2._plot_area._onLogXChanged(True)
        _display()
        self._widget2._plot_area._onLogYChanged(True)
        _display()

        for plot in self._plot_items2:
            plot.setData([], [])
            _display()

    def testCrossCursor(self):
        widget = self._widget1
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
