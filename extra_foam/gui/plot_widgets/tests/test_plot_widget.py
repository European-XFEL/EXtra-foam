import unittest
from unittest.mock import MagicMock

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.plot_widget_base import PlotWidgetF, TimedPlotWidgetF
from extra_foam.logger import logger


app = mkQApp()

logger.setLevel("CRITICAL")


class TestPlotWidget(unittest.TestCase):
    def setUp(self):
        self._widget = PlotWidgetF()

    def testGeneral(self):
        self._widget.plotCurve()
        self._widget.plotScatter()
        self._widget.plotBar()
        self._widget.plotErrorBar()

        self.assertEqual(len(self._widget.plotItem.items), 4)

        self._widget.clear()
        self.assertFalse(self._widget.plotItem.items)

    def testBarPlot(self):
        # set any valid number
        plot = self._widget.plotBar([1, 2], [3, 4])
        app.processEvents()

        # test set empty data
        plot.setData([], [])

        # test if x and y have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [])

    def testErrorBarPlot(self):
        # set any valid number
        plot = self._widget.plotErrorBar([1, 2], [3, 4])
        app.processEvents()

        # set x, y, y_min and y_max together
        plot.setData([1, 2], [1, 2], y_min=[0, 0], y_max=[2, 2])
        app.processEvents()

        # test set empty data
        plot.setData([], [])

        # test if x and y have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [], y_min=[0, 0, 0], y_max=[2, 2, 2])

        # test if y_min/ymax have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [1, 2, 3], y_min=[0, 0, 0], y_max=[2, 2])


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
