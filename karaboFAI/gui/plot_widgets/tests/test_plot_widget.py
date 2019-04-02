import unittest

from karaboFAI.gui.plot_widgets.plot_widgets import PlotWidget

from . import app


class TestPlotWidget(unittest.TestCase):
    def setUp(self):
        self._widget = PlotWidget()

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
        app.processEvents()

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
        app.processEvents()

        # test if x and y have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [], y_min=[0, 0, 0], y_max=[2, 2, 2])

        # test if y_min/ymax have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [1, 2, 3], y_min=[0, 0, 0], y_max=[2, 2])
