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

    def testErrorBarPlot(self):
        plot = self._widget.plotErrorBar()

        plot.setData([], [])
        app.processEvents()

        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [], y_min=[0, 0, 0], y_max=[2, 2, 2])

        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [1, 2, 3], y_min=[0, 0, 0], y_max=[2, 2])

        plot.setData([1, 2], [1, 2], y_min=[0, 0], y_max=[2, 2])
        app.processEvents()
