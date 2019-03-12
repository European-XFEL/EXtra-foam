import unittest

from karaboFAI.gui.plot_widgets.plot_widget import PlotWidget


class TestPlotWidget(unittest.TestCase):
    def setUp(self):
        self._widget = PlotWidget()

        self._widget.plotCurve()
        self._widget.plotScatter()
        self._widget.plotBar()

    def testPlot(self):
        self.assertEqual(len(self._widget.plotItem.items), 3)

    def testClear(self):
        self._widget.clear()
        self.assertFalse(self._widget.plotItem.items)
