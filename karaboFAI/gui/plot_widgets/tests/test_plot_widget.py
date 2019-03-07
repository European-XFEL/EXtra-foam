import unittest

from karaboFAI.gui.plot_widgets.plot_widget import PlotWidget


class TestPlotWidget(unittest.TestCase):
    def setUp(self):
        self._widget = PlotWidget()
        for i in range(5):
            self._widget.plot()

    def testPlot(self):
        self.assertEqual(len(self._widget.plotItem.items), 5)

    def testClear(self):
        self._widget.clear()
        self.assertFalse(self._widget.plotItem.items)
