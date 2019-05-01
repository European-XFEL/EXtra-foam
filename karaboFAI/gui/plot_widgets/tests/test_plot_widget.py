import unittest
from unittest.mock import MagicMock

from karaboFAI.services import FaiServer
from karaboFAI.gui.plot_widgets.plot_widgets import (
    PlotWidget, PumpProbeOnOffWidget
)


class TestPlotWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FaiServer().app

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
        self.app.processEvents()

        # test set empty data
        plot.setData([], [])
        self.app.processEvents()

        # test if x and y have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [])

    def testErrorBarPlot(self):
        # set any valid number
        plot = self._widget.plotErrorBar([1, 2], [3, 4])
        self.app.processEvents()

        # set x, y, y_min and y_max together
        plot.setData([1, 2], [1, 2], y_min=[0, 0], y_max=[2, 2])
        self.app.processEvents()

        # test set empty data
        plot.setData([], [])
        self.app.processEvents()

        # test if x and y have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [], y_min=[0, 0, 0], y_max=[2, 2, 2])

        # test if y_min/ymax have different lengths
        with self.assertRaises(ValueError):
            plot.setData([1, 2, 3], [1, 2, 3], y_min=[0, 0, 0], y_max=[2, 2])


class TestPumpProbeWidgets(unittest.TestCase):

    class Data:
        class PP:
            def __init__(self):
                self.frame_rate = 1
                self.data = None, None, None, None

        def __init__(self):
            self.pp = self.PP()

    def testPumpProbeOnOffWidgetFrameRate1(self):
        data = self.Data()
        widget = PumpProbeOnOffWidget()
        widget._on_pulse.setData = MagicMock()
        widget._off_pulse.setData = MagicMock()
        func1 = widget._on_pulse.setData
        func2 = widget._off_pulse.setData

        # no data
        widget.update(data)
        func1.assert_not_called()
        func2.assert_not_called()

        # data comes
        data.pp.data = [3], [5], [4], [1]
        widget.update(data)
        func1.assert_called_once_with([3], [5])
        func2.assert_called_once_with([3], [4])
        func1.reset_mock()
        func2.reset_mock()

        # no data
        data.pp.data = None, None, None, None
        widget.update(data)
        func1.assert_not_called()
        func2.assert_not_called()

    def testPumpProbeOnOffWidgetFrameRate2(self):
        data = self.Data()
        data.pp.frame_rate = 2
        widget = PumpProbeOnOffWidget()
        widget._on_pulse.setData = MagicMock()
        widget._off_pulse.setData = MagicMock()
        func1 = widget._on_pulse.setData
        func2 = widget._off_pulse.setData

        # data comes
        data.pp.data = [3], [5], [4], [1]
        widget.update(data)
        func1.assert_called_once_with([3], [5])
        func2.assert_called_once_with([3], [4])
        func1.reset_mock()
        func2.reset_mock()

        # no data, use cached data
        data.pp.data = None, None, None, None
        widget.update(data)
        func1.assert_called_once_with([3], [5])
        func2.assert_called_once_with([3], [4])
        func1.reset_mock()
        func2.reset_mock()

        # still no data
        data.pp.data = None, None, None, None
        widget.update(data)
        func1.assert_not_called()
        func2.assert_not_called()
        self.assertIs(widget._data, None)  # cached data should be reset

    def testPumpProbeDiffWidgetFrameRate1(self):
        data = self.Data()
        widget = PumpProbeOnOffWidget(diff=True)
        widget._on_off_pulse.setData = MagicMock()
        func = widget._on_off_pulse.setData

        # no data
        widget.update(data)
        self.assertFalse(func.called)

        # data comes
        data.pp.data = [3], [5], [4], [1]
        widget.update(data)
        func.assert_called_once_with([3], [1])
        func.reset_mock()

        # no data
        data.pp.data = None, None, None, None
        widget.update(data)
        func.assert_not_called()

    def testPumpProbeDiffWidgetFrameRate2(self):
        data = self.Data()
        data.pp.frame_rate = 2
        widget = PumpProbeOnOffWidget(diff=True)
        widget._on_off_pulse.setData = MagicMock()
        func = widget._on_off_pulse.setData

        # data comes
        data.pp.data = [3], [5], [4], [1]
        widget.update(data)
        func.assert_called_once_with([3], [1])
        func.reset_mock()

        # no data, use cached data
        data.pp.data = None, None, None, None
        widget.update(data)
        func.assert_called_once_with([3], [1])
        func.reset_mock()

        # still no data
        data.pp.data = None, None, None, None
        widget.update(data)
        func.assert_not_called()
        self.assertIs(widget._data, None)  # cached data should be reset
