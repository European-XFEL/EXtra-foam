import unittest
from unittest.mock import patch

from extra_foam.logger import logger
from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.plot_items import ScatterPlotItem
from extra_foam.pipeline.data_model import ProcessedData
from extra_foam.pipeline.tests import _TestDataMixin

app = mkQApp()

logger.setLevel('CRITICAL')


class testPumpProbeWidgets(_TestDataMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._data = cls.empty_data()

    def testPumpProbeVFomPlot(self):
        from extra_foam.gui.windows.pump_probe_w import PumpProbeVFomPlot

        widget = PumpProbeVFomPlot()
        widget.updateF(self._data)

    def testPumpProbeFomPlot(self):
        from extra_foam.gui.windows.pump_probe_w import PumpProbeFomPlot

        widget = PumpProbeFomPlot()
        widget._data = self._data
        widget.refresh()


class testPulseOfInterestWidgets(_TestDataMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._empty_data = cls.empty_data()
        cls._data, _ = cls.data_with_assembled(1001, (4, 2, 2), histogram=True)

    def testPoiImageView(self):
        from extra_foam.gui.windows.pulse_of_interest_w import PoiImageView

        widget = PoiImageView(0)
        data = self._empty_data
        widget.updateF(data)

    def testPoiFomHist(self):
        from extra_foam.gui.windows.pulse_of_interest_w import PoiFomHist

        widget = PoiFomHist(0)

        # empty data
        widget._data = self._empty_data
        widget.refresh()

        # non-empty data
        widget._data = self._data
        widget.refresh()

    def testPoiRoiHist(self):
        from extra_foam.gui.windows.pulse_of_interest_w import PoiRoiHist

        widget = PoiRoiHist(0)

        # empty data
        data = self._empty_data
        widget.updateF(data)

        # non-empty data
        data = self._data
        widget.updateF(data)


class testBinningWidgets(_TestDataMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._empty_data = cls.empty_data()

    def testBin1dHist(self):
        from extra_foam.gui.windows.binning_w import Bin1dHist

        widget = Bin1dHist()
        widget._data = self._empty_data

        widget.refresh()

    def testHeatmap1D(self):
        from extra_foam.gui.windows.binning_w import Bin1dHeatmap

        widget = Bin1dHeatmap()
        widget._data = self._empty_data

        # test "Auto level" reset
        widget._auto_level = True
        widget.refresh()
        self.assertFalse(widget._auto_level)

    def testHeatmap2D(self):
        from extra_foam.gui.windows.binning_w import Bin2dHeatmap

        for is_count in [False, True]:
            widget = Bin2dHeatmap(count=is_count)
            widget._data = self._empty_data

            # test "Auto level" reset
            widget._auto_level = True
            widget.refresh()
            self.assertFalse(widget._auto_level)


class testCorrrelationWidgets(_TestDataMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._empty_data = cls.empty_data()
        cls._data, _ = cls.data_with_assembled(1001, (4, 2, 2), correlation=True)

    def testGeneral(self):
        from extra_foam.gui.windows.correlation_w import CorrelationPlot

        for i in range(2):
            widget = CorrelationPlot(0)
            widget._data = self._empty_data
            widget.refresh()

    def testResolutionSwitch(self):
        from extra_foam.gui.windows.correlation_w import CorrelationPlot
        from extra_foam.gui.plot_widgets.plot_items import StatisticsBarItem, pg

        # resolution1 = 0.0 and resolution2 > 0.0
        widget = CorrelationPlot(0)
        widget._data = self._data
        widget.refresh()
        plot_item, plot_item_slave = widget._plot, widget._plot_slave
        self.assertIsInstance(plot_item, ScatterPlotItem)
        self.assertIsInstance(plot_item_slave, ScatterPlotItem)

        widget._idx = 1  # a trick
        widget.refresh()
        self.assertNotIn(plot_item, widget._plot_area._items)  # being deleted
        self.assertNotIn(plot_item_slave, widget._plot_area._items)  # being deleted
        plot_item, plot_item_slave = widget._plot, widget._plot_slave
        self.assertIsInstance(plot_item, StatisticsBarItem)
        self.assertIsInstance(plot_item_slave, StatisticsBarItem)
        self.assertEqual(2, plot_item._beam)
        self.assertEqual(2, plot_item_slave._beam)
        # beam size changes with resolution
        widget._data["processed"].corr[1].resolution = 4
        widget.refresh()
        self.assertEqual(4, plot_item._beam)
        self.assertEqual(4, plot_item_slave._beam)

        widget._idx = 0  # a trick
        widget.refresh()
        self.assertNotIn(plot_item, widget._plot_area._items)  # being deleted
        self.assertNotIn(plot_item_slave, widget._plot_area._items)  # being deleted
        self.assertIsInstance(widget._plot, ScatterPlotItem)
        self.assertIsInstance(widget._plot_slave, ScatterPlotItem)


class testHistogramWidgets(_TestDataMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._empty_data = cls.empty_data()
        cls._data, _ = cls.data_with_assembled(1001, (4, 2, 2), histogram=True)

    def testFomHist(self):
        from extra_foam.gui.windows.histogram_w import FomHist

        widget = FomHist()

        # empty data
        widget._data = self._empty_data
        widget.refresh()

        # non-empty data
        widget._data = self._data
        widget.refresh()

    def testInTrainFomPlot(self):
        from extra_foam.gui.windows.histogram_w import InTrainFomPlot

        widget = InTrainFomPlot()
        data = self._empty_data
        widget.updateF(data)
