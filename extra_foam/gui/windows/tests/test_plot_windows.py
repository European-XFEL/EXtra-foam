import unittest
from collections import Counter, deque
import os
import tempfile

from extra_foam.logger import logger
from extra_foam.config import _Config, ConfigWrapper, config
from extra_foam.gui import mkQApp, MainGUI
from extra_foam.gui.windows import (
    BinningWindow, StatisticsWindow, PumpProbeWindow, RoiWindow
)
from extra_foam.gui.plot_widgets import RoiImageView
from extra_foam.pipeline.data_model import ProcessedData

app = mkQApp()

logger.setLevel('CRITICAL')


class TestPlotWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()  # ensure file
        # FIXME: 1. we must load a detector and set the topic since it is required
        #        by the tree model.
        #        2. if we set "DSSC" and "SCS" here, it affects other tests.
        config.load("LPD")
        config.set_topic("FXE")

        cls.gui = MainGUI()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testPumpProbeWindow(self):
        from extra_foam.gui.windows.pump_probe_w import (
            PumpProbeImageView, PumpProbeVFomPlot, PumpProbeFomPlot
        )
        win = PumpProbeWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(5, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[PumpProbeImageView])
        self.assertEqual(2, counter[PumpProbeVFomPlot])
        self.assertEqual(1, counter[PumpProbeFomPlot])

    def testRoiWindow(self):
        win = RoiWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[RoiImageView])

    def testBinningWindow(self):
        from extra_foam.gui.windows.bin_w import Bin1dHeatmap, Bin1dHist, Bin2dHeatmap

        win = BinningWindow(deque(maxlen=1), pulse_resolved=True, parent=self.gui)

        self.assertEqual(5, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[Bin1dHeatmap])
        self.assertEqual(2, counter[Bin1dHist])
        self.assertEqual(2, counter[Bin2dHeatmap])

    def testStatisticsWindow(self):
        from extra_foam.gui.windows.statistics_w import (
            CorrelationPlot, FomHist, InTrainFomPlot
        )

        win = StatisticsWindow(deque(maxlen=1), pulse_resolved=True, parent=self.gui)

        self.assertEqual(4, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[InTrainFomPlot])
        self.assertEqual(1, counter[FomHist])
        self.assertEqual(2, counter[CorrelationPlot])

    def testPulseOfInterestWindow(self):
        from extra_foam.gui.windows.pulse_of_interest_w import (
            PulseOfInterestWindow, PoiImageView, PoiHist
        )

        win = PulseOfInterestWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(4, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[PoiImageView])
        self.assertEqual(2, counter[PoiHist])

    def testTrXasWindow(self):
        from extra_foam.gui.windows.tri_xas_w import (
            TrXasWindow, TrXasAbsorptionPlot, TrXasHeatmap
        )
        win = TrXasWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(6, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(3, counter[RoiImageView])
        self.assertEqual(2, counter[TrXasAbsorptionPlot])
        self.assertEqual(1, counter[TrXasHeatmap])
