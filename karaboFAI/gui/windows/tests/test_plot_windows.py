import unittest
from collections import Counter, deque
import os
import tempfile

from karaboFAI.logger import logger
from karaboFAI.config import _Config, ConfigWrapper, config
from karaboFAI.gui import mkQApp, MainGUI
from karaboFAI.gui.windows import (
    Bin1dWindow, Bin2dWindow, CorrelationWindow,
    StatisticsWindow, PumpProbeWindow, RoiWindow
)
from karaboFAI.gui.plot_widgets import (
    FomHistogramWidget,
    PumpProbeOnOffWidget, PumpProbeFomWidget, PumpProbeImageView,
    PulsesInTrainFomWidget,
    RoiImageView,
    Bin1dHist, Bin1dHeatmap, Bin2dHeatmap,
    CorrelationWidget,
)
from karaboFAI.pipeline.data_model import ProcessedData

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
        win = PumpProbeWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(5, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[PumpProbeImageView])
        self.assertEqual(2, counter[PumpProbeOnOffWidget])
        self.assertEqual(1, counter[PumpProbeFomWidget])

    def testRoiWindow(self):
        win = RoiWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[RoiImageView])

    def testBin1dWindow(self):
        win = Bin1dWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(6, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[Bin1dHeatmap])
        self.assertEqual(4, counter[Bin1dHist])

    def testBin2dWindow(self):
        win = Bin2dWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[Bin2dHeatmap])

    def testCorrelationWindow(self):
        from karaboFAI.gui.ctrl_widgets.correlation_ctrl_widget import _N_PARAMS

        win = CorrelationWindow(deque(maxlen=1), pulse_resolved=True, parent=self.gui)

        self.assertEqual(_N_PARAMS, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(_N_PARAMS, counter[CorrelationWidget])

        # -----------------------
        # test data visualization
        # -----------------------

        # the upper two plots have error bars
        data = ProcessedData(1)
        for i in range(1000):
            data.corr.correlation1.hist = (int(i/5), 100*i)
            data.corr.correlation2.hist = (int(i/5), -100*i)
            data.corr.correlation3.hist = (i, i+1)
            data.corr.correlation4.hist = (i, -i)

        win._queue.append(data)
        win.updateWidgetsF()
        app.processEvents()

        # change the resolutions
        data.corr.correlation1.reset = True
        data.corr.correlation2.reset = True
        data.corr.correlation3.resolution = 15
        data.corr.correlation4.resolution = 20
        data.corr.update_hist()

        # the data is cleared after the resolutions were changed
        # now the lower two plots have error bars but the upper ones do not
        for i in range(1000):
            data.corr.correlation1.hist = (i, i+1)
            data.corr.correlation2.hist = (i, -i)
            data.corr.correlation3.hist = (int(i/5), 100*i)
            data.corr.correlation4.hist = (int(i/5), -100*i)

        win._queue.append(data)
        win.updateWidgetsF()
        app.processEvents()

    def testStatisticsWindow(self):
        win = StatisticsWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[PulsesInTrainFomWidget])
        self.assertEqual(1, counter[FomHistogramWidget])

    def testPulseOfInterestWindow(self):
        from karaboFAI.gui.windows.pulse_of_interest_w import (
            PulseOfInterestWindow, PoiImageView, PoiStatisticsWidget
        )

        win = PulseOfInterestWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(4, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[PoiImageView])
        self.assertEqual(2, counter[PoiStatisticsWidget])

    def testTrXasWindow(self):
        from karaboFAI.gui.windows.tri_xas_w import (
            TrXasWindow, _TrXasAbsorptionWidget, _TrXasHeatmap
        )
        win = TrXasWindow(deque(), pulse_resolved=True, parent=self.gui)

        self.assertEqual(6, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(3, counter[RoiImageView])
        self.assertEqual(2, counter[_TrXasAbsorptionWidget])
        self.assertEqual(1, counter[_TrXasHeatmap])
