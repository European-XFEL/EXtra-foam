import unittest
from collections import Counter
import os
import tempfile

from karaboFAI.logger import logger
from karaboFAI.config import _Config, ConfigWrapper
from karaboFAI.gui import mkQApp, MainGUI
from karaboFAI.gui.misc_widgets import BulletinWidget
from karaboFAI.gui.windows import (
    Bin1dWindow, Bin2dWindow, OverviewWindow, AzimuthalIntegrationWindow,
    StatisticsWindow, PumpProbeWindow, RoiWindow, PulseOfInterestWindow,
)
from karaboFAI.gui.plot_widgets import (
    AssembledImageView, TrainAiWidget, FomHistogramWidget,
    PumpProbeOnOffWidget, PumpProbeFomWidget, PumpProbeImageView,
    PoiStatisticsWidget, PulsesInTrainFomWidget, SinglePulseImageView,
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

        cls.gui = MainGUI()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testOverviewWindow(self):
        win = OverviewWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[BulletinWidget])
        self.assertEqual(1, counter[AssembledImageView])

    def testPumpProbeWindow(self):
        win = PumpProbeWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(5, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[PumpProbeImageView])
        self.assertEqual(2, counter[PumpProbeOnOffWidget])
        self.assertEqual(1, counter[PumpProbeFomWidget])

    def testRoiWindow(self):
        win = RoiWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(4, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(4, counter[RoiImageView])

    def testBin1dWindow(self):
        win = Bin1dWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(6, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[Bin1dHeatmap])
        self.assertEqual(4, counter[Bin1dHist])

    def testBin2dWindow(self):
        win = Bin2dWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[Bin2dHeatmap])

    def testCorrelationWindow(self):

        self.gui._tool_bar.actions()[6].trigger()
        win = list(self.gui._windows.keys())[-1]

        self.assertEqual(4, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(4, counter[CorrelationWidget])

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
        self.gui._data.set(data)
        win.update()
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

        self.gui._data.set(data)
        win.update()
        app.processEvents()

    def testStatisticsWindow(self):
        win = StatisticsWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[PulsesInTrainFomWidget])
        self.assertEqual(1, counter[FomHistogramWidget])

    def testPulseOfInterestWindow(self):
        win = PulseOfInterestWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(4, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[SinglePulseImageView])
        self.assertEqual(2, counter[PoiStatisticsWidget])

    def testAzimuthalIntegrationWindow(self):
        gui = MainGUI()

        self._win = AzimuthalIntegrationWindow(
            pulse_resolved=True, parent=gui)

        self.assertEqual(1, len(self._win._plot_widgets))
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[TrainAiWidget])

        gui.close()
