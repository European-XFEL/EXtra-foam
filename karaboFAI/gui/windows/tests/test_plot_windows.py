import unittest
from collections import Counter
import os
import tempfile

from karaboFAI.logger import logger
from karaboFAI.config import _Config, ConfigWrapper
from karaboFAI.gui import mkQApp, MainGUI
from karaboFAI.gui.misc_widgets import BulletinWidget
from karaboFAI.gui.windows import (
    Bin1DWindow, OverviewWindow, PulsedAzimuthalIntegrationWindow,
    PumpProbeWindow, RoiWindow, XasWindow
)
from karaboFAI.gui.plot_widgets import (
    AssembledImageView, TrainAiWidget,
    PumpProbeOnOffWidget, PumpProbeFomWidget, PumpProbeImageView,
    PulsedFOMWidget, SinglePulseAiWidget, SinglePulseImageView,
    RoiImageView, RoiValueMonitor,
    XasSpectrumBinCountWidget, XasSpectrumWidget, XasSpectrumDiffWidget,
    BinCountWidget, BinWidget,
)


class TestOverviewWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.setLevel('CRITICAL')
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()  # ensure file

        mkQApp()
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

    def testPumpProbeWIndow(self):
        win = PumpProbeWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(8, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(5, counter[PumpProbeImageView])
        self.assertEqual(2, counter[PumpProbeOnOffWidget])
        self.assertEqual(1, counter[PumpProbeFomWidget])

    def testXasWindow(self):
        win = XasWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(6, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(3, counter[RoiImageView])
        self.assertEqual(1, counter[XasSpectrumWidget])
        self.assertEqual(1, counter[XasSpectrumDiffWidget])
        self.assertEqual(1, counter[XasSpectrumBinCountWidget])

    def testRoiWindow(self):
        win = RoiWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(5, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(4, counter[RoiImageView])
        self.assertEqual(1, counter[RoiValueMonitor])

    def testBin1DWindow(self):
        win = Bin1DWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[BinWidget])
        self.assertEqual(1, counter[BinCountWidget])


class TestPulsedAiWindow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logger.setLevel('CRITICAL')

        mkQApp()

    def testPulseResolved(self):
        gui = MainGUI()

        self._win = PulsedAzimuthalIntegrationWindow(
            pulse_resolved=True, parent=gui)

        self.assertEqual(6, len(self._win._plot_widgets))
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[TrainAiWidget])
        self.assertEqual(1, counter[PulsedFOMWidget])
        self.assertEqual(2, counter[SinglePulseAiWidget])
        self.assertEqual(2, counter[SinglePulseImageView])

        gui.close()

    def testTrainResolved(self):
        gui = MainGUI()

        self._win = PulsedAzimuthalIntegrationWindow(
            pulse_resolved=False, parent=gui)

        self.assertEqual(1, len(self._win._plot_widgets))
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[TrainAiWidget])

        gui.close()
