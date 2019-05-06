import unittest
from collections import Counter

from karaboFAI.services import FaiServer
from karaboFAI.gui.bulletin_widget import BulletinWidget
from karaboFAI.gui.windows.base_window import AbstractWindow
from karaboFAI.gui.windows import (
    OverviewWindow, PulsedAzimuthalIntegrationWindow, PumpProbeWindow,
    RoiWindow, SingletonWindow, XasWindow
)
from karaboFAI.gui.plot_widgets import (
    AssembledImageView, MultiPulseAiWidget, PumpProbeOnOffWidget,
    PumpProbeFomWidget, PumpProbeImageView,
    PulsedFOMWidget, SinglePulseAiWidget, SinglePulseImageView,
    RoiImageView, RoiValueMonitor, XasSpectrumBinCountWidget,
    XasSpectrumWidget, XasSpectrumDiffWidget
)


class TestOverviewWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = FaiServer('LPD').gui

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testOverviewWindow(self):
        win = OverviewWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(len(win._plot_widgets), 2)
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[AssembledImageView], 1)

    def testPumpProbeWIndow(self):
        win = PumpProbeWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(len(win._plot_widgets), 8)
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[PumpProbeImageView], 5)
        self.assertEqual(counter[PumpProbeOnOffWidget], 2)
        self.assertEqual(counter[PumpProbeFomWidget], 1)

    def testXasWindow(self):
        win = XasWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(len(win._plot_widgets), 6)
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[RoiImageView], 3)
        self.assertEqual(counter[XasSpectrumWidget], 1)
        self.assertEqual(counter[XasSpectrumDiffWidget], 1)
        self.assertEqual(counter[XasSpectrumBinCountWidget], 1)

    def testRoiWindow(self):
        win = RoiWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(len(win._plot_widgets), 5)
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[RoiImageView], 4)
        self.assertEqual(counter[RoiValueMonitor], 1)


class TestPulsedAiWindow(unittest.TestCase):

    def testPulseResolved(self):
        gui = FaiServer('LPD').gui

        self._win = PulsedAzimuthalIntegrationWindow(
            pulse_resolved=True, parent=gui)

        self.assertEqual(len(self._win._plot_widgets),6)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[MultiPulseAiWidget], 1)
        self.assertEqual(counter[PulsedFOMWidget], 1)
        self.assertEqual(counter[SinglePulseAiWidget], 2)
        self.assertEqual(counter[SinglePulseImageView], 2)

        gui.close()

    def testTrainResolved(self):
        gui = FaiServer('JungFrau').gui
        self._win = PulsedAzimuthalIntegrationWindow(
            pulse_resolved=False, parent=gui)

        self.assertEqual(len(self._win._plot_widgets), 1)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[SinglePulseAiWidget], 1)

        gui.close()


class TestSingletonWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FaiServer.make_app()
        SingletonWindow._instances.clear()

    @SingletonWindow
    class FooWindow(AbstractWindow):
        pass

    def test_singleton(self):
        win1 = self.FooWindow()
        win2 = self.FooWindow()
        self.assertEqual(win1, win2)

        self.assertEqual(1, len(SingletonWindow._instances))
        key = list(SingletonWindow._instances.keys())[0]
        self.assertEqual('FooWindow', key.__name__)
