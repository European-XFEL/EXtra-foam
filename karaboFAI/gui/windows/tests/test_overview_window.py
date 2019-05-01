import unittest
from collections import Counter

from karaboFAI.services import FaiServer
from karaboFAI.gui.plot_widgets import (
    AssembledImageView, MultiPulseAiWidget, RoiImageView, RoiValueMonitor,
    PulseResolvedAiFomWidget, SinglePulseAiWidget, SinglePulseImageView
)
from karaboFAI.gui.bulletin_widget import BulletinWidget
from karaboFAI.gui.windows import OverviewWindow


class TestOverviewWindow(unittest.TestCase):

    def testPulseResolved(self):
        gui = FaiServer('LPD').gui

        self._win = OverviewWindow(pulse_resolved=True, parent=gui)

        self.assertEqual(len(self._win._plot_widgets), 11)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[AssembledImageView], 1)
        self.assertEqual(counter[MultiPulseAiWidget], 1)
        self.assertEqual(counter[PulseResolvedAiFomWidget], 1)
        self.assertEqual(counter[SinglePulseAiWidget], 2)
        self.assertEqual(counter[SinglePulseImageView], 2)
        self.assertEqual(counter[RoiImageView], 2)
        self.assertEqual(counter[RoiImageView], 2)
        self.assertEqual(counter[RoiValueMonitor], 1)

        gui.close()

    def testTrainResolved(self):
        gui = FaiServer('JungFrau').gui
        self._win = OverviewWindow(pulse_resolved=False, parent=gui)

        self.assertEqual(len(self._win._plot_widgets), 6)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[AssembledImageView], 1)
        self.assertEqual(counter[SinglePulseAiWidget], 1)
        self.assertEqual(counter[RoiImageView], 2)
        self.assertEqual(counter[RoiImageView], 2)
        self.assertEqual(counter[RoiValueMonitor], 1)

        gui.close()

