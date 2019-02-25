import unittest
from collections import Counter

from karaboFAI.widgets import (
    BulletinWidget, AssembledImageView, MultiPulseAiWidget, RoiImageView,
    RoiValueMonitor, SampleDegradationWidget, SinglePulseAiWidget,
    SinglePulseImageView
)
from karaboFAI.main_fai_gui import MainGUI
from karaboFAI.windows import OverviewWindow
from karaboFAI.data_processing import Data4Visualization


class TestOverviewWindow(unittest.TestCase):

    def testPulseResolvedOverviewWindow(self):
        main_gui = MainGUI('LPD')

        self._win = OverviewWindow(Data4Visualization(),
                                   mediator=main_gui._mediator,
                                   pulse_resolved=True,
                                   parent=main_gui)

        self.assertEqual(len(self._win._plot_widgets), 13)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[AssembledImageView], 1)
        self.assertEqual(counter[MultiPulseAiWidget], 1)
        self.assertEqual(counter[SampleDegradationWidget], 1)
        self.assertEqual(counter[SinglePulseAiWidget], 2)
        self.assertEqual(counter[SinglePulseImageView], 2)
        self.assertEqual(counter[RoiImageView], 2)
        self.assertEqual(counter[RoiImageView], 2)
        self.assertEqual(counter[RoiValueMonitor], 1)

    def testTrainResolvedOverviewWindow(self):
        main_gui = MainGUI('JungFrau')

        self._win = OverviewWindow(Data4Visualization(),
                                   mediator=main_gui._mediator,
                                   pulse_resolved=False,
                                   parent=main_gui)

        self.assertEqual(len(self._win._plot_widgets), 8)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[AssembledImageView], 1)
        self.assertEqual(counter[SinglePulseAiWidget], 1)
        self.assertEqual(counter[RoiImageView], 2)
        self.assertEqual(counter[RoiImageView], 2)
        self.assertEqual(counter[RoiValueMonitor], 1)
