import unittest
from collections import Counter

from karaboFAI.widgets import (
    BulletinWidget, ImageAnalysisWidget, MultiPulseAiWidget,
    SampleDegradationWidget, SinglePulseAiWidget, SinglePulseImageWidget
)
from karaboFAI.main_fai_gui import MainFaiGUI
from karaboFAI.windows import OverviewWindow
from karaboFAI.data_processing import Data4Visualization


class TestOverviewWindow(unittest.TestCase):
    gui = MainFaiGUI('LPD')

    def setUp(self):
        self._win = OverviewWindow(Data4Visualization(), parent=self.gui)

    def testInstantiateOverviewWindow(self):
        self.assertEqual(len(self._win._plot_widgets), 8)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[ImageAnalysisWidget], 1)
        self.assertEqual(counter[MultiPulseAiWidget], 1)
        self.assertEqual(counter[SampleDegradationWidget], 1)
        self.assertEqual(counter[SinglePulseAiWidget], 2)
        self.assertEqual(counter[SinglePulseImageWidget], 2)
