import unittest
from collections import Counter

from karaboFAI.services import FaiServer
from karaboFAI.gui.plot_widgets import (
    PumpProbeOnOffWidget, PumpProbeDiffWidget, PumpProbeFomWidget,
    PumpProbeImageView
)
from karaboFAI.gui.bulletin_widget import BulletinWidget
from karaboFAI.gui.windows import PumpProbeWindow


class TestPumpProbeWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = FaiServer('LPD').gui

    def testInstantiation(self):
        self._win = PumpProbeWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(len(self._win._plot_widgets), 8)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[PumpProbeImageView], 4)
        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[PumpProbeOnOffWidget], 1)
        self.assertEqual(counter[PumpProbeDiffWidget], 1)
        self.assertEqual(counter[PumpProbeFomWidget], 1)
