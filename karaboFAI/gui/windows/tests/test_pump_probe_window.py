import unittest
from collections import Counter

from karaboFAI.gui.plot_widgets import (
    AssembledImageView, LaserOnOffAiWidget, LaserOnOffDiffWidget,
    LaserOnOffFomWidget, ReferenceImageView
)
from karaboFAI.gui.bulletin_widget import BulletinWidget
from karaboFAI.gui.main_gui import MainGUI
from karaboFAI.gui.windows import PumpProbeWindow


class TestPumpProbeWindow(unittest.TestCase):

    def testInstantiation(self):
        main_gui = MainGUI('LPD')

        self._win = PumpProbeWindow(pulse_resolved=True, parent=main_gui)

        self.assertEqual(len(self._win._plot_widgets), 6)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[AssembledImageView], 1)
        self.assertEqual(counter[ReferenceImageView], 1)
        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[LaserOnOffAiWidget], 1)
        self.assertEqual(counter[LaserOnOffDiffWidget], 1)
        self.assertEqual(counter[LaserOnOffFomWidget], 1)
