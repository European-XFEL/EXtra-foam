import unittest
from collections import Counter

from karaboFAI.services import FaiServer
from karaboFAI.gui.plot_widgets import (
    AssembledImageView, RoiImageView, XasSpectrumBinCountWidget,
    XasSpectrumWidget, XasSpectrumDiffWidget
)
from karaboFAI.gui.bulletin_widget import BulletinWidget
from karaboFAI.gui.windows import XasWindow


class TestXasWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = FaiServer('LPD').gui

    def testInstantiation(self):
        self._win = XasWindow(pulse_resolved=True, parent=self.gui)

        self.assertEqual(len(self._win._plot_widgets), 8)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[AssembledImageView], 1)
        self.assertEqual(counter[RoiImageView], 3)
        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[XasSpectrumWidget], 1)
        self.assertEqual(counter[XasSpectrumDiffWidget], 1)
        self.assertEqual(counter[XasSpectrumBinCountWidget], 1)
