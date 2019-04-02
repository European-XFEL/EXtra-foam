import unittest
from collections import Counter

from karaboFAI.gui.plot_widgets import (
    AssembledImageView, RoiImageView, XasSpectrumBinCountWidget,
    XasSpectrumWidget, XasSpectrumDiffWidget
)
from karaboFAI.gui.bulletin_widget import BulletinWidget
from karaboFAI.gui.main_gui import MainGUI
from karaboFAI.gui.windows import XasWindow


class TestXasWindow(unittest.TestCase):

    def testInstantiation(self):
        main_gui = MainGUI('LPD')

        self._win = XasWindow(pulse_resolved=True, parent=main_gui)

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
