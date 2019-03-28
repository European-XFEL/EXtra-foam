import unittest
from collections import Counter

from karaboFAI.gui.plot_widgets import (
    RoiImageView, XasSpectrumWidget, XasSpectrumDiffWidget
)
from karaboFAI.gui.bulletin_widget import BulletinWidget
from karaboFAI.gui.main_gui import MainGUI
from karaboFAI.gui.windows import XasWindow
from karaboFAI.pipeline import Data4Visualization


class TestXasWindow(unittest.TestCase):

    def testInstantiation(self):
        main_gui = MainGUI('LPD')

        self._win = XasWindow(Data4Visualization(),
                              pulse_resolved=True,
                              parent=main_gui)

        self.assertEqual(len(self._win._plot_widgets), 7)
        counter = Counter()
        for key in self._win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(counter[RoiImageView], 4)
        self.assertEqual(counter[BulletinWidget], 1)
        self.assertEqual(counter[XasSpectrumWidget], 1)
        self.assertEqual(counter[XasSpectrumDiffWidget], 1)
