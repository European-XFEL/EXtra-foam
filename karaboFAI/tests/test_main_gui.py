import unittest
from enum import IntEnum

from karaboFAI.main_fai_gui import MainFaiGUI
from karaboFAI.main_bdp_gui import MainBdpGUI


class FaiWin(IntEnum):
    DrawMask = 2
    Overview = 4
    OnOffPulses = 5


class TestMainFaiGui(unittest.TestCase):
    gui = MainFaiGUI('LPD')
    actions = gui._tool_bar.actions()

    def testOpenCloseWindows(self):
        count = 0
        for idx in (FaiWin.Overview, FaiWin.OnOffPulses):
            count += 1
            self.actions[idx].trigger()
            self.assertEqual(count, len(self.gui._plot_windows))

        self.actions[FaiWin.DrawMask].trigger()
        self.assertEqual(count, len(self.gui._plot_windows))

        # test Window instances will be unregistered after being closed
        with self.assertRaises(StopIteration):
            i = 0
            while i < 100 and self.gui._plot_windows.keys():
                key = next(self.gui._plot_windows.keys())
                key.close()
                i += 1


class BdpWin(IntEnum):
    DrawMask = 2
    BraggDiffractionPeak = 4


class TestMainBdpGui(unittest.TestCase):
    gui = MainBdpGUI('LPD')
    actions = gui._tool_bar.actions()

    def testOpenCloseWindows(self):
        count = 0
        for idx in (BdpWin.BraggDiffractionPeak,):
            count += 1
            self.actions[idx].trigger()
            self.assertEqual(count, len(self.gui._plot_windows))

        self.actions[BdpWin.DrawMask].trigger()
        self.assertEqual(count, len(self.gui._plot_windows))

        # test Window instances will be unregistered after being closed
        with self.assertRaises(StopIteration):
            i = 0
            while i < 100 and self.gui._plot_windows.keys():
                key = next(self.gui._plot_windows.keys())
                key.close()
                i += 1