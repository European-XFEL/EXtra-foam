import unittest
from enum import IntEnum

from karaboFAI.main_fai_gui import MainFaiGUI


class FaiWin(IntEnum):
    DrawMask = 2
    Overview = 5


class TestMainFaiGui(unittest.TestCase):
    gui = MainFaiGUI('LPD')
    actions = gui._tool_bar.actions()

    def testOpenCloseWindows(self):
        count = 0
        for idx in (FaiWin.DrawMask, FaiWin.Overview):
            count += 1
            self.actions[idx].trigger()
            self.assertEqual(count, len(self.gui._windows))

        # test Window instances will be unregistered after being closed
        with self.assertRaises(StopIteration):
            i = 0
            while i < 100 and self.gui._windows.keys():
                key = next(self.gui._windows.keys())
                key.close()
                i += 1
