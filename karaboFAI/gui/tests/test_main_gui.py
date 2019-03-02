import unittest
from enum import IntEnum

from karaboFAI.gui.main_fai_gui import MainGUI


class Win(IntEnum):
    ImageTool = 2
    Overview = 3
    Correlation = 4


class TestMainFaiGui(unittest.TestCase):
    gui = MainGUI('LPD')
    actions = gui._tool_bar.actions()

    def testOpenCloseWindows(self):
        count = 0
        for idx in (Win.ImageTool, Win.Overview, Win.Correlation):
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
