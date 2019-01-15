import unittest
from enum import IntEnum

from karaboFAI.main_gui import MainGUI


class Win(IntEnum):
    Overview = 2
    OnOffPulses = 3
    BraggSpots = 4
    DrawMask = 5


class TestMainGui(unittest.TestCase):
    gui = MainGUI('FXE')
    actions = gui._tool_bar.actions()

    def testOpenCloseWindows(self):
        count = 0
        for idx in (Win.Overview, Win.OnOffPulses, Win.BraggSpots):
            count += 1
            self.actions[idx].trigger()
            self.assertEqual(count, len(self.gui._plot_windows))

        self.actions[Win.DrawMask].trigger()
        self.assertEqual(count, len(self.gui._plot_windows))

        # test Window instances will be unregistered after being closed
        with self.assertRaises(StopIteration):
            i = 0
            while i < 100 and self.gui._plot_windows.keys():
                key = next(self.gui._plot_windows.keys())
                key.close()
                i += 1
