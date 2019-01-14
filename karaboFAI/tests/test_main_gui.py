import unittest

from karaboFAI.main_gui import MainGUI


class TestMainGui(unittest.TestCase):
    gui = MainGUI('FXE')
    actions = gui._tool_bar.actions()

    def testOpenCloseWindows(self):
        count = 0
        for idx in (2, 3, 4):
            count += 1
            self.actions[idx].trigger()
            self.assertEqual(count, len(self.gui._plot_windows))

        self.actions[5].trigger()
        self.assertEqual(count, len(self.gui._plot_windows))

        with self.assertRaises(StopIteration):
            i = 0
            while i < 100 and self.gui._plot_windows.keys():
                key = next(self.gui._plot_windows.keys())
                key.close()
                i += 1
