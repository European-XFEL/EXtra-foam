import unittest

from karaboFAI.widgets.pyqtgraph import mkQApp
from karaboFAI.main_gui import MainGUI


class TestMainGui(unittest.TestCase):
    app = mkQApp()

    gui = MainGUI('FXE')
    actions = gui._tool_bar.actions()

    def testInstantiateOverviewWindow(self):
        pass

    def testOpenWindows(self):
        count = 0
        for idx in (2, 3, 4):
            count += 1
            self.actions[idx].trigger()
            self.assertEqual(count, len(self.gui._plot_windows))

        self.actions[5].trigger()
        self.assertEqual(count, len(self.gui._plot_windows))
