import unittest

from karaboFAI.widgets.pyqtgraph import mkQApp
from karaboFAI.main_gui import MainGUI


class TestMainGui(unittest.TestCase):
    app = mkQApp()

    def setUp(self):
        self._gui = MainGUI('FXE')

    def testInstantiateOverviewWindow(self):
        pass
