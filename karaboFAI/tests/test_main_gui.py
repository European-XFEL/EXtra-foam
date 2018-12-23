import unittest
import logging

from karaboFAI.widgets.pyqtgraph import mkQApp
from karaboFAI.main_gui import MainGUI


class TestMainGui(unittest.TestCase):
    app = mkQApp()

    def setUp(self):
        self._gui = MainGUI('FXE')

    def testInstantiateOverviewWindow(self):
        pass

    def tearDown(self):
        # TODO: disable logger during test
        # must remove this handler from the logger, otherwise other tests
        # which use logging will fail!
        logging.getLogger().removeHandler(self._gui._logger)
