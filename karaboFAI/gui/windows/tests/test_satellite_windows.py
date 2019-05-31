import unittest

from karaboFAI.gui import mkQApp, MainGUI
from karaboFAI.gui.windows import (
    ProcessMonitor
)


class TestProcessMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mkQApp()
        cls.gui = MainGUI()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testProcessMonitor(self):
        win = self.gui.openProcessMonitor()

        self.gui.process_info_sgn.emit(['a', 'b', 'c'])
        self.assertEqual("a\nb\nc", win._cw.toPlainText())
        # test old text will be removed
        self.gui.process_info_sgn.emit(['e', 'f'])
        self.assertEqual("e\nf", win._cw.toPlainText())
