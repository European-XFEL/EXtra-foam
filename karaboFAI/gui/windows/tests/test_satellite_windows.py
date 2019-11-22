import unittest

from karaboFAI.gui import mkQApp, MainGUI
from karaboFAI.logger import logger
from karaboFAI.processes import ProcessInfoList
from karaboFAI.gui.windows import ProcessMonitor

app = mkQApp()

logger.setLevel('CRITICAL')


class TestProcessMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = MainGUI()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testProcessMonitor(self):
        win = self.gui.onOpenSatelliteWindow(ProcessMonitor)
        self.assertIsInstance(win, ProcessMonitor)

        self.gui.process_info_sgn.emit([ProcessInfoList(
            name='ZeroMQ',
            fai_name='fai name',
            fai_type='fai type',
            pid=1234,
            status='zombie'
        )])
        self.assertIn("ZeroMQ", win._cw.toPlainText())
        self.assertIn("zombie", win._cw.toPlainText())

        # test old text will be removed
        self.gui.process_info_sgn.emit([ProcessInfoList(
            name='kafka',
            fai_name='fai name',
            fai_type='fai type',
            pid=1234,
            status='sleeping'
        )])
        self.assertNotIn("zombie", win._cw.toPlainText())
        self.assertNotIn("ZeroMQ", win._cw.toPlainText())
        self.assertIn("kafka", win._cw.toPlainText())
        self.assertIn("sleeping", win._cw.toPlainText())
