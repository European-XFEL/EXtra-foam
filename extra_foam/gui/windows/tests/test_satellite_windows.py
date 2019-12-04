import unittest
from unittest.mock import patch

from extra_foam.gui import mkQApp
from extra_foam.logger import logger
from extra_foam.processes import ProcessInfoList
from extra_foam.gui.windows import ProcessMonitor

app = mkQApp()

logger.setLevel('CRITICAL')


class TestProcessMonitor(unittest.TestCase):

    @patch("extra_foam.gui.windows.process_monitor_w.list_foam_processes")
    def testProcessMonitor(self, func):
        win = ProcessMonitor()
        win._timer.stop()

        func.return_value = [ProcessInfoList(
            name='ZeroMQ',
            foam_name='foam name',
            foam_type='foam type',
            pid=1234,
            status='zombie'
        )]
        win.updateProcessInfo()
        self.assertIn("ZeroMQ", win._cw.toPlainText())
        self.assertIn("zombie", win._cw.toPlainText())

        # test old text will be removed
        func.return_value = [ProcessInfoList(
            name='kafka',
            foam_name='foam name',
            foam_type='foam type',
            pid=1234,
            status='sleeping'
        )]
        win.updateProcessInfo()
        self.assertNotIn("zombie", win._cw.toPlainText())
        self.assertNotIn("ZeroMQ", win._cw.toPlainText())
        self.assertIn("kafka", win._cw.toPlainText())
        self.assertIn("sleeping", win._cw.toPlainText())
