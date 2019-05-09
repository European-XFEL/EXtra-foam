import unittest

from PyQt5.QtTest import QSignalSpy

from karaboFAI.logger import logger
from karaboFAI.services import FaiServer
from karaboFAI.gui.misc_widgets import SmartBoundaryLineEdit


class TestSmartLineEdit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FaiServer.qt_app()

    def testSmartBoundaryLineEdit(self):
        # require at least one argument for initialization
        with self.assertRaises(TypeError):
            SmartBoundaryLineEdit()

        # test initialization with invalid content
        with self.assertRaises(ValueError):
            SmartBoundaryLineEdit("0, -1")

        # test initialization
        widget = SmartBoundaryLineEdit("0, 1")
        self.assertEqual("0, 1", widget._cached)

        # set a valid value
        spy = QSignalSpy(widget.value_changed_sgn)
        self.assertEqual(0, len(spy))

        widget.setText("0, 2")
        widget.returnPressed.emit()
        self.assertEqual("0, 2", widget.text())
        self.assertEqual("0, 2", widget._cached)
        self.assertEqual(1, len(spy))

        # set an invalid value
        with self.assertLogs(logger=logger, level='ERROR'):
            widget.setText("2, 0")
            widget.returnPressed.emit()
        self.assertEqual("0, 2", widget.text())
        self.assertEqual("0, 2", widget._cached)
        self.assertEqual(1, len(spy))

        # set a valid value again
        widget.setText("0, 2")
        widget.returnPressed.emit()
        self.assertEqual(2, len(spy))
