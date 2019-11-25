import unittest

from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtCore import Qt

from extra_foam.gui import mkQApp
from extra_foam.gui.ctrl_widgets.smart_widgets import (
    SmartLineEdit, SmartBoundaryLineEdit, SmartRangeLineEdit,
    SmartSliceLineEdit, SmartStringLineEdit
)
from extra_foam.logger import logger

app = mkQApp()

logger.setLevel("CRITICAL")


class TestSmartLineEdit(unittest.TestCase):
    def testSmartLineEdit(self):
        widget = SmartLineEdit()

        QTest.keyClicks(widget, 'abc')
        self.assertTrue(widget._text_modified)
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertFalse(widget._text_modified)

        spy = QSignalSpy(widget.returnPressed)
        widget.setText('abc')
        self.assertEqual(1, len(spy))
        widget.setTextWithoutSignal('efg')
        self.assertEqual(1, len(spy))

    def testSmartStringLineEdit(self):
        widget = SmartStringLineEdit("abc")
        spy = QSignalSpy(widget.value_changed_sgn)

        # set an empty string
        widget.clear()
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(0, len(spy))

        # set a space
        widget.clear()
        QTest.keyClicks(widget, ' ')
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(0, len(spy))

        # a string started with a space is allowed
        widget.clear()
        QTest.keyClicks(widget, ' Any')
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(1, len(spy))

        # a Karabo device ID
        widget.clear()
        QTest.keyClicks(widget, 'SA3_XTD10_MONO/MDL/PHOTON_ENERGY')
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(2, len(spy))

        # a string started with number and contains special characters and white spaces
        widget.clear()
        QTest.keyClicks(widget, '123 *$ Any')
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(3, len(spy))

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
        spy = QSignalSpy(widget.value_changed_sgn)
        self.assertEqual(0, len(spy))

        widget.clear()
        QTest.keyClicks(widget, "0, 2")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("0, 2", widget.text())
        self.assertEqual("0, 2", widget._cached)
        self.assertEqual(1, len(spy))

        # set an invalid value
        widget.clear()
        QTest.keyClicks(widget, "2, 0")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("2, 0", widget.text())
        self.assertEqual("0, 2", widget._cached)
        self.assertEqual(1, len(spy))

        # set a valid value again
        widget.clear()
        QTest.keyClicks(widget, "0, 2")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("0, 2", widget.text())
        self.assertEqual("0, 2", widget._cached)
        self.assertEqual(2, len(spy))

    def testSmartRangeLineEdit(self):
        # test initialization with invalid content
        with self.assertRaises(ValueError):
            SmartRangeLineEdit("0:10:")

        # test initialization
        widget = SmartRangeLineEdit("0:10:1")
        self.assertEqual("0:10:1", widget._cached)
        spy = QSignalSpy(widget.value_changed_sgn)
        self.assertEqual(0, len(spy))

        widget.clear()
        QTest.keyClicks(widget, "0:10:2")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("0:10:2", widget.text())
        self.assertEqual("0:10:2", widget._cached)
        self.assertEqual(1, len(spy))

        # set an invalid value
        widget.clear()
        QTest.keyClicks(widget, "0:10:")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("0:10:", widget.text())
        self.assertEqual("0:10:2", widget._cached)
        self.assertEqual(1, len(spy))

        # set a valid value again
        widget.clear()
        QTest.keyClicks(widget, "0:10")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("0:10", widget.text())
        self.assertEqual("0:10", widget._cached)
        self.assertEqual(2, len(spy))

    def testSmartSliceLineEdit(self):
        # test initialization with invalid content
        with self.assertRaises(ValueError):
            SmartSliceLineEdit("")

        # test initialization
        widget = SmartRangeLineEdit("0:10:1")
        self.assertEqual("0:10:1", widget._cached)
        spy = QSignalSpy(widget.value_changed_sgn)
        self.assertEqual(0, len(spy))

        # set an invalid value
        widget.clear()
        QTest.keyClicks(widget, "0:10:2:2")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("0:10:2:2", widget.text())
        self.assertEqual("0:10:1", widget._cached)
        self.assertEqual(0, len(spy))

        # set a valid value again
        widget.clear()
        QTest.keyClicks(widget, "0:10")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("0:10", widget.text())
        self.assertEqual("0:10", widget._cached)
        self.assertEqual(1, len(spy))
