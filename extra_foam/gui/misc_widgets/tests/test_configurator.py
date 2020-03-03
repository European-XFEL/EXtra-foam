import unittest
from unittest.mock import MagicMock, patch

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QMessageBox

from extra_foam.gui import mkQApp
from extra_foam.gui.misc_widgets.configurator import Configurator

app = mkQApp()


class TestConfigurator(unittest.TestCase):
    def setUp(self):
        self._widget = Configurator()
        self._widget._meta = MagicMock()
        self._widget._meta.take_snapshot.return_value = (
            self._widget.LAST_SAVED, "2020-01-01 01:01:01", "")
        self._widget.onInit()

    def testAddCopyRemoveConfiguration(self):
        widget = self._widget
        table = widget._table

        self.assertEqual(1, table.rowCount())
        self.assertEqual(widget.LAST_SAVED, table.item(0, 0).text())
        self.assertEqual("2020-01-01 01:01:01", table.item(0, 1).text())
        self.assertEqual("", table.item(0, 2).text())

        # test add

        cfg = ["abc", "2020-02-02 02:02:02", "abc setup"]
        widget._insertConfigurationToList(cfg)
        self.assertEqual("abc", table.item(1, 0).text())
        self.assertEqual("2020-02-02 02:02:02", table.item(1, 1).text())
        self.assertEqual("abc setup", table.item(1, 2).text())
        self.assertDictEqual({widget.LAST_SAVED: 0, 'abc': 1}, widget._config)

        # test copy

        widget._copyConfiguration(1, "efg")
        self.assertDictEqual({widget.LAST_SAVED: 0, 'abc': 1, 'efg': 2}, widget._config)

        # test remove

        widget._removeConfiguration(1)
        self.assertEqual("efg", table.item(1, 0).text())
        self.assertEqual("2020-02-02 02:02:02", table.item(1, 1).text())
        self.assertEqual("abc setup", table.item(1, 2).text())
        self.assertDictEqual({widget.LAST_SAVED: 0, 'efg': 1}, widget._config)
        self._widget._meta.remove_snapshot.assert_called_with('abc')

        widget._removeConfiguration(1)
        self.assertDictEqual({widget.LAST_SAVED: 0}, widget._config)
        self._widget._meta.remove_snapshot.assert_called_with('efg')

    def testRenameConfiguration(self):
        widget = self._widget
        table = widget._table

        cfg = ["efg", "2020-03-03 03:03:03", "efg setup"]
        widget._insertConfigurationToList(cfg)
        widget._renameConfiguration(1, "abc")
        widget._meta.rename_snapshot.assert_called_with('efg', 'abc')
        self.assertEqual("abc", table.item(1, 0).text())
        self.assertDictEqual({widget.LAST_SAVED: 0, 'abc': 1}, widget._config)

    def testSetConfiguration(self):
        widget = self._widget
        table = widget._table
        spy = QSignalSpy(widget.load_metadata_sgn)
        spy_count = 0

        cfg = ["efg", "2020-03-03 03:03:03", "efg setup"]
        widget._insertConfigurationToList(cfg)

        for row in range(2):
            for i in range(2):
                table.itemDoubleClicked.emit(table.item(row, i))
                widget._meta.load_snapshot.assert_called_with(table.item(row, i).text())
                widget._meta.load_snapshot.reset_mock()
                spy_count += 1
                self.assertEqual(spy_count, len(spy))
            table.itemDoubleClicked.emit(table.item(row, 2))
            widget._meta.load_snapshot.assert_not_called()
            self.assertEqual(spy_count, len(spy))

        # test reset
        spy = QSignalSpy(widget.load_metadata_sgn)
        widget._reset_btn.clicked.emit()
        widget._meta.load_snapshot.assert_called_with(widget.DEFAULT)
        self.assertEqual(1, len(spy))

    def testSaveLoad(self):
        widget = self._widget
        table = widget._table

        widget._insertConfigurationToList(["abc", "2020-02-02 02:02:02", "abc setup"])
        widget._insertConfigurationToList(["efg", "2020-03-03 03:03:03", "efg setup"])

        with patch("PyQt5.QtWidgets.QMessageBox.question", return_value=QMessageBox.No):
            widget._save_cfg_btn.clicked.emit()
            widget._meta.dump_configurations.assert_not_called()

        with patch("PyQt5.QtWidgets.QMessageBox.question", return_value=QMessageBox.Yes):
            widget._save_cfg_btn.clicked.emit()
            widget._meta.dump_configurations.assert_called_with(
                [(widget.LAST_SAVED, ""), ("abc", "abc setup"), ("efg", "efg setup")])

        with patch("PyQt5.QtWidgets.QMessageBox.question", return_value=QMessageBox.No):
            widget._meta.load_configurations.reset_mock()
            widget._load_cfg_btn.clicked.emit()
            widget._meta.load_configurations.assert_not_called()

        with patch("PyQt5.QtWidgets.QMessageBox.question", return_value=QMessageBox.Yes):
            widget._meta.load_configurations.return_value = [
                (widget.LAST_SAVED, "2020-02-13 04:04:04", ""),
                ("abc", "2020-02-12 02:02:02", "abc setup 1"),
                ("efg", "2020-03-13 03:03:03", "efg setup 1")
            ]
            widget._load_cfg_btn.clicked.emit()
            widget._meta.load_configurations.assert_called_once()

        self.assertEqual(3, table.rowCount())
        self.assertEqual(widget.LAST_SAVED, table.item(0, 0).text())
        self.assertEqual("2020-02-13 04:04:04", table.item(0, 1).text())
        self.assertEqual("abc", table.item(1, 0).text())
        self.assertEqual("2020-02-12 02:02:02", table.item(1, 1).text())
        self.assertEqual("abc setup 1", table.item(1, 2).text())
        self.assertEqual("efg", table.item(2, 0).text())
        self.assertEqual("2020-03-13 03:03:03", table.item(2, 1).text())
        self.assertEqual("efg setup 1", table.item(2, 2).text())
