"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout, QHeaderView, QInputDialog, QLineEdit, QMenu, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QWidget,
)

from ...database import MetaProxy
from ...logger import logger


class AnalysisSetupManager(QWidget):

    load_metadata_sgn = pyqtSignal()

    DEFAULT = "default"
    LAST_SAVED = "Last saved"

    def __init__(self):
        super().__init__()

        self._table = QTableWidget()
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)

        self._snapshot_btn = QPushButton("Take snapshot")
        self._reset_btn = QPushButton("Reset to default")
        self._save_cfg_btn = QPushButton("Save all to file")
        self._load_cfg_btn = QPushButton("Load from file")

        self._meta = MetaProxy()
        self._config = dict()  # key: name, value: # of row

        self.initUI()
        self.initConnections()

    def initUI(self):
        table = self._table
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['Name', 'Timestamp', 'Description'])

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        layout = QGridLayout()
        layout.addWidget(table, 0, 0, 1, 4)
        layout.addWidget(self._snapshot_btn, 1, 0)
        layout.addWidget(self._reset_btn, 1, 1)
        layout.addWidget(self._save_cfg_btn, 1, 2)
        layout.addWidget(self._load_cfg_btn, 1, 3)
        self.setLayout(layout)

    def initConnections(self):
        self._snapshot_btn.clicked.connect(lambda x: self._takeSnapshot())
        self._reset_btn.clicked.connect(self._resetToDefault)
        self._table.itemDoubleClicked.connect(self.onItemDoubleClicked)
        self._table.customContextMenuRequested.connect(self.showContextMenu)
        self._save_cfg_btn.clicked.connect(self._askSaveSetups)
        self._load_cfg_btn.clicked.connect(self._askLoadSetups)

    def onInit(self):
        """Called by the MainGUI on initialization."""
        # save a default snapshot
        self._meta.copy_snapshot(None, self.DEFAULT)
        self._loadAllSetups()
        self._table.resizeColumnToContents(0)

    def onStart(self):
        self._table.setDisabled(True)
        self._snapshot_btn.setDisabled(True)
        self._reset_btn.setDisabled(True)
        self._save_cfg_btn.setDisabled(True)
        self._load_cfg_btn.setDisabled(True)

    def onStop(self):
        self._table.setEnabled(True)
        self._snapshot_btn.setEnabled(True)
        self._reset_btn.setEnabled(True)
        self._save_cfg_btn.setEnabled(True)
        self._load_cfg_btn.setEnabled(True)

    def _insertSetupToList(self, cfg, row=None):
        """Insert a row for the new setup.

        Note: this method does not involve any database operation.

        :param list/tuple cfg: (name, timestamp, description) of the setup.
        """
        table = self._table

        if cfg[0] == self.LAST_SAVED:
            row = 0
        elif row is None:
            row = table.rowCount()

        table.insertRow(row)
        for col, text in zip(range(table.columnCount()), cfg):
            item = QTableWidgetItem()
            table.setItem(row, col, item)
            flag = Qt.ItemIsEnabled
            if cfg[0] != self.LAST_SAVED and col == 2:
                flag |= Qt.ItemIsEditable
            item.setFlags(flag)
            try:
                item.setText(text)
            except TypeError:
                # I hope it will not happen in real life.
                item.setText(" ")
                logger.error(f"TypeError: Invalid value {text} for column "
                             f"{col+1} in AnalysisSetupManager!")

        if row != len(self._config):
            # adjust the row indices for the other setups
            for k, v in self._config.items():
                if v >= row:
                    self._config[k] += 1

        self._config[cfg[0]] = row

    def _removeSetupFromList(self, name):
        """Remove a row from the table and setup list.

        :param str name: name of the setup.
        """
        row = self._config[name]
        del self._config[name]
        self._table.removeRow(row)

        if row != len(self._config):
            # adjust the row indices for the other setups
            for k, v in self._config.items():
                if v > row:
                    self._config[k] -= 1

    def _copySetup(self, row, new_name):
        """Copy a row and insert the new one at the end of the table."""
        cfg = [self._table.item(row, i).text()
               for i in range(self._table.columnCount())]

        self._meta.copy_snapshot(cfg[0], new_name)
        cfg[0] = new_name
        self._insertSetupToList(cfg)

    def _removeSetup(self, row):
        """Remove a row of setup."""
        table = self._table
        name = table.item(row, 0).text()

        self._meta.remove_snapshot(name)
        self._removeSetupFromList(name)

    def _renameSetup(self, row, new_name):
        """Rename a row of setup."""
        table = self._table
        name = table.item(row, 0).text()

        self._meta.rename_snapshot(name, new_name)

        del self._config[name]
        self._config[new_name] = row
        table.item(row, 0).setText(new_name)

    def _askSaveSetups(self):
        reply = QMessageBox.question(
            self, "Save setups",
            "Setups in the file will be lost. Continue?",
            QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self._saveAllSetups()

    def _saveAllSetups(self):
        """Save all setups to file."""
        table = self._table
        sorted_config = sorted(self._config.items(), key=lambda x: x[1])
        lst = []
        for name, row in sorted_config:
            # "description" was not saved in Redis when edited in the table
            lst.append((name, table.item(row, 2).text()))
        self._meta.dump_all_setups(lst)

    def _askLoadSetups(self):
        reply = QMessageBox.question(
            self, "Load setups",
            "Current snapshots will be overwritten in case of name conflict. "
            "Continue?",
            QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self._loadAllSetups()

    def _loadAllSetups(self):
        """Load setups from file."""
        cfg_list = self._meta.load_all_setups()

        for cfg in cfg_list:
            # 'self._meta.load_all_setups' has already
            # write all the setups into Redis.
            if cfg[0] in self._config:
                self._removeSetupFromList(cfg[0])

        for cfg in cfg_list:
            self._insertSetupToList(cfg)

        if self.LAST_SAVED not in self._config:
            self._takeSnapshot()

    def _takeSnapshot(self, row=0):
        """Take a snapshot of the current setup.

        :param int row: row number. If row != 0, setups in both
            "Last saved" and this row will be updated.
        """
        table = self._table
        last_save = self.LAST_SAVED

        # update "Last saved"
        cfg = self._meta.take_snapshot(last_save)
        if table.rowCount() == 0 or table.item(0, 0).text() != last_save:
            self._insertSetupToList(cfg, 0)
        else:
            # update timestamp
            table.item(0, 1).setText(cfg[1])

        # maybe update
        name = table.item(row, 0).text()
        if name != last_save:
            self._meta.copy_snapshot(last_save, name)
            # update timestamp
            table.item(row, 1).setText(cfg[1])

    def _resetToDefault(self):
        self._meta.load_snapshot(self.DEFAULT)
        self.load_metadata_sgn.emit()

    def onItemDoubleClicked(self, item):
        """Double-click the name to apply the setup."""
        if self._table.column(item) < 2:
            self._meta.load_snapshot(item.text())
            self.load_metadata_sgn.emit()

    def showContextMenu(self, pos):
        table = self._table
        item = table.itemAt(pos)
        if table.column(item) != 0:
            # show context menu only when right-clicking on the name
            return

        row = self._table.row(item)
        menu = QMenu()
        save_action = menu.addAction("Take snapshot")
        copy_action = menu.addAction("Copy")
        if row != 0:
            # The first one is always "Last saved" and is not allowed
            # to be deleted
            delete_action = menu.addAction("Delete")
            rename_action = menu.addAction("Rename")
        else:
            delete_action = object()
            rename_action = object()

        action = menu.exec_(self.mapToGlobal(pos))
        if action == copy_action:
            new_name, ok = QInputDialog.getText(
                self, "", "New name: ", QLineEdit.Normal, "")

            if not self._checkConfigName(new_name):
                return

            if ok:
                self._copySetup(row, new_name)
        elif action == save_action:
            self._takeSnapshot(row)
        elif action == delete_action:
            self._removeSetup(row)
        elif action == rename_action:
            new_name, ok = QInputDialog.getText(
                self, "", "New name: ", QLineEdit.Normal, "")

            if not self._checkConfigName(new_name):
                return

            if ok:
                self._renameSetup(row, new_name)

    def _checkConfigName(self, name):
        if name in self._config:
            logger.error(f"Setup '{name}' already exists!")
            return False

        if name in [self.DEFAULT, self.LAST_SAVED]:
            logger.error(
                f"'{name}' is not allowed for user-defined setup!")
            return False

        if name == '':
            return False

        return True
