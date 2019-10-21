"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import copy

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QModelIndex, Qt, pyqtSignal

from .base_ctrl_widgets import AbstractCtrlWidget
from ..mediator import Mediator
from ...database import (
    DATA_SOURCE_CATEGORIES, EXCLUSIVE_SOURCE_CATEGORIES,
    DATA_SOURCE_PROPERTIES, SourceItem
)
from ...config import config, DataSource


class DSPropertyDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        """Override."""
        cb = QtWidgets.QComboBox(parent)
        channels = index.sibling(index.row(), 0).data(Qt.DisplayRole).split(':')
        ctg = index.parent().data(Qt.DisplayRole)
        cb.addItems(DATA_SOURCE_PROPERTIES[ctg if len(channels) == 1
                                           else f"{ctg}:{channels[-1]}"])
        return cb

    def setEditorData(self, editor, index):
        """Override."""
        value = index.data(Qt.EditRole)
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class DataSourceTreeItem:
    """Item used in DataSourceTreeModel."""
    def __init__(self, data, exclusive=False, parent=None):
        self._children = []
        self._data = data
        self._parent = parent

        self._exclusive = exclusive

        self._rank = 0 if parent is None else parent.rank() + 1

        self._checked = False

    def child(self, number):
        """Return the child at an index position."""
        try:
            return self._children[number]
        except IndexError:
            return None

    def appendChild(self, item):
        """Append a child item."""
        if not isinstance(item, DataSourceTreeItem):
            raise TypeError(f"Child item must be a {self.__class__}")
        self._children.append(item)

    def childCount(self):
        """Return the total number of children."""
        return len(self._children)

    def row(self):
        """Return the index of child in its parents' list of children."""
        if self._parent is not None:
            return self._parent._children.index(self)

        return 0

    def columnCount(self):
        """Return the length of the item data."""
        return len(self._data)

    def data(self, column):
        """Get data by column index."""
        try:
            return self._data[column]
        except IndexError:
            return None

    def setData(self, value, column):
        if 0 <= column < len(self._data):
            self._data[column] = value

    def parent(self):
        return self._parent

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = checked

    def rank(self):
        return self._rank

    def isExclusive(self):
        return self._exclusive


class DataSourceTreeModel(QtCore.QAbstractItemModel):
    """Tree model interface for managing data sources."""

    device_toggled_sgn = pyqtSignal(object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._mediator = Mediator()

        self._root = DataSourceTreeItem(["Source name", "Property"])
        self.setupModelData()

        self.device_toggled_sgn.connect(self._mediator.onDataSourceToggled)

    def data(self, index, role=None):
        """Override."""
        if not index.isValid():
            return

        item = index.internalPointer()

        if role == Qt.DisplayRole:
            return item.data(index.column())

        if role == Qt.CheckStateRole and index.column() == 0 and item.rank() > 1:
            return Qt.Checked if item.isChecked() else Qt.Unchecked

        return

    def setData(self, index, value, role=None) -> bool:
        """Override."""
        if role == Qt.CheckStateRole or role == Qt.EditRole:
            item = index.internalPointer()
            if role == Qt.CheckStateRole:
                n_rows = index.model().rowCount(index.parent())
                if item.isExclusive() and not item.isChecked():
                    for i in range(n_rows):
                        if i != index.row():
                            item_sb = index.sibling(i, 0).internalPointer()
                            if item_sb.isChecked():
                                item_sb.setChecked(False)
                                self.dataChanged.emit(index.sibling(i, 0),
                                                      index.sibling(i, 0))
                                self.device_toggled_sgn.emit(
                                    SourceItem(item_sb.parent().data(0),
                                               item_sb.data(0),
                                               item_sb.data(1)),
                                    False)
                                break

                item.setChecked(value)
            else:
                old_ppt = item.data(1)
                item.setData(value, index.column())
                if index.column() == 1:
                    # remove registered item with the old property
                    self.device_toggled_sgn.emit(
                        SourceItem(item.parent().data(0), item.data(0), old_ppt),
                        False)

            self.device_toggled_sgn.emit(SourceItem(item.parent().data(0),
                                                    item.data(0),
                                                    item.data(1)),
                                         item.isChecked())
            return True
        return False

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags

        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable

        item = index.internalPointer()
        if item.rank() > 1:
            if index.column() == 0:
                flags |= Qt.ItemIsUserCheckable
            elif item.isChecked() and index.column() >= 1:
                flags |= Qt.ItemIsEditable

        return flags

    def headerData(self, section, orientation, role=None):
        """Override."""
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._root.data(section)

    def index(self, row, column, parent=None, *args, **kwargs) -> QModelIndex:
        """Override."""
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if parent is None or not parent.isValid():
            parent_item = self._root
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item is None:
            return QModelIndex()
        return self.createIndex(row, column, child_item)

    def parent(self, index=None) -> QModelIndex:
        """Override."""
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent()

        if parent_item == self._root:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def rowCount(self, parent=None, *args, **kwargs) -> int:
        """Override."""
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parent_item = self._root
        else:
            parent_item = parent.internalPointer()

        return parent_item.childCount()

    def columnCount(self, parent=None, *args, **kwargs) -> int:
        """Override."""
        if parent.isValid():
            return parent.internalPointer().columnCount()
        return self._root.columnCount()

    def setupModelData(self):
        """Setup the data for the whole tree."""
        topic = config["TOPIC"]
        det = config["DETECTOR"]

        sources = copy.deepcopy(DATA_SOURCE_CATEGORIES[topic])

        if det not in sources:
            raise KeyError(f"{det} is not installed in instrument {topic}")

        sources[det].update(config["SOURCE_NAME_BRIDGE"])
        sources[det].update(config["SOURCE_NAME_FILE"])
        sources[det] = sorted(sources[det])

        for ctg, srcs in sources.items():
            self._root.appendChild(DataSourceTreeItem([ctg, ""],
                                                      exclusive=False,
                                                      parent=self._root))

            last_child = self._root.child(-1)

            try:
                exclusive = ctg in EXCLUSIVE_SOURCE_CATEGORIES
                for src in srcs:
                    channels = src.split(':')
                    key = ctg if len(channels) == 1 else f"{ctg}:{channels[-1]}"
                    default_ppt = list(DATA_SOURCE_PROPERTIES[key].keys())[0]
                    last_child.appendChild(DataSourceTreeItem(
                        [src, default_ppt], exclusive, parent=last_child))
            except KeyError:
                pass


class DataSourceListModel(QtCore.QAbstractListModel):
    """List model interface for monitoring available sources."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self._devices = []

    def data(self, index, role=None):
        """Override."""
        if not index.isValid() or index.row() > len(self._devices):
            return None

        if role == Qt.DisplayRole:
            return self._devices[index.row()]

    def rowCount(self, parent=None, *args, **kwargs):
        """Override."""
        return len(self._devices)

    def setupModelData(self, device_list):
        if self._devices != device_list:
            self.beginResetModel()
            self._devices = device_list
            self.endResetModel()


class ConnectionCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the TCP connection."""

    def __init__(self, *args, **kwargs):
        super().__init__("Connection", *args, **kwargs)

        self._hostname_le = QtGui.QLineEdit()
        self._hostname_le.setMinimumWidth(150)
        self._port_le = QtGui.QLineEdit()
        self._port_le.setValidator(QtGui.QIntValidator(0, 65535))

        self._source_type_cb = QtGui.QComboBox()
        self._source_type_cb.addItem("run directory", DataSource.FILE)
        self._source_type_cb.addItem("ZeroMQ bridge", DataSource.BRIDGE)
        self._source_type_cb.setCurrentIndex(config['DEFAULT_SOURCE_TYPE'])
        self._current_source_type = None

        self._non_reconfigurable_widgets = [
            self._source_type_cb,
            self._hostname_le,
            self._port_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        AR = Qt.AlignRight

        src_layout = QtGui.QGridLayout()
        src_layout.addWidget(QtGui.QLabel("Data streamed from: "), 0, 0, AR)
        src_layout.addWidget(self._source_type_cb, 0, 1)
        src_layout.addWidget(QtGui.QLabel("Hostname: "), 1, 0, AR)
        src_layout.addWidget(self._hostname_le, 1, 1)
        src_layout.addWidget(QtGui.QLabel("Port: "), 2, 0, AR)
        src_layout.addWidget(self._port_le, 2, 1)

        layout.addLayout(src_layout)
        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._source_type_cb.currentIndexChanged.connect(
            lambda x: self.onSourceTypeChange(
                self._source_type_cb.itemData(x)))
        self._source_type_cb.currentIndexChanged.connect(
            lambda x: mediator.onSourceTypeChange(
                self._source_type_cb.itemData(x)))

        # Emit once to fill the QLineEdit
        self._source_type_cb.currentIndexChanged.emit(
            self._source_type_cb.currentIndex())

        # Note: use textChanged signal for non-reconfigurable QLineEdit
        self._hostname_le.textChanged.connect(self.onEndpointChange)
        self._port_le.textChanged.connect(self.onEndpointChange)

    def updateMetaData(self):
        self._source_type_cb.currentIndexChanged.emit(
            self._source_type_cb.currentIndex())

        self._hostname_le.textChanged.emit(self._hostname_le.text())
        self._port_le.textChanged.emit(self._port_le.text())

        return True

    @QtCore.pyqtSlot()
    def onEndpointChange(self):
        endpoint = f"tcp://{self._hostname_le.text()}:{self._port_le.text()}"
        self._mediator.onBridgeEndpointChange(endpoint)

    @QtCore.pyqtSlot(object)
    def onSourceTypeChange(self, source_type):
        if source_type == self._current_source_type:
            return
        self._current_source_type = source_type

        if source_type == DataSource.BRIDGE:
            hostname = config["SERVER_ADDR"]
            port = config["SERVER_PORT"]
        else:
            hostname = config["LOCAL_HOST"]
            port = config["LOCAL_PORT"]

        self._hostname_le.setText(hostname)
        self._port_le.setText(str(port))


class DataSourceWidget(QtWidgets.QWidget):
    """DataSourceWidget class.

    A widget container which holds ConnectionCtrlWidget, DeviceListWidget
    and DeviceTreeWidget.
    """

    SPLITTER_HANDLE_WIDTH = 9

    def __init__(self, parent=None):
        super().__init__(parent)

        self.connection_ctrl_widget = ConnectionCtrlWidget()

        self._tree_view = QtWidgets.QTreeView()
        self._tree_model = DataSourceTreeModel(self)
        self._tree_ppt_delegate = DSPropertyDelegate(self)
        self._tree_view.setModel(self._tree_model)
        self._tree_view.setItemDelegateForColumn(1, self._tree_ppt_delegate)

        self._list_container = QtWidgets.QTabWidget()
        self._list_view = QtWidgets.QListView()
        self._list_model = DataSourceListModel()
        self._list_view.setModel(self._list_model)

        self.initUI()

    def initUI(self):
        self._list_container.addTab(self._list_view, "Available sources")

        splitter = QtWidgets.QSplitter(Qt.Vertical)
        splitter.setHandleWidth(self.SPLITTER_HANDLE_WIDTH)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._tree_view)
        splitter.addWidget(self._list_container)

        self._tree_view.expandToDepth(1)
        self._tree_view.resizeColumnToContents(0)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.connection_ctrl_widget)
        layout.addWidget(splitter)
        self.setLayout(layout)

    def updateDeviceList(self, devices):
        self._list_model.setupModelData(devices)
