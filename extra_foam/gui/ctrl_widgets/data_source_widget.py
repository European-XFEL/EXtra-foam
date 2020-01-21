"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import (
    QAbstractItemModel, QAbstractListModel, QAbstractTableModel, QModelIndex,
    Qt, QTimer, pyqtSignal, pyqtSlot
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QComboBox, QGridLayout, QHeaderView, QLabel, QLineEdit, QListView,
    QSplitter, QStyledItemDelegate, QTableView, QTabWidget, QTreeView,
    QVBoxLayout, QWidget
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartSliceLineEdit
)
from ..gui_helpers import parse_boundary, parse_id, parse_slice
from ..mediator import Mediator
from ...database import MonProxy, SourceItem
from ...config import config, DataSource
from ...processes import list_foam_processes
from ...logger import logger


class _BaseItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def setEditorData(self, editor, index):
        """Override."""
        editor.setTextWithoutSignal(index.data(Qt.DisplayRole))

    def setModelData(self, editor, model, index):
        """Override."""
        model.setData(index, editor.text(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        """Override."""
        editor.setGeometry(option.rect)


class LineEditItemDelegate(_BaseItemDelegate):
    def createEditor(self, parent, option, index):
        """Override."""
        return SmartLineEdit(index.data(Qt.DisplayRole), parent)


class SliceItemDelegate(_BaseItemDelegate):
    def createEditor(self, parent, option, index):
        """Override."""
        return SmartSliceLineEdit(index.data(Qt.DisplayRole), parent)


class BoundaryItemDelegate(_BaseItemDelegate):
    def createEditor(self, parent, option, index):
        """Override."""
        return SmartBoundaryLineEdit(index.data(Qt.DisplayRole), parent)


class DataSourceTreeItem:
    """Item used in DataSourceItemModel."""
    def __init__(self, data, *, exclusive=False, parent=None):
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
            pass

    def appendChild(self, item):
        """Append a child item."""
        if not isinstance(item, DataSourceTreeItem):
            raise TypeError(f"Child item must be a {self.__class__}")
        if item not in self._children:
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
            pass

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


class DataSourceItemModel(QAbstractItemModel):
    """Tree model interface for managing data sources."""

    # checked, SourceItem
    source_item_toggled_sgn = pyqtSignal(bool, object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._mediator = Mediator()

        self._root = DataSourceTreeItem([
            "Source name", "Property", "Pulse slicer", "Value range"])
        self.setupModelData()

        self.source_item_toggled_sgn.connect(
            self._mediator.onSourceItemToggled)

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
        def _parse_slice(x):
            return str(parse_slice(x)) if x else x

        def _parse_boundary(x):
            return str(parse_boundary(x)) if x else x

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
                                self.source_item_toggled_sgn.emit(
                                    False,
                                    SourceItem('',
                                               item_sb.data(0),
                                               [],
                                               item_sb.data(1),
                                               '',
                                               ''))
                                break

                item.setChecked(value)
            else:  # role == Qt.EditRole
                old_ppt = item.data(1)
                item.setData(value, index.column())
                if index.column() >= 1:
                    # remove registered item with the old property
                    self.source_item_toggled_sgn.emit(
                        False,
                        SourceItem('', item.data(0), [], old_ppt, '', ''))

            ctg = item.parent().data(0)
            src_name = item.data(0)
            if ctg == config["DETECTOR"] \
                    and config["NUMBER_OF_MODULES"] > 1 and '*' in src_name:
                modules = [*range(config["NUMBER_OF_MODULES"])]
            else:
                modules = []

            self.source_item_toggled_sgn.emit(
                item.isChecked(),
                SourceItem(
                    ctg,
                    src_name,
                    modules,
                    item.data(1),
                    _parse_slice(item.data(2)),
                    _parse_boundary(item.data(3))))
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
            if item.isChecked():
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
        src_categories = dict()

        for ctg, srcs in config.pipeline_sources.items():
            ctg_item = DataSourceTreeItem(
                [ctg, "", "", "", ""], exclusive=False, parent=self._root)
            self._root.appendChild(ctg_item)
            src_categories[ctg] = ctg_item

            default_slicer = ':'
            # for 2D detectors we does not apply pixel-wise filtering for now
            default_v_range = '-inf, inf' if ctg not in config.detectors else ''
            for src, ppts in srcs.items():
                for ppt in ppts:
                    # only the main detector is exclusive since only it can
                    # have so many different names :-(
                    src_item = DataSourceTreeItem(
                        [src, ppt, default_slicer, default_v_range],
                        exclusive=ctg == config["DETECTOR"],
                        parent=ctg_item)
                    ctg_item.appendChild(src_item)

        for ctg, srcs in config.control_sources.items():
            if ctg not in src_categories:
                ctg_item = DataSourceTreeItem(
                    [ctg, "", "", "", ""], exclusive=False, parent=self._root)
                self._root.appendChild(ctg_item)
                src_categories[ctg] = ctg_item
            else:
                ctg_item = src_categories[ctg]

            # ctrl source does not support slice by default
            default_slicer = ''
            default_v_range = '-inf, inf'
            for src, ppts in srcs.items():
                for ppt in ppts:
                    ctg_item.appendChild(DataSourceTreeItem(
                        [src, ppt, default_slicer, default_v_range],
                        exclusive=False,
                        parent=ctg_item))

        # Add a couple of user defined instrument sources
        # Note: In order to meet the requirement to change sources on
        #       the fly, instead of allowing users to insert and delete
        #       items in the tree, we provided a couple of user defined
        #       instrument sources, which greatly simplifies te
        #       implementation and avoids subtle bugs. I believe there
        #       will be no need to modify the pipeline source on the fly.
        #       For example, if the device ID or the property of the main
        #       detector is not there, the user should close the app and
        #       modify the configuration file. And this should be usually
        #       done before the experiment.
        user_defined = config["SOURCE_USER_DEFINED_CATEGORY"]
        n_user_defined = 4
        assert user_defined not in src_categories
        ctg_item = DataSourceTreeItem([user_defined, "", "", "", ""],
                                      exclusive=False,
                                      parent=self._root)
        self._root.appendChild(ctg_item)
        for i in range(n_user_defined):
            ctg_item.appendChild(DataSourceTreeItem(
                [f"Device-ID-{i+1}", f"Property-{i+1}", "", '-inf, inf'],
                exclusive=False,
                parent=ctg_item))


class DataSourceListModel(QAbstractListModel):
    """List model interface for monitoring available sources."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self._sources = []

    def data(self, index, role=None):
        """Override."""
        if not index.isValid() or index.row() > len(self._sources):
            return

        if role == Qt.DisplayRole:
            return self._sources[index.row()]

    def rowCount(self, parent=None, *args, **kwargs):
        """Override."""
        return len(self._sources)

    def setupModelData(self, sources):
        if self._sources != sources:
            self.beginResetModel()
            self._sources = sources
            self.endResetModel()


class ProcessMonitorTableModel(QAbstractTableModel):
    """Table model interface for monitoring running processes."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self._processes = []
        self._headers = ["Process name", "Foam name", "Foam type", "pid", "status"]

    def headerData(self, section, orientation, role=None):
        """Override."""
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]

    def data(self, index, role=None):
        """Override."""
        row, col = index.row(), index.column()
        if not index.isValid() \
                or row > len(self._processes) or col > len(self._headers):
            return

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        if role == Qt.DisplayRole:
            return self._processes[row][col]

    def rowCount(self, parent=None, *args, **kwargs):
        """Override."""
        return len(self._processes)

    def columnCount(self, parent=None, *args, **kwargs):
        """Override."""
        return len(self._headers)

    def setupModelData(self, processes):
        self.beginResetModel()
        self._processes = processes
        self.endResetModel()


class ConnectionCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up the TCP connection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hostname_le = QLineEdit()
        self._hostname_le.setMinimumWidth(150)
        self._port_le = QLineEdit()
        self._port_le.setValidator(QIntValidator(0, 65535))

        self._source_type_cb = QComboBox()
        self._source_type_cb.addItem("run directory", DataSource.FILE)
        self._source_type_cb.addItem("ZeroMQ bridge", DataSource.BRIDGE)
        self._source_type_cb.setCurrentIndex(
            config['SOURCE_DEFAULT_TYPE'])
        self._current_source_type = None

        self._non_reconfigurable_widgets = [
            self._source_type_cb,
            self._hostname_le,
            self._port_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QVBoxLayout()
        AR = Qt.AlignRight

        src_layout = QGridLayout()
        src_layout.addWidget(QLabel("Data streamed from: "), 0, 0, AR)
        src_layout.addWidget(self._source_type_cb, 0, 1)
        src_layout.addWidget(QLabel("Hostname: "), 1, 0, AR)
        src_layout.addWidget(self._hostname_le, 1, 1)
        src_layout.addWidget(QLabel("Port: "), 2, 0, AR)
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

    @pyqtSlot()
    def onEndpointChange(self):
        endpoint = f"tcp://{self._hostname_le.text()}:{self._port_le.text()}"
        self._mediator.onBridgeEndpointChange(endpoint)

    @pyqtSlot(object)
    def onSourceTypeChange(self, source_type):
        if source_type == self._current_source_type:
            return
        self._current_source_type = source_type

        if source_type == DataSource.BRIDGE:
            hostname = config["BRIDGE_ADDR"]
            port = config["BRIDGE_PORT"]
        else:
            hostname = config["LOCAL_ADDR"]
            port = config["LOCAL_PORT"]

        self._hostname_le.setText(hostname)
        self._port_le.setText(str(port))


class DataSourceWidget(QWidget):
    """DataSourceWidget class.

    A widget container which holds ConnectionCtrlWidget, DeviceListWidget
    and DeviceTreeWidget.
    """

    SPLITTER_HANDLE_WIDTH = 9

    def __init__(self, parent):
        super().__init__(parent)

        self._connection_ctrl_widget = parent.createCtrlWidget(
            ConnectionCtrlWidget)

        self._tree_view = QTreeView()
        self._tree_model = DataSourceItemModel(self)
        self._tree_device_delegate = LineEditItemDelegate(self)
        self._tree_ppt_delegate = LineEditItemDelegate(self)
        self._tree_slicer_delegate = SliceItemDelegate(self)
        self._tree_boundary_delegate = BoundaryItemDelegate(self)
        self._tree_view.setModel(self._tree_model)
        self._tree_view.setItemDelegateForColumn(0, self._tree_device_delegate)
        self._tree_view.setItemDelegateForColumn(1, self._tree_ppt_delegate)
        self._tree_view.setItemDelegateForColumn(2, self._tree_slicer_delegate)
        self._tree_view.setItemDelegateForColumn(3, self._tree_boundary_delegate)

        self._monitor_tb = QTabWidget()
        self._avail_src_view = QListView()
        self._avail_src_model = DataSourceListModel()
        self._avail_src_view.setModel(self._avail_src_model)
        self._process_mon_view = QTableView()
        self._process_mon_model = ProcessMonitorTableModel()
        self._process_mon_view.setModel(self._process_mon_model)
        self._process_mon_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        self.initUI()

        self._mon = MonProxy()

        self._avail_src_timer = QTimer()
        self._avail_src_timer.timeout.connect(self.updateSourceList)
        self._avail_src_timer.start(config["SOURCE_AVAIL_UPDATE_TIMER"])

        self._process_mon_timer = QTimer()
        self._process_mon_timer.timeout.connect(self.updateProcessInfo)
        self._process_mon_timer.start(config["PROCESS_MONITOR_UPDATE_TIMER"])

    def initUI(self):
        self._monitor_tb.setTabPosition(QTabWidget.TabPosition.South)
        self._monitor_tb.addTab(self._avail_src_view, "Available sources")
        self._monitor_tb.addTab(self._process_mon_view, "Process monitor")

        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(self.SPLITTER_HANDLE_WIDTH)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._tree_view)
        splitter.addWidget(self._monitor_tb)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self._tree_view.expandToDepth(1)
        self._tree_view.resizeColumnToContents(0)
        self._tree_view.resizeColumnToContents(1)

        layout = QVBoxLayout()
        layout.addWidget(self._connection_ctrl_widget)
        layout.addWidget(splitter)
        self.setLayout(layout)

    def updateSourceList(self):
        available_sources = self._mon.get_available_sources()
        if available_sources is not None:  # for unittest
            self._avail_src_model.setupModelData(list(available_sources.keys()))

    def updateProcessInfo(self):
        info = []
        for p in list_foam_processes():
            info.append(list(p))
        self._process_mon_model.setupModelData(info)
