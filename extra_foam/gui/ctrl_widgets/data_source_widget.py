"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import (
    QAbstractItemModel, QAbstractListModel, QAbstractTableModel, QModelIndex,
    Qt, QTimer, pyqtSignal
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QComboBox, QHeaderView, QLineEdit, QListView,
    QSplitter, QStyledItemDelegate, QTableView, QTabWidget, QTreeView,
    QVBoxLayout
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartSliceLineEdit
)
from ..gui_helpers import parse_boundary, parse_slice
from ..mediator import Mediator
from ...database import MonProxy
from ...config import config, DataSource
from ...geometries import module_indices
from ...processes import list_foam_processes
from ...logger import logger


class _BaseSmartEditItemDelegate(QStyledItemDelegate):
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


class LineEditItemDelegate(_BaseSmartEditItemDelegate):
    def __init__(self, parent=None, *, validator=None):
        super().__init__(parent=parent)
        self._validator = validator

    def createEditor(self, parent, option, index):
        """Override."""
        value = index.data(Qt.DisplayRole)
        if not value:
            return

        widget = SmartLineEdit(value, parent)
        if self._validator is not None:
            widget.setValidator(self._validator)
        return widget


class SliceItemDelegate(_BaseSmartEditItemDelegate):
    def createEditor(self, parent, option, index):
        """Override."""
        value = index.data(Qt.DisplayRole)
        if not value:
            return
        return SmartSliceLineEdit(value, parent)


class BoundaryItemDelegate(_BaseSmartEditItemDelegate):
    def createEditor(self, parent, option, index):
        """Override."""
        value = index.data(Qt.DisplayRole)
        if not value:
            return
        return SmartBoundaryLineEdit(value, parent)


class LineEditItemDelegateN(QStyledItemDelegate):
    """The non-smart one."""
    def __init__(self, parent=None, *, validator=None):
        super().__init__(parent)
        self._validator = validator

    def setEditorData(self, editor, index):
        """Override."""
        editor.setText(index.data(Qt.DisplayRole))

    def setModelData(self, editor, model, index):
        """Override."""
        model.setData(index, editor.text(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        """Override."""
        editor.setGeometry(option.rect)

    def createEditor(self, parent, option, index):
        """Override."""
        value = index.data(Qt.DisplayRole)
        if not value:
            return

        widget = QLineEdit(value, parent)
        if self._validator is not None:
            widget.setValidator(self._validator)
        return widget


class ComboBoxDelegate(QStyledItemDelegate):
    def __init__(self, items, parent=None):
        super().__init__(parent)

        self._items = items

    def createEditor(self, parent, option, index):
        """Override."""
        cb = QComboBox(parent)
        cb.addItems(self._items.keys())
        cb.setCurrentText(index.model().data(index, Qt.DisplayRole))
        return cb

    def setEditorData(self, editor, index):
        """Override."""
        value = index.data(Qt.EditRole)
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        """Override."""
        model.setData(index, editor.currentText(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        """Override."""
        editor.setGeometry(option.rect)


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

    # True/False, tuple/str
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
        if role == Qt.CheckStateRole or role == Qt.EditRole:
            item = index.internalPointer()
            if role == Qt.CheckStateRole:
                n_rows = index.model().rowCount(index.parent())
                if item.isExclusive() and not item.isChecked():
                    for i in range(n_rows):
                        if i != index.row():
                            item_sb = index.sibling(i, 0).internalPointer()
                            if item_sb.isExclusive() and item_sb.isChecked():
                                item_sb.setChecked(False)
                                self.dataChanged.emit(index.siblingAtRow(i),
                                                      index.siblingAtRow(i))
                                self.source_item_toggled_sgn.emit(
                                    False,
                                    f'{item_sb.data(0)} {item_sb.data(1)}')
                                break

                item.setChecked(value)
            else:  # role == Qt.EditRole
                old_src_name = item.data(0)
                old_ppt = item.data(1)
                item.setData(value, index.column())
                # remove registered item with the old device ID and property
                self.source_item_toggled_sgn.emit(
                    False, f'{old_src_name} {old_ppt}')

            main_det = config["DETECTOR"]
            ctg = item.parent().data(0)
            src_name = item.data(0)
            n_modules = config["NUMBER_OF_MODULES"]
            if ctg == main_det and n_modules > 1 and '*' in src_name:
                modules = module_indices(n_modules, detector=main_det)
            else:
                modules = []

            slicer = item.data(2)
            vrange = item.data(3)
            if item.isChecked():
                self.source_item_toggled_sgn.emit(
                    item.isChecked(),
                    (ctg, src_name, str(modules), item.data(1),
                     str(parse_slice(slicer)) if slicer else '',
                     str(parse_boundary(vrange)) if vrange else '',
                     item.data(4))
                )
            else:
                self.source_item_toggled_sgn.emit(
                    False, f'{src_name} {item.data(1)}')

            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags

        flags = Qt.ItemIsEnabled

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

            # train-resolved detectors do not need slicer
            default_slicer = ':'
            if ctg in config.detectors and not config["PULSE_RESOLVED"]:
                default_slicer = ''
            # for 2D detectors we does not apply pixel-wise filtering for now
            default_v_range = '-inf, inf' if ctg not in config.detectors else ''
            for src, ppts in srcs.items():
                for ppt in ppts:
                    # For now, all pipeline data are exclusive
                    src_item = DataSourceTreeItem(
                        [src, ppt, default_slicer, default_v_range, 1],
                        exclusive=True,
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
                        [src, ppt, default_slicer, default_v_range, 0],
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
                [f"Device-ID-{i+1}", f"Property-{i+1}", "", '-inf, inf', 0],
                exclusive=False,
                parent=ctg_item))


class DataSourceListModel(QAbstractListModel):
    """List model interface for monitoring available sources."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self._sources = []

    def data(self, index, role=None):
        """Override."""
        if not index.isValid():
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

    def flags(self, index):
        """Override."""
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled

    def data(self, index, role=None):
        """Override."""
        if not index.isValid():
            return

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        row, col = index.row(), index.column()
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


class ConnectionTableModel(QAbstractTableModel):
    """Table model interface for connections management."""

    def __init__(self, source_types, parent=None):
        super().__init__(parent=parent)

        self._source_types = source_types
        self._connections = None
        self._headers = ("Name", "Source type", "IP address", "Port")

        connections = [[True, config["DETECTOR"], "", "", ""]]
        for con in config.appendix_streamers:
            connections.append([False,
                                con.name,
                                self._getSourceTypeString(con.type),
                                con.address,
                                str(con.port)])
        self.setupModelData(connections)
        # trigger initialization for addr and port
        self.setData(self.index(0, 1),
                     self._getSourceTypeString(config['SOURCE_DEFAULT_TYPE']),
                     role=Qt.EditRole)

    def _getSourceTypeString(self, src_type):
        for k, v in self._source_types.items():
            if v == src_type:
                return k
        raise ValueError(f"Unknown source type: {repr(src_type)}")

    def headerData(self, section, orientation, role=None):
        """Override."""
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]

    def flags(self, index):
        """Override."""
        if not index.isValid():
            return Qt.NoItemFlags

        flags = Qt.ItemIsEnabled
        row, col = index.row(), index.column()
        if col == 0:
            if row == 0:
                # Check state of the main detector is not allowed to be changed
                return flags
            return flags | Qt.ItemIsUserCheckable

        if self._connections[row][0]:
            return flags | Qt.ItemIsEditable

        return Qt.NoItemFlags

    def setData(self, index, value, role=None):
        """Override."""
        if not index.isValid():
            return False

        if role == Qt.CheckStateRole or role == Qt.EditRole:
            row, col = index.row(), index.column()
            if role == Qt.CheckStateRole:
                self._connections[row][col] = value
                self.dataChanged.emit(index.siblingAtColumn(0),
                                      index.siblingAtColumn(1))
            else:
                self._connections[row][col+1] = value
                if role == Qt.EditRole and row == 0 and col == 1:
                    # only for the main detector
                    assert len(self._source_types) == 2
                    if self._source_types[value] == DataSource.BRIDGE:
                        addr = config["BRIDGE_ADDR"]
                        port = config["BRIDGE_PORT"]
                    else:
                        addr = config["LOCAL_ADDR"]
                        port = config["LOCAL_PORT"]
                    self._connections[row][col+2] = addr
                    self._connections[row][col+3] = str(port)

                self.dataChanged.emit(index, index)
            return True
        return False

    def data(self, index, role=None):
        """Override."""
        if not index.isValid():
            return

        row, col = index.row(), index.column()

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        if role == Qt.CheckStateRole and col == 0:
            return Qt.Checked if self._connections[row][0] else Qt.Unchecked

        if role == Qt.DisplayRole:
            return self._connections[row][col+1]

    def rowCount(self, parent=None, *args, **kwargs):
        """Override."""
        return len(self._connections)

    def columnCount(self, parent=None, *args, **kwargs):
        """Override."""
        return len(self._headers)

    def setupModelData(self, connections):
        self.beginResetModel()
        self._connections = connections
        self.endResetModel()

    def connections(self):
        endpoints = []
        src_types = set()
        cons = dict()
        for checked, _, src_type, addr, port in self._connections:
            if checked:
                src_type = int(self._source_types[src_type])
                endpoint = f"tcp://{addr}:{port}"
                if endpoint in endpoints:
                    raise ValueError(f"Duplicated endpoint: {endpoint}")
                endpoints.append(endpoint)
                src_types.add(src_type)
                if len(src_types) > 1:
                    raise ValueError(f"All endpoints must have the same "
                                     f"source type!")
                cons[endpoint] = src_type
        return cons


class DataSourceWidget(_AbstractCtrlWidget):
    """DataSourceWidget class.

    Widgets provide data source management and monitoring.
    """

    _source_types = {
        "Run directory": DataSource.FILE,
        "ZeroMQ bridge": DataSource.BRIDGE,
    }

    SPLITTER_HANDLE_WIDTH = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._con_view = QTableView()
        self._con_model = ConnectionTableModel(self._source_types)
        self._con_view.setModel(self._con_model)
        self._con_src_type_delegate = ComboBoxDelegate(self._source_types)
        self._con_addr_delegate = LineEditItemDelegateN(self)
        self._con_port_delegate = LineEditItemDelegateN(
            self, validator=QIntValidator(0, 65535))
        self._con_view.setItemDelegateForColumn(1, self._con_src_type_delegate)
        self._con_view.setItemDelegateForColumn(2, self._con_addr_delegate)
        self._con_view.setItemDelegateForColumn(3, self._con_port_delegate)

        self._src_view = QTreeView()
        self._src_tree_model = DataSourceItemModel(self)
        self._src_device_delegate = LineEditItemDelegate(self)
        self._src_ppt_delegate = LineEditItemDelegate(self)
        self._src_slicer_delegate = SliceItemDelegate(self)
        self._src_boundary_delegate = BoundaryItemDelegate(self)
        self._src_view.setModel(self._src_tree_model)
        self._src_view.setItemDelegateForColumn(0, self._src_device_delegate)
        self._src_view.setItemDelegateForColumn(1, self._src_ppt_delegate)
        self._src_view.setItemDelegateForColumn(2, self._src_slicer_delegate)
        self._src_view.setItemDelegateForColumn(3, self._src_boundary_delegate)

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
        self.initConnections()

        self._non_reconfigurable_widgets = [
            self._con_view,
        ]

        self._mon = MonProxy()

        self._avail_src_timer = QTimer()
        self._avail_src_timer.timeout.connect(self.updateSourceList)
        self._avail_src_timer.start(config["SOURCE_AVAIL_UPDATE_TIMER"])

        self._process_mon_timer = QTimer()
        self._process_mon_timer.timeout.connect(self.updateProcessInfo)
        self._process_mon_timer.start(config["PROCESS_MONITOR_UPDATE_TIMER"])

    def initUI(self):
        """Override."""
        self._monitor_tb.setTabPosition(QTabWidget.TabPosition.South)
        self._monitor_tb.addTab(self._avail_src_view, "Available sources")
        self._monitor_tb.addTab(self._process_mon_view, "Process monitor")

        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(self.SPLITTER_HANDLE_WIDTH)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._con_view)
        splitter.addWidget(self._src_view)
        splitter.addWidget(self._monitor_tb)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        h = splitter.sizeHint().height()
        splitter.setSizes([0.1 * h, 0.6 * h, 0.3 * h])

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

        self._con_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        self._src_view.expandToDepth(1)
        self._src_view.resizeColumnToContents(0)
        self._src_view.resizeColumnToContents(1)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        mediator.file_stream_initialized_sgn.connect(self.updateMetaData)

    def updateMetaData(self):
        """Override."""
        try:
            cons = self._con_model.connections()
            self._mediator.onBridgeConnectionsChange(cons)
        except ValueError as e:
            logger.error(e)
            return False
        return True

    def updateSourceList(self):
        available_sources = self._mon.get_available_sources()
        if available_sources is not None:  # for unittest
            self._avail_src_model.setupModelData(list(available_sources.keys()))

    def updateProcessInfo(self):
        info = []
        for p in list_foam_processes():
            info.append(list(p))
        self._process_mon_model.setupModelData(info)
