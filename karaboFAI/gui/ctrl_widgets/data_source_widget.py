"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import copy

from PyQt5.QtCore import (
    QAbstractItemModel, QAbstractListModel, QModelIndex, Qt, QTimer,
    pyqtSignal, pyqtSlot
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QComboBox, QGridLayout, QLabel, QLineEdit, QListView, QSplitter,
    QStyledItemDelegate, QTabWidget, QTreeView, QVBoxLayout, QWidget
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartSliceLineEdit, SmartBoundaryLineEdit
from ..gui_helpers import parse_boundary, parse_slice
from ..mediator import Mediator
from ...database import (
    DATA_SOURCE_CATEGORIES, EXCLUSIVE_SOURCE_CATEGORIES,
    DATA_SOURCE_PROPERTIES, DATA_SOURCE_SLICER, DATA_SOURCE_VRANGE,
    MonProxy, SourceItem
)
from ...config import config, DataSource


class DSPropertyDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        """Override."""
        cb = QComboBox(parent)
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


class DSSlicerDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        """Override."""
        channels = index.sibling(index.row(), 0).data(Qt.DisplayRole).split(':')
        ctg = index.parent().data(Qt.DisplayRole)
        if len(channels) > 1 or ctg in config.detectors:
            # pipeline data
            le = SmartSliceLineEdit(DATA_SOURCE_SLICER['default'], parent)
            return le

    def setEditorData(self, editor, index):
        """Override."""
        value = index.data(Qt.DisplayRole)
        editor.setTextWithoutSignal(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.text(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class DSVrangeDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        """Override."""
        channels = index.sibling(index.row(), 0).data(Qt.DisplayRole).split(':')
        ctg = index.parent().data(Qt.DisplayRole)
        # TODO: add more supports
        if ctg == 'XGM' and len(channels) > 1:
            le = SmartBoundaryLineEdit(DATA_SOURCE_VRANGE['XGM'], parent)
            return le

    def setEditorData(self, editor, index):
        """Override."""
        value = index.data(Qt.DisplayRole)
        editor.setTextWithoutSignal(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.text(), Qt.EditRole)

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


class DataSourceTreeModel(QAbstractItemModel):
    """Tree model interface for managing data sources."""

    device_toggled_sgn = pyqtSignal(object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._mediator = Mediator()

        self._root = DataSourceTreeItem([
            "Source name", "Property", "Pulse slicer", "Value range"])
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
        def _parse_slice(x):
            return slice(*parse_slice(x)) if x else None

        def _parse_boundary(x):
            return parse_boundary(x) if x else None

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
                                    SourceItem(
                                        item_sb.parent().data(0),
                                        item_sb.data(0),
                                        item_sb.data(1),
                                        _parse_slice(item_sb.data(2)),
                                        _parse_boundary(item_sb.data(3)),
                                    ),
                                    False)
                                break

                item.setChecked(value)
            else:  # role == Qt.EditRole
                old_ppt = item.data(1)
                old_slicer = item.data(2)
                old_vrange = item.data(3)
                item.setData(value, index.column())
                if index.column() >= 1:
                    # remove registered item with the old property
                    self.device_toggled_sgn.emit(
                        SourceItem(
                            item.parent().data(0),
                            item.data(0),
                            old_ppt,
                            _parse_slice(old_slicer),
                            _parse_boundary(old_vrange)
                        ),
                        False)

            self.device_toggled_sgn.emit(
                SourceItem(
                    item.parent().data(0),
                    item.data(0),
                    item.data(1),
                    _parse_slice(item.data(2)),
                    _parse_boundary(item.data(3)),
                ),
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
            self._root.appendChild(DataSourceTreeItem([ctg, "", "", ""],
                                                      exclusive=False,
                                                      parent=self._root))

            last_child = self._root.child(-1)

            try:
                exclusive = ctg in EXCLUSIVE_SOURCE_CATEGORIES
                for src in srcs:
                    channels = src.split(':')
                    key = ctg if len(channels) == 1 else f"{ctg}:{channels[-1]}"
                    default_ppt = list(DATA_SOURCE_PROPERTIES[key].keys())[0]
                    if len(channels) > 1 or ctg in config.detectors:
                        default_slicer = DATA_SOURCE_SLICER['default']
                    else:
                        default_slicer = DATA_SOURCE_SLICER['not_supported']

                    # TODO: add more supports
                    if ctg == 'XGM' and len(channels) > 1:
                        default_v_range = DATA_SOURCE_VRANGE['XGM']
                    else:
                        default_v_range = DATA_SOURCE_VRANGE['not_supported']

                    last_child.appendChild(DataSourceTreeItem(
                        [src, default_ppt, default_slicer, default_v_range],
                        exclusive,
                        parent=last_child))
            except KeyError as e:
                # TODO: log the error information!
                pass


class DataSourceListModel(QAbstractListModel):
    """List model interface for monitoring available sources."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self._sources = []

    def data(self, index, role=None):
        """Override."""
        if not index.isValid() or index.row() > len(self._sources):
            return None

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
            hostname = config["SERVER_ADDR"]
            port = config["SERVER_PORT"]
        else:
            hostname = config["LOCAL_HOST"]
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
        self._tree_model = DataSourceTreeModel(self)
        self._tree_ppt_delegate = DSPropertyDelegate(self)
        self._tree_slicer_delegate = DSSlicerDelegate(self)
        self._tree_range_delegate = DSVrangeDelegate(self)
        self._tree_view.setModel(self._tree_model)
        self._tree_view.setItemDelegateForColumn(1, self._tree_ppt_delegate)
        self._tree_view.setItemDelegateForColumn(2, self._tree_slicer_delegate)
        self._tree_view.setItemDelegateForColumn(3, self._tree_range_delegate)

        self._list_container = QTabWidget()
        self._list_view = QListView()
        self._list_model = DataSourceListModel()
        self._list_view.setModel(self._list_model)

        self.initUI()

        self._mon = MonProxy()

        self._timer = QTimer()
        self._timer.timeout.connect(self.updateSourceList)
        self._timer.start(config["SOURCES_UPDATE_INTERVAL"])

    def initUI(self):
        self._list_container.setTabPosition(QTabWidget.TabPosition.South)
        self._list_container.addTab(self._list_view, "Available sources")

        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(self.SPLITTER_HANDLE_WIDTH)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._tree_view)
        splitter.addWidget(self._list_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self._tree_view.expandToDepth(1)
        self._tree_view.resizeColumnToContents(0)

        layout = QVBoxLayout()
        layout.addWidget(self._connection_ctrl_widget)
        layout.addWidget(splitter)
        self.setLayout(layout)

    def updateSourceList(self):
        available_sources = self._mon.get_available_sources()
        self._list_model.setupModelData(list(available_sources.keys()))
