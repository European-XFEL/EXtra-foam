"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

DarkRunWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from PyQt5 import QtCore, QtGui

from .base_window import AbstractWindow
from ..plot_widgets import ImageView
from ..misc_widgets import Colors
from ...config import config


class _DarkRunActionWidget(QtGui.QWidget):
    """Ctrl widget for dark run in the action bar."""

    class DescriptionLabel(QtGui.QLabel):
        def __init__(self, text, parent=None):
            super().__init__(text, parent=parent)
            font = QtGui.QFont()
            font.setBold(True)
            self.setFont(font)

    class NumberLabel(QtGui.QLabel):
        def __init__(self, text, parent=None):
            super().__init__(text, parent=parent)
            font = QtGui.QFont('times')
            font.setBold(True)
            font.setPointSize(16)
            self.setFont(font)
            self.setStyleSheet(f"color: rgb{Colors().p[:3]};")

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.proc_data_cb = QtGui.QCheckBox("Process while recording")
        self.proc_data_cb.setChecked(True)

        self.dark_train_count_lb = self.NumberLabel("")
        self.updateCount(0)

        self.initUI()

        self.setFixedSize(self.minimumSizeHint())

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.proc_data_cb)
        layout.addStretch(2)
        layout.addWidget(self.DescriptionLabel("Dark train count: "))
        layout.addWidget(self.dark_train_count_lb)

        self.setLayout(layout)
        self.layout().setContentsMargins(2, 1, 2, 1)

    def updateCount(self, count):
        self.dark_train_count_lb.setText(f"{count:0{6}d}")


class DarkRunWindow(AbstractWindow):
    """DarkRunWindow class.

    This is the second Main GUI which focuses on manipulating the image
    , e.g. selecting ROI, masking, normalization, for different
    data analysis scenarios.
    """

    title = "Dark run"

    _root_dir = osp.dirname(osp.abspath(__file__))

    __instance = None

    @classmethod
    def reset(cls):
        cls.__instance = None

    def __new__(cls, *args, **kwargs):
        """Create a singleton."""
        if cls.__instance is None:
            instance = super().__new__(cls, *args, **kwargs)
            instance._is_initialized = False
            cls.__instance = instance
            return instance

        instance = cls.__instance
        parent = instance.parent()
        if parent is not None:
            parent.registerWindow(instance)

        instance.show()
        instance.activateWindow()
        return instance

    def __init__(self, *args, **kwargs):
        if self._is_initialized:
            return
        super().__init__(*args, **kwargs)

        self._tool_bar = self.addToolBar("Control")

        self._record_at = self._addAction("Record dark", "record.png")
        self._record_at.setCheckable(True)

        self._remove_at = self._addAction("Remove dark", "delete.png")

        self._tool_bar_ctrl = self.addToolBar("ctrl")
        self._ctrl_action = _DarkRunActionWidget(self)
        self._ctrl_action_at = QtGui.QWidgetAction(self._tool_bar_ctrl)
        self._ctrl_action_at.setDefaultWidget(self._ctrl_action)
        self._tool_bar_ctrl.addAction(self._ctrl_action_at)

        self._image_view = ImageView(hide_axis=False, parent=self)

        self.initUI()
        self.initConnections()
        self.updateMetaData()

        self.resize(800, 600)
        self.update()

        self._is_initialized = True

    def initUI(self):
        """Override."""
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._image_view)

        self._cw.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def initConnections(self):
        """Override."""
        self._record_at.toggled.connect(self._mediator.onRdStateChange)
        self._record_at.toggled.connect(self.onRecordingToggled)

        self._remove_at.triggered.connect(self._mediator.onRdResetDark)

        self._ctrl_action.proc_data_cb.toggled.connect(
            self._mediator.onRdProcessStateChange)

    def update(self):
        """Update widgets.

        This method is called by the main GUI.
        """
        data = self._data.get()

        if data is None or data.image.dark_mean is None:
            return

        self._image_view.setImage(
            data.image.dark_mean, auto_levels=(not self._is_initialized))
        self._ctrl_action.updateCount(data.image.dark_count)

        if not self._is_initialized:
            self._is_initialized = True

    def updateMetaData(self):
        """Override."""
        self._ctrl_action.proc_data_cb.toggled.emit(
            self._ctrl_action.proc_data_cb.isChecked())
        return True

    @QtCore.pyqtSlot(bool)
    def onRecordingToggled(self, state):
        if state:
            self._ctrl_action.proc_data_cb.setEnabled(False)
        else:
            self._ctrl_action.proc_data_cb.setEnabled(True)

    def _addAction(self, description, filename):
        icon = QtGui.QIcon(osp.join(self._root_dir, "../icons/" + filename))
        action = QtGui.QAction(icon, description, self)
        self._tool_bar.addAction(action)
        return action
