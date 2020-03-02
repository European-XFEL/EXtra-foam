"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QGridLayout, QLabel, QLineEdit, QPushButton
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartSliceLineEdit
from ..gui_helpers import create_icon_button
from ...ipc import CalConstantsPub


class CalibrationCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up calibration parameters."""

    gain_const_sgn = pyqtSignal()
    offset_const_sgn = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._correct_gain_cb = QCheckBox()
        self._correct_offset_cb = QCheckBox()

        self._load_gain_btn = QPushButton("Load constants from file")
        self._load_offset_btn = QPushButton("Load constants from file")

        self._gain_fp_le = QLineEdit()
        self._gain_fp_le.setEnabled(False)
        self._offset_fp_le = QLineEdit()
        self._offset_fp_le.setEnabled(False)

        self._remove_gain_btn = create_icon_button('remove.png', 20)
        self._remove_offset_btn = create_icon_button('remove.png', 20)

        self._gain_slicer_le = SmartSliceLineEdit(":")
        self._offset_slicer_le = SmartSliceLineEdit(":")
        if not self._pulse_resolved:
            self._gain_slicer_le.setEnabled(False)
            self._offset_slicer_le.setEnabled(False)

        self._dark_as_offset_cb = QCheckBox("Use dark as offset")
        self._dark_as_offset_cb.setChecked(True)
        self._record_dark_btn = create_icon_button('record.png', 20)
        self._record_dark_btn.setCheckable(True)
        self._remove_dark_btn = create_icon_button('remove.png', 20)

        self._pub = CalConstantsPub()

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(self._correct_gain_cb, 0, 0)
        layout.addWidget(QLabel("Gain correction"), 0, 1, AR)
        layout.addWidget(self._load_gain_btn, 0, 2)
        layout.addWidget(self._gain_fp_le, 0, 3)
        layout.addWidget(self._remove_gain_btn, 0, 5)
        layout.addWidget(QLabel("Memory cells: "), 0, 6, AR)
        layout.addWidget(self._gain_slicer_le, 0, 7)
        self._gain_slicer_le.setFixedWidth(100)

        layout.addWidget(self._correct_offset_cb, 1, 0)
        layout.addWidget(QLabel("Offset correction"), 1, 1, AR)
        layout.addWidget(self._load_offset_btn, 1, 2)
        layout.addWidget(self._offset_fp_le, 1, 3)
        layout.addWidget(self._remove_offset_btn, 1, 5)
        layout.addWidget(QLabel("Memory cells: "), 1, 6, AR)
        layout.addWidget(self._offset_slicer_le, 1, 7)
        self._offset_slicer_le.setFixedWidth(100)

        layout.addWidget(self._dark_as_offset_cb, 2, 2)
        layout.addWidget(self._record_dark_btn, 2, 4)
        layout.addWidget(self._remove_dark_btn, 2, 5)

        layout.setColumnStretch(5, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self._correct_gain_cb.toggled.connect( mediator.onCalGainCorrection)
        self._correct_offset_cb.toggled.connect(mediator.onCalOffsetCorrection)

        self._load_gain_btn.clicked.connect(self._loadGainConst)
        self._load_offset_btn.clicked.connect(self._loadOffsetConst)

        self._remove_gain_btn.clicked.connect(self._removeGain)
        self._remove_offset_btn.clicked.connect(self._removeOffset)

        self._gain_slicer_le.value_changed_sgn.connect(
            mediator.onCalGainSlicerChange)
        self._offset_slicer_le.value_changed_sgn.connect(
            mediator.onCalOffsetSlicerChange)

        self._dark_as_offset_cb.toggled.connect(mediator.onCalDarkAsOffset)
        self._record_dark_btn.toggled.connect(mediator.onCalDarkRecording)
        self._record_dark_btn.toggled.emit(self._record_dark_btn.isChecked())
        self._remove_dark_btn.clicked.connect(mediator.onCalDarkRemove)

    def updateMetaData(self):
        """Override."""
        self._correct_gain_cb.toggled.emit(
            self._correct_gain_cb.isChecked())
        self._correct_offset_cb.toggled.emit(
            self._correct_offset_cb.isChecked())
        self._dark_as_offset_cb.toggled.emit(
            self._dark_as_offset_cb.isChecked())
        self._gain_slicer_le.returnPressed.emit()
        self._offset_slicer_le.returnPressed.emit()
        return True

    @pyqtSlot()
    def _loadGainConst(self):
        filepath = QFileDialog.getOpenFileName(
            caption="Load constants", directory=osp.expanduser("~"))[0]
        if filepath:
            self._gain_fp_le.setText(filepath)
            self._pub.set_gain(filepath)

    @pyqtSlot()
    def _loadOffsetConst(self):
        filepath = QFileDialog.getOpenFileName(
            caption="Load constants", directory=osp.expanduser("~"))[0]
        if filepath:
            self._offset_fp_le.setText(filepath)
            self._pub.set_offset(filepath)

    @pyqtSlot()
    def _removeGain(self):
        self._gain_fp_le.setText('')
        self._pub.remove_gain()

    @pyqtSlot()
    def _removeOffset(self):
        self._offset_fp_le.setText('')
        self._pub.remove_offset()

    def onDeactivated(self):
        if self._record_dark_btn.isChecked():
            self._record_dark_btn.setChecked(False)
