"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QGridLayout, QLabel, QLineEdit, QPushButton
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartSliceLineEdit
from ..gui_helpers import create_icon_button, parse_slice_inv
from ...database import Metadata as mt


class CalibrationCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up calibration parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._correct_gain_cb = QCheckBox()
        self._correct_offset_cb = QCheckBox()

        self.load_gain_btn = QPushButton("Load gain")
        self.load_offset_btn = QPushButton("Load offset")

        self.gain_fp_le = QLineEdit()
        self.gain_fp_le.setEnabled(False)
        self.offset_fp_le = QLineEdit()
        self.offset_fp_le.setEnabled(False)

        self.remove_gain_btn = create_icon_button('remove.png', 20)
        self.remove_offset_btn = create_icon_button('remove.png', 20)

        self._gain_slicer_le = SmartSliceLineEdit(":")
        self._offset_slicer_le = SmartSliceLineEdit(":")
        if not self._pulse_resolved:
            self._gain_slicer_le.setEnabled(False)
            self._offset_slicer_le.setEnabled(False)

        self._dark_as_offset_cb = QCheckBox("Use dark as offset")
        self._dark_as_offset_cb.setChecked(True)
        self.record_dark_btn = create_icon_button('record.png', 20)
        self.record_dark_btn.setCheckable(True)
        self._remove_dark_btn = create_icon_button('remove.png', 20)

        self._non_reconfigurable_widgets = [
            self.load_gain_btn,
            self.load_offset_btn,
        ]

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(self._correct_gain_cb, 0, 0)
        layout.addWidget(QLabel("Gain correction"), 0, 1, AR)
        layout.addWidget(self.load_gain_btn, 0, 2)
        layout.addWidget(self.gain_fp_le, 0, 3)
        layout.addWidget(self.remove_gain_btn, 0, 5)
        layout.addWidget(QLabel("Memory cells: "), 0, 6, AR)
        layout.addWidget(self._gain_slicer_le, 0, 7)
        self._gain_slicer_le.setFixedWidth(100)

        layout.addWidget(self._correct_offset_cb, 1, 0)
        layout.addWidget(QLabel("Offset correction"), 1, 1, AR)
        layout.addWidget(self.load_offset_btn, 1, 2)
        layout.addWidget(self.offset_fp_le, 1, 3)
        layout.addWidget(self.remove_offset_btn, 1, 5)
        layout.addWidget(QLabel("Memory cells: "), 1, 6, AR)
        layout.addWidget(self._offset_slicer_le, 1, 7)
        self._offset_slicer_le.setFixedWidth(100)

        layout.addWidget(self._dark_as_offset_cb, 2, 2)
        layout.addWidget(self.record_dark_btn, 2, 4)
        layout.addWidget(self._remove_dark_btn, 2, 5)

        layout.setColumnStretch(5, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self._correct_gain_cb.toggled.connect(mediator.onCalGainCorrection)
        self._correct_offset_cb.toggled.connect(mediator.onCalOffsetCorrection)

        self._gain_slicer_le.value_changed_sgn.connect(
            mediator.onCalGainSlicerChange)
        self._offset_slicer_le.value_changed_sgn.connect(
            mediator.onCalOffsetSlicerChange)

        self._dark_as_offset_cb.toggled.connect(mediator.onCalDarkAsOffset)
        self.record_dark_btn.toggled.connect(mediator.onCalDarkRecording)
        self.record_dark_btn.toggled.emit(self.record_dark_btn.isChecked())
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

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.IMAGE_PROC)

        self._correct_gain_cb.setChecked(cfg["correct_gain"] == 'True')
        self._correct_offset_cb.setChecked(cfg["correct_offset"] == 'True')
        self._dark_as_offset_cb.setChecked(cfg["dark_as_offset"] == 'True')

        if self._pulse_resolved:
            self._gain_slicer_le.setText(
                parse_slice_inv(cfg["gain_slicer"]))
            self._offset_slicer_le.setText(
                parse_slice_inv(cfg["offset_slicer"]))
