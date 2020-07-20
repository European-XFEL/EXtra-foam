"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
import copy

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QLabel, QPushButton
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartSliceLineEdit
from ..gui_helpers import invert_dict
from ...config import PumpProbeMode, AnalysisType
from ...database import Metadata as mt


class PumpProbeCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up pump-probe analysis parameters."""

    __available_modes = OrderedDict({
        "": PumpProbeMode.UNDEFINED,
        "reference as off": PumpProbeMode.REFERENCE_AS_OFF,
        "even/odd train": PumpProbeMode.EVEN_TRAIN_ON,
        "odd/even train": PumpProbeMode.ODD_TRAIN_ON,
        "same train": PumpProbeMode.SAME_TRAIN,
    })

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI FOM": AnalysisType.ROI_FOM,
        "ROI proj": AnalysisType.ROI_PROJ,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG,
    })
    _analysis_types_inv = invert_dict(_analysis_types)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mode_cb = QComboBox()

        self._on_pulse_le = SmartSliceLineEdit(":")
        self._off_pulse_le = SmartSliceLineEdit(":")

        self._available_modes = copy.copy(self.__available_modes)
        self._available_modes_inv = invert_dict(self._available_modes)
        if not self._pulse_resolved:
            del self._available_modes["same train"]
            del self._available_modes_inv[PumpProbeMode.SAME_TRAIN]
            self._on_pulse_le.setEnabled(False)
            self._off_pulse_le.setEnabled(False)
        self._mode_cb.addItems(list(self._available_modes.keys()))

        self._analysis_type_cb = QComboBox()
        self._analysis_type_cb.addItems(list(self._analysis_types.keys()))

        self._abs_difference_cb = QCheckBox("FOM from absolute on-off")
        self._abs_difference_cb.setChecked(True)

        self._reset_btn = QPushButton("Reset")

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(QLabel("Analysis type: "), 0, 0, AR)
        layout.addWidget(self._analysis_type_cb, 0, 1)
        layout.addWidget(QLabel("Mode: "), 1, 0, AR)
        layout.addWidget(self._mode_cb, 1, 1)
        layout.addWidget(self._reset_btn, 2, 0, 1, 2)

        layout.addWidget(QLabel("On-pulse indices: "), 0, 2, AR)
        layout.addWidget(self._on_pulse_le, 0, 3)
        layout.addWidget(QLabel("Off-pulse indices: "), 1, 2, AR)
        layout.addWidget(self._off_pulse_le, 1, 3)

        layout.addWidget(self._abs_difference_cb, 0, 4, AR)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._reset_btn.clicked.connect(mediator.onPpReset)

        self._abs_difference_cb.toggled.connect(
            mediator.onPpAbsDifferenceChange)

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onPpAnalysisTypeChange(
                self._analysis_types[x]))

        self._mode_cb.currentTextChanged.connect(
            lambda x: mediator.onPpModeChange(self._available_modes[x]))
        self._mode_cb.currentTextChanged.connect(
            lambda x: self.onPpModeChange(self._available_modes[x]))

        self._on_pulse_le.value_changed_sgn.connect(
            mediator.onPpOnPulseSlicerChange)
        self._off_pulse_le.value_changed_sgn.connect(
            mediator.onPpOffPulseSlicerChange)

    def updateMetaData(self):
        """Override"""
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())

        self._mode_cb.currentTextChanged.emit(self._mode_cb.currentText())

        self._on_pulse_le.returnPressed.emit()
        self._off_pulse_le.returnPressed.emit()

        self._abs_difference_cb.toggled.emit(
            self._abs_difference_cb.isChecked())

        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.PUMP_PROBE_PROC)
        if not cfg:
            # not initialized
            return

        self._analysis_type_cb.setCurrentText(
            self._analysis_types_inv[int(cfg["analysis_type"])])

        self._mode_cb.setCurrentText(
            self._available_modes_inv[int(cfg["mode"])])

        self._abs_difference_cb.setChecked(cfg["abs_difference"] == 'True')

        if self._pulse_resolved:
            self._updateWidgetValue(
                self._on_pulse_le, cfg, "on_pulse_slicer")
            self._updateWidgetValue(
                self._off_pulse_le, cfg, "off_pulse_slicer")

    def onPpModeChange(self, pp_mode):
        if not self._pulse_resolved:
            return

        if pp_mode == PumpProbeMode.REFERENCE_AS_OFF:
            # off-pulse indices are ignored in REFERENCE_AS_OFF mode.
            self._off_pulse_le.setEnabled(False)
        else:
            self._off_pulse_le.setEnabled(True)

    def resetAnalysisType(self):
        self._analysis_type_cb.setCurrentText(
            self._analysis_types_inv[AnalysisType.UNDEFINED])
