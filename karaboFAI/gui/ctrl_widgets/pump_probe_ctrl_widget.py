"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from .smart_widgets import SmartLineEdit, SmartRangeLineEdit
from ...config import PumpProbeMode, AnalysisType


class PumpProbeCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for pump-probe experiments."""

    _available_modes = OrderedDict({
        "": PumpProbeMode.UNDEFINED,
        "predefined off": PumpProbeMode.PRE_DEFINED_OFF,
        "same train": PumpProbeMode.SAME_TRAIN,
        "even/odd train": PumpProbeMode.EVEN_TRAIN_ON,
        "odd/even train": PumpProbeMode.ODD_TRAIN_ON
    })

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI1 - ROI2": AnalysisType.ROI1_SUB_ROI2,
        "ROI1 / ROI2": AnalysisType.ROI1_DIV_ROI2,
        "proj X (ROI1 - ROI2)": AnalysisType.ROI1_SUB_ROI2_PROJECTION_X,
        "proj Y (ROI2 - ROI2)": AnalysisType.ROI1_SUB_ROI2_PROJECTION_Y,
        "azimuthal integ": AnalysisType.TRAIN_AZIMUTHAL_INTEG,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Pump-probe setup", *args, **kwargs)

        self._mode_cb = QtGui.QComboBox()

        # We keep the definitions of attributes which are not used in the
        # PULSE_RESOLVED = True case. It makes sense since these attributes
        # also appear in the defined methods.

        all_keys = list(self._available_modes.keys())
        if self._pulse_resolved:
            self._mode_cb.addItems(all_keys)
            on_pulse_indices = ":"
            off_pulse_indices = ":"
        else:
            all_keys.remove("same train")
            self._mode_cb.addItems(all_keys)
            on_pulse_indices = "0"
            off_pulse_indices = "0"

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(list(self._analysis_types.keys()))

        self._abs_difference_cb = QtGui.QCheckBox("FOM from absolute on-off")
        self._abs_difference_cb.setChecked(True)

        self._on_pulse_le = SmartRangeLineEdit(on_pulse_indices)
        self._off_pulse_le = SmartRangeLineEdit(off_pulse_indices)

        self._ma_window_le = SmartLineEdit("1")
        self._ma_window_le.setValidator(QtGui.QIntValidator(1, 99999))
        self._reset_btn = QtGui.QPushButton("Reset")

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("On/off mode: "), 0, 0, AR)
        layout.addWidget(self._mode_cb, 0, 1)
        layout.addWidget(QtGui.QLabel("Analysis type: "), 0, 2, AR)
        layout.addWidget(self._analysis_type_cb, 0, 3)
        if self._pulse_resolved:
            layout.addWidget(QtGui.QLabel("On-pulse indices: "), 2, 0, AR)
            layout.addWidget(self._on_pulse_le, 2, 1)
            layout.addWidget(QtGui.QLabel("Off-pulse indices: "), 2, 2, AR)
            layout.addWidget(self._off_pulse_le, 2, 3)

        layout.addWidget(QtGui.QLabel("Moving average window: "), 3, 1, 1, 2, AR)
        layout.addWidget(self._ma_window_le, 3, 3, 1, 1)
        layout.addWidget(self._abs_difference_cb, 4, 0, 1, 2)
        layout.addWidget(self._reset_btn, 4, 3, 1, 1)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._reset_btn.clicked.connect(mediator.onPpReset)

        self._ma_window_le.returnPressed.connect(
            lambda: mediator.onPpMaWindowChange(
                int(self._ma_window_le.text())))

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
            mediator.onPpOnPulseIdsChange)

        self._off_pulse_le.value_changed_sgn.connect(
            mediator.onPpOffPulseIdsChange)

    def updateMetaData(self):
        """Override"""
        self._ma_window_le.returnPressed.emit()

        self._abs_difference_cb.toggled.emit(
            self._abs_difference_cb.isChecked())

        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())

        self._mode_cb.currentTextChanged.emit(self._mode_cb.currentText())

        self._on_pulse_le.returnPressed.emit()

        self._off_pulse_le.returnPressed.emit()

        return True

    def onPpModeChange(self, pp_mode):
        if pp_mode == PumpProbeMode.PRE_DEFINED_OFF:
            # off-pulse indices are ignored in PRE_DEFINED_OFF mode.
            self._off_pulse_le.setEnabled(False)
        else:
            self._off_pulse_le.setEnabled(True)
