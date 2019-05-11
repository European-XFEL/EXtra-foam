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
from ..misc_widgets import SmartLineEdit
from ..gui_helpers import parse_ids
from ...config import PumpProbeMode, PumpProbeType
from ...logger import logger


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
        "azimuthal integ": PumpProbeType.AZIMUTHAL_INTEG,
        "ROI": PumpProbeType.ROI,
        "projection X": PumpProbeType.ROI_PROJECTION_X,
        "projection Y": PumpProbeType.ROI_PROJECTION_Y,
        "ROI1 / ROI2": PumpProbeType.ROI1_BY_ROI2,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Pump-probe analysis setup", *args, **kwargs)

        self._mode_cb = QtGui.QComboBox()

        # We keep the definitions of attributes which are not used in the
        # PULSE_RESOLVED = True case. It makes sense since these attributes
        # also appear in the defined methods.

        all_keys = list(self._available_modes.keys())
        if self._pulse_resolved:
            self._mode_cb.addItems(all_keys)
            on_pulse_ids = "0:8:2"
            off_pulse_ids = "1:8:2"
        else:
            all_keys.remove("same train")
            self._mode_cb.addItems(all_keys)
            on_pulse_ids = "0"
            off_pulse_ids = "0"

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(list(self._analysis_types.keys()))

        self._abs_difference_cb = QtGui.QCheckBox("FOM from absolute on-off")

        self._on_pulse_le = QtGui.QLineEdit(on_pulse_ids)
        self._off_pulse_le = QtGui.QLineEdit(off_pulse_ids)

        self._ma_window_le = SmartLineEdit("1")
        self._ma_window_le.setValidator(QtGui.QIntValidator(1, 99999))
        self._reset_btn = QtGui.QPushButton("Reset")

        self._non_reconfigurable_widgets = [
            self._mode_cb,
            self._on_pulse_le,
            self._off_pulse_le,
        ]

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
            layout.addWidget(QtGui.QLabel("On-pulse IDs: "), 2, 0, AR)
            layout.addWidget(self._on_pulse_le, 2, 1)
            layout.addWidget(QtGui.QLabel("Off-pulse IDs: "), 2, 2, AR)
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
        self._ma_window_le.returnPressed.emit()

        self._abs_difference_cb.toggled.connect(
            mediator.onPpAbsDifferenceChange)
        self._abs_difference_cb.setChecked(True)

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onPpAnalysisTypeChange(
                self._analysis_types[x]))
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())

    def updateSharedParameters(self):
        """Override"""
        mode_str = self._mode_cb.currentText()
        mode = self._available_modes[mode_str]

        try:
            # check pulse ID only when laser on/off pulses are in the same
            # train (the "normal" mode)
            on_pulse_ids = parse_ids(self._on_pulse_le.text())
            if mode == PumpProbeMode.PRE_DEFINED_OFF:
                off_pulse_ids = []
            else:
                off_pulse_ids = parse_ids(self._off_pulse_le.text())

            if mode == PumpProbeMode.SAME_TRAIN and self._pulse_resolved:
                common = set(on_pulse_ids).intersection(off_pulse_ids)
                if common:
                    logger.error("Pulse IDs {} are found in both on- and "
                                 "off- pulses.".
                                 format(','.join([str(v) for v in common])))
                    return False

        except ValueError:
            logger.error("Invalid input! Enter on/off pulse IDs separated "
                         "by ',' and/or use the range operator ':'!")
            return False

        self._mediator.onPpModeChange(mode)
        self._mediator.onPpOnPulseIdsChange(on_pulse_ids)
        self._mediator.onPpOffPulseIdsChange(off_pulse_ids)

        return True
