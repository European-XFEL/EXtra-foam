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
from ..gui_helpers import parse_ids
from ..mediator import Mediator
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
        "A.I.": PumpProbeType.AZIMUTHAL_INTEGRATION,
        "ROI": PumpProbeType.ROI,
        "Projection X": PumpProbeType.ROI_PROJECTION_X,
        "Projection Y": PumpProbeType.ROI_PROJECTION_Y,
    })

    # (mode, on-pulse ids, off-pulse ids)
    pp_pulse_ids_sgn = QtCore.pyqtSignal(object, list, list)
    # analysis type
    pp_analysis_type_sgn = QtCore.pyqtSignal(object)

    abs_difference_sgn = QtCore.pyqtSignal(int)

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

        self.abs_difference_cb = QtGui.QCheckBox("FOM from absolute difference")
        self.abs_difference_cb.setChecked(True)

        self._on_pulse_le = QtGui.QLineEdit(on_pulse_ids)
        self._off_pulse_le = QtGui.QLineEdit(off_pulse_ids)

        self._ma_window_le = QtGui.QLineEdit("1")
        self._ma_window_le.setValidator(QtGui.QIntValidator(1, 99999))
        self.reset_btn = QtGui.QPushButton("Reset")

        self._non_reconfigurable_widgets = [
            self._mode_cb,
            self._analysis_type_cb,
            self._on_pulse_le,
            self._off_pulse_le,
            self.abs_difference_cb,
        ]

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(self.reset_btn, 0, 1)
        layout.addWidget(QtGui.QLabel("On/off mode: "), 1, 0, AR)
        layout.addWidget(self._mode_cb, 1, 1)
        layout.addWidget(QtGui.QLabel("Analysis type: "), 2, 0, AR)
        layout.addWidget(self._analysis_type_cb, 2, 1)
        if self._pulse_resolved:
            layout.addWidget(QtGui.QLabel("On-pulse IDs: "), 3, 0, AR)
            layout.addWidget(self._on_pulse_le, 3, 1)
            layout.addWidget(QtGui.QLabel("Off-pulse IDs: "), 4, 0, AR)
            layout.addWidget(self._off_pulse_le, 4, 1)

        layout.addWidget(QtGui.QLabel("Moving average window: "), 5, 0, 1, 1)
        layout.addWidget(self._ma_window_le, 5, 1, 1, 1)
        layout.addWidget(self.abs_difference_cb, 6, 0, 1, 2)

        self.setLayout(layout)

    def initConnections(self):
        mediator = Mediator()

        self._ma_window_le.editingFinished.connect(
            lambda: mediator.pp_ma_window_change_sgn.emit(
                int(self._ma_window_le.text())))
        self._ma_window_le.editingFinished.emit()

    def updateSharedParameters(self):
        """Override"""
        mode_str = self._mode_cb.currentText()
        mode = self._available_modes[mode_str]

        fom_str = self._analysis_type_cb.currentText()
        type_ = self._analysis_types[fom_str]

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
                                 "off- pulses.".format(','.join([str(v) for v in common])))
                    return False

        except ValueError:
            logger.error("Invalid input! Enter on/off pulse IDs separated "
                         "by ',' and/or use the range operator ':'!")
            return False

        self.pp_pulse_ids_sgn.emit(mode, on_pulse_ids, off_pulse_ids)
        self.pp_analysis_type_sgn.emit(type_)

        abs_diff_state = self.abs_difference_cb.checkState()
        self.abs_difference_sgn.emit(abs_diff_state)

        return True
