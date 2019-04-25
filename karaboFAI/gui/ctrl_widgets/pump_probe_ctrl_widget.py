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
from ...config import PumpProbeFom, PumpProbeMode
from ...logger import logger

mediator = Mediator()


class PumpProbeCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for pump-probe experiments."""

    _available_modes = OrderedDict({
        "": PumpProbeMode.INACTIVATE,
        "pre-defined off": PumpProbeMode.PRE_DEFINED_OFF,
        "same train": PumpProbeMode.SAME_TRAIN,
        "even/odd train": PumpProbeMode.EVEN_TRAIN_ON,
        "odd/even train": PumpProbeMode.ODD_TRAIN_ON
    })

    _analysis_foms = OrderedDict({
        "Azimuthal integration": PumpProbeFom.AZIMUTHAL_INTEGRATION,
        "ROI": PumpProbeFom.ROI,
        "ROI 1D projection": PumpProbeFom.ROI_1D_PROJECTION
    })

    # (mode, on-pulse ids, off-pulse ids)
    pp_pulse_ids_sgn = QtCore.pyqtSignal(object, list, list)

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

        self._fom_cb = QtGui.QComboBox()
        self._fom_cb.addItems(list(self._analysis_foms.keys()))

        self.abs_difference_cb = QtGui.QCheckBox("FOM from absolute difference")
        self.abs_difference_cb.setChecked(True)

        self._on_pulse_le = QtGui.QLineEdit(on_pulse_ids)
        self._off_pulse_le = QtGui.QLineEdit(off_pulse_ids)

        self._ma_window_le = QtGui.QLineEdit("1")
        self._ma_window_le.setValidator(QtGui.QIntValidator(1, 99999))
        self.reset_btn = QtGui.QPushButton("Reset")

        self._disabled_widgets_during_daq = [
            self._mode_cb,
            self._fom_cb,
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
        layout.addWidget(QtGui.QLabel("Mode: "), 1, 0, AR)
        layout.addWidget(self._mode_cb, 1, 1)
        layout.addWidget(QtGui.QLabel("FOM: "), 2, 0, AR)
        layout.addWidget(self._fom_cb, 2, 1)
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
        self._ma_window_le.editingFinished.connect(
            lambda: mediator.pp_ma_window_change_sgn.emit(
                int(self._ma_window_le.text())))
        self._ma_window_le.editingFinished.emit()

    def updateSharedParameters(self):
        """Override"""
        mode_description = self._mode_cb.currentText()
        mode = self._available_modes[mode_description]

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

        abs_diff_state = self.abs_difference_cb.checkState()
        self.abs_difference_sgn.emit(abs_diff_state)

        return True
