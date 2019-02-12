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

from .base_ctrl_widgets import AbstractCtrlWidget
from ..helpers import parse_ids
from ..logger import logger
from ..widgets.pyqtgraph import QtCore, QtGui
from ..data_processing import OpLaserMode


class PumpProbeCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for pump-probe experiments."""

    _available_modes = OrderedDict({
        "inactive": OpLaserMode.INACTIVE,
        "normal": OpLaserMode.NORMAL,
        "even/odd": OpLaserMode.EVEN_ON,
        "odd/even": OpLaserMode.ODD_ON
    })

    # (mode, on-pulse ids, off-pulse ids)
    on_off_pulse_ids_sgn = QtCore.pyqtSignal(object, list, list)

    moving_average_window_sgn = QtCore.pyqtSignal(int)

    abs_difference_sgn = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__("Pump-probe analysis setup", *args, **kwargs)

        self._laser_mode_cb = QtGui.QComboBox()

        # We keep the definitions of attributes which are not used in the
        # PULSE_RESOLVED = True case. It makes sense since these attributes
        # also appear in the defined methods.

        all_keys = list(self._available_modes.keys())
        if self._pulse_resolved:
            self._laser_mode_cb.addItems(all_keys)
            on_pulse_ids = "0:8:2"
            off_pulse_ids = "1:8:2"
        else:
            all_keys.remove("normal")
            self._laser_mode_cb.addItems(all_keys)
            on_pulse_ids = "0"
            off_pulse_ids = "0"

        self.abs_difference_cb = QtGui.QCheckBox("Absolute difference")
        self.abs_difference_cb.setChecked(True)

        self._on_pulse_le = QtGui.QLineEdit(on_pulse_ids)
        self._off_pulse_le = QtGui.QLineEdit(off_pulse_ids)

        self._moving_average_window_le = QtGui.QLineEdit("600")
        self._moving_average_window_le.setValidator(
            QtGui.QIntValidator(1, 600))

        self.reset_btn = QtGui.QPushButton("Reset")

        self._disabled_widgets_during_daq = [
            self._laser_mode_cb,
            self._on_pulse_le,
            self._off_pulse_le,
            self._moving_average_window_le,
            self.abs_difference_cb,
        ]

        self.initUI()

    def initUI(self):
        """Overload."""
        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        layout.addRow("Laser on/off mode: ", self._laser_mode_cb)
        layout.addRow(self.abs_difference_cb)
        if self._pulse_resolved:
            layout.addRow("On-pulse IDs: ", self._on_pulse_le)
            layout.addRow("Off-pulse IDs: ", self._off_pulse_le)
        layout.addRow("Moving average window: ",
                      self._moving_average_window_le)
        layout.addRow(self.reset_btn)

        self.setLayout(layout)

    def updateSharedParameters(self):
        """Override"""
        mode_description = self._laser_mode_cb.currentText()
        mode = self._available_modes[mode_description]

        if mode != OpLaserMode.INACTIVE:
            try:
                # check pulse ID only when laser on/off pulses are in the same
                # train (the "normal" mode)
                on_pulse_ids = parse_ids(self._on_pulse_le.text())
                off_pulse_ids = parse_ids(self._off_pulse_le.text())
                if not on_pulse_ids or not off_pulse_ids:
                    raise ValueError
                if mode == OpLaserMode.NORMAL and self._pulse_resolved:
                    common = set(on_pulse_ids).intersection(off_pulse_ids)
                    if common:
                        logger.error("Pulse IDs {} are found in both on- and "
                                     "off- pulses.".format(','.join([str(v) for v in common])))
                        return None

            except ValueError:
                logger.error("Invalid input! Enter on/off pulse IDs separated "
                             "by ',' and/or use the range operator ':'!")
                return None
        else:
            on_pulse_ids = []
            off_pulse_ids = []

        self.on_off_pulse_ids_sgn.emit(mode, on_pulse_ids, off_pulse_ids)

        abs_diff_state = self.abs_difference_cb.checkState()
        self.abs_difference_sgn.emit(abs_diff_state)

        window_size = int(self._moving_average_window_le.text())
        if window_size < 1:
            logger.error("Moving average window < 1!")
            return None
        self.moving_average_window_sgn.emit(window_size)

        info = "\n<Optical laser mode>: {}".format(mode_description)
        if on_pulse_ids and off_pulse_ids:
            if self._pulse_resolved:
                info += "\n<On-pulse IDs>: {}".format(on_pulse_ids)
                info += "\n<Off-pulse IDs>: {}".format(off_pulse_ids)
            info += "\n<Moving average window>: {}".format(window_size)

        info += "\n<Use absolute difference>: {}".format(bool(abs_diff_state))

        return info
