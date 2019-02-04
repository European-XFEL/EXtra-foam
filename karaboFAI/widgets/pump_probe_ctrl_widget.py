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
from ..config import config
from ..helpers import parse_ids, parse_boundary
from ..logger import logger
from ..widgets.pyqtgraph import QtCore, QtGui


class PumpProbeCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for pump-probe experiments."""

    available_modes = OrderedDict({
        "normal": "Laser-on/off pulses in the same train",
        "even/odd": "Laser-on/off pulses in even/odd train",
        "odd/even": "Laser-on/off pulses in odd/even train"
    })
    # (mode, on-pulse ids, off-pulse ids)
    on_off_pulse_ids_sgn = QtCore.pyqtSignal(str, list, list)

    integration_range_sgn = QtCore.pyqtSignal(float, float)
    normalization_range_sgn = QtCore.pyqtSignal(float, float)
    moving_average_window_sgn = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__("Pump-probe analysis setup", *args, **kwargs)

        self._laser_mode_cb = QtGui.QComboBox()

        # We keep the definitions of attributes which are not used in the
        # PULSE_RESOLVED = True case. It makes sense since these attributes
        # also appear in the defined methods.

        if self._pulse_resolved:
            self._laser_mode_cb.addItems(self.available_modes.keys())
            on_pulse_ids = "0:8:2"
            off_pulse_ids = "1:8:2"
        else:
            self._laser_mode_cb.addItems(list(self.available_modes.keys())[1:])
            on_pulse_ids = "0"
            off_pulse_ids = "0"

        self._on_pulse_le = QtGui.QLineEdit(on_pulse_ids)
        self._off_pulse_le = QtGui.QLineEdit(off_pulse_ids)

        self._normalization_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._integration_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._moving_average_window_le = QtGui.QLineEdit("9999")

        self._disabled_widgets_during_daq = [
            self._laser_mode_cb,
            self._on_pulse_le,
            self._off_pulse_le,
            self._normalization_range_le,
            self._integration_range_le,
            self._moving_average_window_le,
        ]

        self.initUI()

    def initUI(self):
        """Overload."""
        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        layout.addRow("Laser on/off mode: ", self._laser_mode_cb)
        if self._pulse_resolved:
            layout.addRow("On-pulse IDs: ", self._on_pulse_le)
            layout.addRow("Off-pulse IDs: ", self._off_pulse_le)
        layout.addRow("Normalization range (1/A): ",
                      self._normalization_range_le)
        layout.addRow("Integration range (1/A): ",
                      self._integration_range_le)
        layout.addRow("Moving average window: ",
                      self._moving_average_window_le)

        self.setLayout(layout)

    def updateSharedParameters(self):
        """Override"""
        try:
            # check pulse ID only when laser on/off pulses are in the same
            # train (the "normal" mode)
            mode = self._laser_mode_cb.currentText()
            on_pulse_ids = parse_ids(self._on_pulse_le.text())
            off_pulse_ids = parse_ids(self._off_pulse_le.text())
            if mode == "normal" and self._pulse_resolved:
                common = set(on_pulse_ids).intersection(off_pulse_ids)
                if common:
                    logger.error(
                        "Pulse IDs {} are found in both on- and off- pulses.".
                        format(','.join([str(v) for v in common])))
                    return None

            self.on_off_pulse_ids_sgn.emit(mode, on_pulse_ids, off_pulse_ids)
        except ValueError:
            logger.error("Invalid input! Enter on/off pulse IDs separated "
                         "by ',' and/or use the range operator ':'!")
            return None

        try:
            normalization_range = parse_boundary(
                self._normalization_range_le.text())
            self.normalization_range_sgn.emit(*normalization_range)
        except ValueError as e:
            logger.error("<Normalization range>: " + str(e))
            return None

        try:
            integration_range = parse_boundary(
                self._integration_range_le.text())
            self.integration_range_sgn.emit(*integration_range)
        except ValueError as e:
            logger.error("<Integration range>: " + str(e))
            return None

        try:
            window_size = int(self._moving_average_window_le.text())
            if window_size < 1:
                logger.error("Moving average window < 1!")
                return None
            self.moving_average_window_sgn.emit(window_size)
        except ValueError as e:
            logger.error("<Moving average window>: " + str(e))
            return None

        info = "\n<Optical laser mode>: {}".format(mode)
        if self._pulse_resolved:
            info += "\n<On-pulse IDs>: {}".format(on_pulse_ids)
            info += "\n<Off-pulse IDs>: {}".format(off_pulse_ids)
        info += "\n<Normalization range>: ({}, {})".format(*normalization_range)
        info += "\n<Integration range>: ({}, {})".format(*integration_range)
        info += "\n<Moving average window>: {}".format(window_size)

        return info
