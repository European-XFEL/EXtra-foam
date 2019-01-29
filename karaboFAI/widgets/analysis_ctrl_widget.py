"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AnalysisCtrlWidget.

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


class AnalysisCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the analysis parameters."""

    available_modes = OrderedDict({
        "normal": "Laser-on/off pulses in the same train",
        "even/odd": "Laser-on/off pulses in even/odd train",
        "odd/even": "Laser-on/off pulses in odd/even train"
    })

    diff_integration_range_sgn = QtCore.pyqtSignal(float, float)
    normalization_range_sgn = QtCore.pyqtSignal(float, float)
    ma_window_size_sgn = QtCore.pyqtSignal(int)
    # (mode, on-pulse ids, off-pulse ids)
    on_off_pulse_ids_sgn = QtCore.pyqtSignal(str, list, list)
    photon_energy_sgn = QtCore.pyqtSignal(float)
    mask_range_sgn = QtCore.pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__("Analysis setup", *args, **kwargs)

        self._photon_energy_le = QtGui.QLineEdit(str(config["PHOTON_ENERGY"]))
        self._laser_mode_cb = QtGui.QComboBox()

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
        self._diff_integration_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._ma_window_le = QtGui.QLineEdit("9999")
        self._mask_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["MASK_RANGE"]]))

        self._disabled_widgets_during_daq = [
            self._photon_energy_le,
            self._laser_mode_cb,
            self._on_pulse_le,
            self._off_pulse_le,
            self._normalization_range_le,
            self._diff_integration_range_le,
            self._ma_window_le,
            self._mask_range_le
        ]

        self.initUI()

    def initUI(self):
        """Overload."""
        photon_energy_lb = QtGui.QLabel("Photon energy (keV): ")
        laser_mode_lb = QtGui.QLabel("Laser on/off mode: ")
        on_pulse_lb = QtGui.QLabel("On-pulse IDs: ")
        off_pulse_lb = QtGui.QLabel("Off-pulse IDs: ")
        normalization_range_lb = QtGui.QLabel("Normalization range (1/A): ")
        diff_integration_range_lb = QtGui.QLabel(
            "Diff integration range (1/A): ")
        ma_window_lb = QtGui.QLabel("M.A. window size: ")
        mask_range_lb = QtGui.QLabel("Mask range: ")

        layout = QtGui.QHBoxLayout()
        key_layout = QtGui.QVBoxLayout()
        key_layout.addWidget(photon_energy_lb)
        key_layout.addWidget(laser_mode_lb)
        if self._pulse_resolved:
            key_layout.addWidget(on_pulse_lb)
            key_layout.addWidget(off_pulse_lb)
        key_layout.addWidget(normalization_range_lb)
        key_layout.addWidget(diff_integration_range_lb)
        key_layout.addWidget(ma_window_lb)
        key_layout.addWidget(mask_range_lb)

        value_layout = QtGui.QVBoxLayout()
        value_layout.addWidget(self._photon_energy_le)
        value_layout.addWidget(self._laser_mode_cb)
        if self._pulse_resolved:
            value_layout.addWidget(self._on_pulse_le)
            value_layout.addWidget(self._off_pulse_le)
        value_layout.addWidget(self._normalization_range_le)
        value_layout.addWidget(self._diff_integration_range_le)
        value_layout.addWidget(self._ma_window_le)
        value_layout.addWidget(self._mask_range_le)

        layout.addLayout(key_layout)
        layout.addLayout(value_layout)
        self.setLayout(layout)

    def updateSharedParameters(self, log=False):
        """Override"""
        photon_energy = float(self._photon_energy_le.text().strip())
        if photon_energy <= 0:
            logger.error("<Photon energy>: Invalid input! Must be positive!")
            return False
        else:
            self.photon_energy_sgn.emit(photon_energy)

        try:
            # check pulse ID only when laser on/off pulses are in the same
            # train (the "normal" mode)
            mode = self._laser_mode_cb.currentText()
            on_pulse_ids = parse_ids(self._on_pulse_le.text())
            off_pulse_ids = parse_ids(self._off_pulse_le.text())
            if mode == list(self.available_modes.keys())[0] and self._pulse_resolved:
                common = set(on_pulse_ids).intersection(off_pulse_ids)
                if common:
                    logger.error(
                        "Pulse IDs {} are found in both on- and off- pulses.".
                        format(','.join([str(v) for v in common])))
                    return False

            self.on_off_pulse_ids_sgn.emit(mode, on_pulse_ids, off_pulse_ids)
        except ValueError:
            logger.error("Invalid input! Enter on/off pulse IDs separated "
                         "by ',' and/or use the range operator ':'!")
            return False

        try:
            normalization_range = parse_boundary(
                self._normalization_range_le.text())
            self.normalization_range_sgn.emit(*normalization_range)
        except ValueError as e:
            logger.error("<Normalization range>: " + str(e))
            return False

        try:
            diff_integration_range = parse_boundary(
                self._diff_integration_range_le.text())
            self.diff_integration_range_sgn.emit(*diff_integration_range)
        except ValueError as e:
            logger.error("<Diff integration range>: " + str(e))
            return False

        try:
            window_size = int(self._ma_window_le.text())
            if window_size < 1:
                logger.error("Moving average window width < 1!")
                return False
            self.ma_window_size_sgn.emit(window_size)
        except ValueError as e:
            logger.error("<Moving average window size>: " + str(e))
            return False

        try:
            mask_range = parse_boundary(self._mask_range_le.text())
            self.mask_range_sgn.emit(*mask_range)
        except ValueError as e:
            logger.error("<Mask range>: " + str(e))
            return False

        if log:
            logger.info("<Optical laser mode>: {}".format(mode))
            logger.info("<On-pulse IDs>: {}".format(on_pulse_ids))
            logger.info("<Off-pulse IDs>: {}".format(off_pulse_ids))
            logger.info("<Normalization range>: ({}, {})".
                        format(*normalization_range))
            logger.info("<Diff integration range>: ({}, {})".
                        format(*diff_integration_range))
            logger.info("<Moving average window size>: {}".
                        format(window_size))
            logger.info("<Photon energy (keV)>: {}".format(photon_energy))
            logger.info("<Mask range>: ({}, {})".format(*mask_range))

        return True
