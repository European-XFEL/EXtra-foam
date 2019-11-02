"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5 import QtGui

from ..mediator import Mediator


class AbstractCtrlWidget(QtGui.QGroupBox):
    GROUP_BOX_STYLE_SHEET = 'QGroupBox:title {'\
                            'color: #8B008B;' \
                            'border: 1px;' \
                            'subcontrol-origin: margin;' \
                            'subcontrol-position: top left;' \
                            'padding-left: 10px;' \
                            'padding-top: 10px;' \
                            'margin-top: 0.0em;}'
    class SourcePropertyItem:
        def __init__(self, device_ids=None, properties=None):
            self.device_ids = device_ids if device_ids is not None else []
            self.properties = properties if properties is not None else []

    # Data categories for different topics
    _TOPIC_DATA_CATEGORIES = {
        "UNKNOWN": OrderedDict({
            "": SourcePropertyItem(),
            "Train ID": SourcePropertyItem(
                device_ids=["", "Any"],
                properties=["timestamp.tid"]
            ),
            "User defined": SourcePropertyItem()}),
        "SPB": OrderedDict({
            "": SourcePropertyItem(),
            "Train ID": SourcePropertyItem(
                device_ids=["", "Any"],
                properties=["timestamp.tid"]
            ),
            "User defined": SourcePropertyItem(),
        }),
        "FXE": OrderedDict({
            "": SourcePropertyItem(),
            "Train ID": SourcePropertyItem(
                device_ids=["", "Any"],
                properties=["timestamp.tid"]
            ),
            "User defined": SourcePropertyItem(),
            "Motor": SourcePropertyItem(
                device_ids=[
                    "",
                    "FXE_SMS_USR/MOTOR/UM01",
                    "FXE_SMS_USR/MOTOR/UM02",
                    "FXE_SMS_USR/MOTOR/UM04",
                    "FXE_SMS_USR/MOTOR/UM05",
                    "FXE_SMS_USR/MOTOR/UM13",
                    "FXE_AUXT_LIC/DOOCS/PPLASER",
                    "FXE_AUXT_LIC/DOOCS/PPODL",
                ],
                properties=["actualPosition"],
            ),
        }),
        "SCS": OrderedDict({
            "": SourcePropertyItem(),
            "Train ID": SourcePropertyItem(
                device_ids=["", "Any"],
                properties=["timestamp.tid"]
            ),
            "User defined": SourcePropertyItem(),
            "MonoChromator": SourcePropertyItem(
                device_ids=[
                    "",
                    "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"
                ],
                properties=["actualEnergy"],
            ),
            "Motor": SourcePropertyItem(
                device_ids=[
                    "",
                    "SCS_ILH_LAS/PHASESHIFTER/DOOCS",
                    "SCS_ILH_LAS/DOOCS/PP800_PHASESHIFTER",
                    "SCS_ILH_LAS/MOTOR/LT3",
                ],
                properties=["actualPosition"],
            ),
            "MAGNET": SourcePropertyItem(
                device_ids=[
                    "",
                    "SCS_CDIFFT_MAG/SUPPLY/CURRENT",
                ],
                properties=["actualCurrent"],
            ),
        }),
        "SQS": OrderedDict({
            "": SourcePropertyItem(),
            "Train ID": SourcePropertyItem(
                device_ids=["", "Any"],
                properties=["timestamp.tid"]
            ),
            "User defined": SourcePropertyItem(),
        }),
        "MID": OrderedDict({
            "": SourcePropertyItem(),
            "Train ID": SourcePropertyItem(
                device_ids=["", "Any"],
                properties=["timestamp.tid"]
            ),
            "User defined": SourcePropertyItem(),
        }),
        "HED": OrderedDict({
            "": SourcePropertyItem(),
            "Train ID": SourcePropertyItem(
                device_ids=["", "Any"],
                properties=["timestamp.tid"]
            ),
            "User defined": SourcePropertyItem(),
        }),
    }

    def __init__(self, title, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(title, parent=parent)
        self.setStyleSheet(self.GROUP_BOX_STYLE_SHEET)

        self._mediator = Mediator()

        # widgets whose values are not allowed to change after the "run"
        # button is clicked
        self._non_reconfigurable_widgets = []

        # whether the related detector is pulse resolved or not
        self._pulse_resolved = pulse_resolved

    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    def initConnections(self):
        """Initialization of signal-slot connections."""
        pass

    def onStart(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(False)

    def onStop(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(True)

    def updateMetaData(self):
        """Update metadata belong to this control widget.

        :return: None if any of the parameters is invalid. Otherwise, a
            string of to be logged information.
        """
        return True
