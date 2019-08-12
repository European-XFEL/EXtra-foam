"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AbstractCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtGui

from ..mediator import Mediator
from ...config import config


class CorrelationParam:
    def __init__(self, device_ids=None, properties=None):
        if device_ids is None:
            self.device_ids = []
        else:
            self.device_ids = device_ids

        if properties is None:
            self.properties = []
        else:
            self.properties = properties


# Leave the default device ID empty since the available devices
# in different instruments are different.
#
_TOPIC_DATA_CATEGORIES = {
    "GENERAL": OrderedDict({
        "": CorrelationParam(),
        "User defined": CorrelationParam(),
        "Train ID": CorrelationParam(
            device_ids=["", "Any"],
            properties=["timestamp.tid"]
        )}),
    "FXE": OrderedDict({
        "": CorrelationParam(),
        "XGM": CorrelationParam(
            device_ids=[
                "",
                "SA1_XTD2_XGM/DOOCS/MAIN",
                "SPB_XTD9_XGM/DOOCS/MAIN",
            ],
            properties=["data.intensityTD"],
        ),
        "Train ID": CorrelationParam(
            device_ids=["", "Any"],
            properties=["timestamp.tid"]
        ),
        "Motor": CorrelationParam(
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
        "MonoChromator": CorrelationParam(
            device_ids=[
                "",
                "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"
            ],
            properties=["actualEnergy"],
        ),
        "User defined": CorrelationParam()
    }),
    "SCS": OrderedDict({
        "": CorrelationParam(),
        "XGM": CorrelationParam(
            device_ids=[
                "",
                "SA3_XTD10_XGM/XGM/DOOCS",
                "SCS_BLU_XGM/XGM/DOOCS"
            ],
            properties=["data.intensityTD"],
        ),
        "Digitizer": CorrelationParam(
            device_ids=[
                "",
                "SCS_UTC1_ADQ/ADC/1"
            ],
            properties=["MCP1", "MCP2", "MCP3", "MCP4"],
        ),
        "Train ID": CorrelationParam(
            device_ids=["", "Any"],
            properties=["timestamp.tid"]
        ),
        "Motor": CorrelationParam(
            device_ids=[
                "",
                "SCS_SMS_USR/MOTOR/UM01",
                "SCS_SMS_USR/MOTOR/UM02",
                "SCS_SMS_USR/MOTOR/UM04",
                "SCS_SMS_USR/MOTOR/UM05",
                "SCS_SMS_USR/MOTOR/UM13",
                "SCS_AUXT_LIC/DOOCS/PPLASER",
                "SCS_AUXT_LIC/DOOCS/PPODL",
            ],
            properties=["actualPosition"],
        ),
        "MonoChromator": CorrelationParam(
            device_ids=[
                "",
                "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"
            ],
            properties=["actualEnergy"],
        ),
        "User defined": CorrelationParam()
    }),
}


class AbstractCtrlWidget(QtGui.QGroupBox):
    GROUP_BOX_STYLE_SHEET = 'QGroupBox:title {'\
                            'color: #8B008B;' \
                            'border: 1px;' \
                            'subcontrol-origin: margin;' \
                            'subcontrol-position: top left;' \
                            'padding-left: 10px;' \
                            'padding-top: 10px;' \
                            'margin-top: 0.0em;}'

    def __init__(self, title, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(title, parent=parent)
        self.setStyleSheet(self.GROUP_BOX_STYLE_SHEET)

        parent = self.parent()
        if parent is not None:
            parent.registerCtrlWidget(self)

        self._mediator = Mediator()

        # widgets whose values are not allowed to change after the "run"
        # button is clicked
        self._non_reconfigurable_widgets = []

        # whether the related detector is pulse resolved or not
        self._pulse_resolved = pulse_resolved
        try:
            self._data_categories = _TOPIC_DATA_CATEGORIES[config["TOPIC"]]
        except KeyError:
            self._data_categories = _TOPIC_DATA_CATEGORIES["GENERAL"]

    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

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
