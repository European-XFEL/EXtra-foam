"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

SPB instrument GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import sys

from ..widgets.pyqtgraph import QtGui
from ..main_gui import MainGUI


class SpbGUI(MainGUI):
    _default_config = {
        "TITLE": "SPB Azimuthal Integration",
        "QUAD_POSITIONS": [(-13.0, -299.0),
                           (11.0, -8.0),
                           (-254.0, 16.0),
                           (-278.0, -275.0)],
        "SERVER_ADDR": "10.253.0.53",
        "SERVER_PORT": 4501,
        "SOURCE_NAME": "FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED",
        "DISTANCE": 0.2,
        "CENTER_Y": 620,
        "CENTER_X": 580,
        "PIXEL_SIZE": 0.5e-3,
        "INTEGRATION_RANGE": (0.2, 5),
        "INTEGRATION_POINTS": 512,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(default_config=self._default_config, *args, **kwargs)


def spb_gui():
    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = SpbGUI(screen_size=screen_size)
    app.exec_()
