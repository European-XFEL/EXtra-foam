"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

FXE instrument GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import sys

from ..widgets.pyqtgraph import QtGui
from ..main_gui import MainGUI


def fxe_gui():
    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainGUI("FXE Azimuthal Integration", screen_size=screen_size)
    app.exec_()
