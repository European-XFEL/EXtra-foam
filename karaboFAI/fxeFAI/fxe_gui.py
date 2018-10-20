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
import argparse

from ..widgets.pyqtgraph import QtGui
from ..main_gui import MainGUI
from ..config import config


def fxe_gui(argv=None):
    parser = argparse.ArgumentParser(prog="fxe-gui")

    parser.add_argument(
        '--np',
        type=int,
        help="Number of processors for data processing (default 4)")

    args = parser.parse_args(argv)
    if args.np is None:
        pass
    elif args.np <= 1:
        config["WORKERS"] = 1
    elif args.np > 1:
        config["WORKERS"] = args.np

    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainGUI("FXE", screen_size=screen_size)
    app.exec_()
