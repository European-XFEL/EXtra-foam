"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Services.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import sys

from PyQt5.QtWidgets import QApplication

from . import __version__
from .config import config
from .logger import logger
from .pipeline import Bridge, Scheduler
from .gui import MainGUI


class FaiServer:
    _app = None

    def __init__(self, detector=None):
        """Initialization."""
        self.__class__.start()

        if detector is None:
            return

        # update global configuration
        config.load(detector)

        # a zmq bridge which acquires the data in another thread
        bridge = Bridge()
        # a data processing worker which processes the data in another thread
        scheduler = Scheduler()
        bridge.connect(scheduler)

        self.gui = MainGUI(bridge=bridge, scheduler=scheduler)

    @classmethod
    def start(cls):
        if cls._app is None:
            cls._app = QApplication(sys.argv)

    @property
    def app(self):
        return self._app.instance()

    def exec(self):
        self._app.exec_()


def start():
    parser = argparse.ArgumentParser(prog="karaboFAI")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=['AGIPD', 'LPD', 'JUNGFRAU', 'FASTCCD'],
                        type=lambda s: s.upper())
    parser.add_argument('--debug', action='store_true',
                        help="Enable faulthandler")

    args = parser.parse_args()

    if args.debug:
        import faulthandler
        faulthandler.enable()
        logger.debug("'faulthandler enabled")

    detector = args.detector
    if detector == 'JUNGFRAU':
        detector = 'JungFrau'
    elif detector == 'FASTCCD':
        detector = 'FastCCD'
    else:
        detector = detector.upper()

    FaiServer(detector).exec()


if __name__ == "__main__":

    start()
