from PyQt5.QtWidgets import QApplication

import logging

from extra_foam import __version__
from extra_foam.logger import create_logger


logger = create_logger("EXtra-foam-special-suite")
logger.setLevel(logging.INFO)


def mkQApp(args=None):
    app = QApplication.instance()
    if app is None:
        if args is None:
            return QApplication([])
        return QApplication(args)

    return app
