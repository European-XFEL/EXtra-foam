"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

Logging module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler


# disable DEBUG information from imported module pyFAI
logging.getLogger("pyFAI").setLevel(logging.CRITICAL)


def create_logger():
    """Create the logger object for the whole API."""
    name = "FXE_Azimuthal_Integration"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_file_dir = os.path.join(os.path.expanduser('~'), '.fxe-tools')
    if not os.path.isdir(log_file_dir):
        os.mkdir(log_file_dir)
    log_file = os.path.join(log_file_dir, name + ".log")
    fh = TimedRotatingFileHandler(log_file, when='midnight')

    fh.suffix = "%Y%m%d"
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(filename)s - %(levelname)s - %(message)s'
    )

    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = create_logger()


class GuiLogger(logging.Handler):
    def __init__(self, edit):
        super().__init__(level=logging.INFO)
        self._edit = edit

    def emit(self, record):
        self._edit.appendPlainText(self.format(record))
