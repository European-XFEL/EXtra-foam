"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Loggers.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler

from .config import ROOT_PATH

# disable DEBUG information from imported module pyFAI
logging.getLogger("pyFAI").setLevel(logging.CRITICAL)


def create_logger():
    """Create the logger object for the whole API."""
    name = "karaboFAI"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_file = os.path.join(ROOT_PATH, name + ".log")
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
