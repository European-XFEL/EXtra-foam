"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler

from . import ROOT_PATH


# disable DEBUG information from imported module pyFAI
logging.getLogger("pyFAI").setLevel(logging.CRITICAL)


def create_logger():
    """Create the logger object for the whole API."""
    name = "EXtra-foam"
    logger = logging.getLogger(name)

    log_file = os.path.join(ROOT_PATH, name + ".log")
    fh = TimedRotatingFileHandler(log_file, when='midnight')

    fh.suffix = "%Y%m%d"
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(filename)s - %(levelname)s - %(message)s'
    )

    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = create_logger()
logger.setLevel(logging.INFO)
