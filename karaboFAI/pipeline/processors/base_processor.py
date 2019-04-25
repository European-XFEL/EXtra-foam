"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

BaseProcessor

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""


class AbstractProcessor:
    """Base class for specific data processor."""

    def __init__(self):
        self.__enabled = False

    def setEnabled(self, state):
        self.__enabled = state

    def isEnabled(self):
        return self.__enabled

    def process(self, proc_data, raw_data=None):
        """Process data.

        :param ProcessedData proc_data: processed data.
        :param dict raw_data: raw data received from the bridge.

        :return str: error message.
        """
        raise NotImplementedError
