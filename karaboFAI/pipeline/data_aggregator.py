"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data aggregator.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from .data_model import ProcessedData


class DataAggregator:
    """Aggregate data other than the detector image data."""

    def __init__(self):
        self.xgm_src = None
        self.mono_src = None

    def aggregate(self, proc_data, raw_data):
        """Process data.

        :param ProcessedData proc_data: processed data.
        :param dict raw_data: raw data received from the bridge.

        :return str: error message.
        """
        self._aggregate_xgm()
        self._aggregate_mono()

    def _aggregate_xgm(self):
        if not self.xgm_src:
            return

    def _aggregate_mono(self):
        if not self.mono_src:
            return
