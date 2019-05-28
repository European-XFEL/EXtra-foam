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
from .exceptions import AggregatingError
from ..metadata import Metadata as mt
from ..metadata import MetaProxy


class DataAggregator:
    """Aggregate data other than the detector image data."""

    def __init__(self):
        self._xgm_src = None

        self._meta = MetaProxy()

    def update(self):
        cfg = self._meta.get_all(mt.DATA_SOURCE)

        self._xgm_src = cfg['xgm_source_name']

    def aggregate(self, proc_data, raw_data):
        """Aggregate data.

        :param ProcessedData proc_data: processed data.
        :param dict raw_data: raw data received from the bridge.

        :return str: error message.
        """
        self._aggregate_xgm(self._xgm_src, proc_data, raw_data)

    @staticmethod
    def _aggregate_xgm(src_name, proc_data, raw_data):

        if not src_name:
            proc_data.xgm.source = None
            proc_data.xgm.intensity = 0
            return

        if src_name not in raw_data:
            raise AggregatingError(
                f"XGM device '{src_name}' is not in the data!")

        xgm_data = raw_data[src_name]
        ppt = 'pulseEnergy.photonFlux'
        if ppt not in xgm_data:
            ppt = f"{ppt}.value"  # From the file

        proc_data.xgm.source = src_name
        proc_data.xgm.intensity = xgm_data[ppt]
