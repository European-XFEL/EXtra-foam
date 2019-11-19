"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor
from ..data_model import ProcessedData
from ...database import MonProxy


class Broker(_BaseProcessor):
    """Broker class."""
    def __init__(self):
        super().__init__()

        self._mon = MonProxy()

    def update(self):
        """Override."""
        pass

    def process(self, data):
        """Override."""
        # get the train ID of the first metadata
        # Note: this is better than meta[src_name] because:
        #       1. For streaming AGIPD/LPD data from files, 'src_name' does
        #          not work;
        #       2. Prepare for the scenario where a 2D detector is not
        #          mandatory.
        tid = next(iter(data['meta'].values()))["timestamp.tid"]
        processed = ProcessedData(tid)
        data["processed"] = processed

        available_sources = {k: v["timestamp.tid"]
                             for k, v in data['meta'].items()}

        self._mon.set_available_sources(available_sources)
