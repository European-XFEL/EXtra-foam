"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

XgmProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor, _get_slow_data
from ..exceptions import ProcessingError
from ...utils import profiler


class XgmProcessor(_BaseProcessor):
    """Process XGM data."""

    def __init__(self):
        super().__init__()

        self._pipeline_srcs = []
        self._instrument_srcs = []

    def update(self):
        srcs = self._meta.get_all_data_sources("XGM")
        self._pipeline_srcs = []
        self._instrument_srcs = []
        for src in srcs:
            if src.name.split(":")[-1] == "output":
                self._pipeline_srcs.append(src)
            else:
                self._instrument_srcs.append(src)

    @profiler("XGM Processor")
    def process(self, data):
        """Process XGM data"""
        processed = data['processed']
        raw = data['raw']
        src_type = data['source_type']
        tid = processed.tid

        err_msgs = []

        for src in self._instrument_srcs:
            fom, err = _get_slow_data(tid, raw, src.name, "pulseEnergy.photonFlux")
            processed.xgm.fom = fom
            if err:
                err_msgs.append(err)

        for src in self._pipeline_srcs:
            print(raw[src.name]["data.intensitySa3TD"])

        for msg in err_msgs:
            raise ProcessingError('[XGM] ' + msg)
