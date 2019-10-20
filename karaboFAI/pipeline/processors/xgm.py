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
from ...database import Metadata as mt


class XgmProcessor(_BaseProcessor):
    """Process XGM data."""

    def __init__(self):
        super().__init__()

        self._pipeline_src = None
        self._instrument_src = None

        self._pulse_slicer = slice(None, None)

    def update(self):
        srcs = self._meta.get_all_data_sources("XGM")
        assert(len(srcs) <= 2)
        for src in srcs:
            if src.name.split(":")[-1] == "output":
                self._pipeline_src = src
            else:
                self._instrument_src = src

        self._pulse_slicer = self.str2slice(
            self._meta.get(mt.GLOBAL_PROC, 'selected_xgm_pulse_indices'))

        # pump-probe
        pp_cfg = self._meta.get_all(mt.PUMP_PROBE_PROC)

    @profiler("XGM Processor")
    def process(self, data):
        """Process XGM data"""
        processed = data['processed']
        raw = data['raw']
        src_type = data['source_type']
        tid = processed.tid

        err_msgs = []

        # instrument data
        src = self._instrument_src

        v, err = _get_slow_data(tid, raw, src.name, src.property)
        processed.xgm.fom = v
        if err:
            err_msgs.append(err)

        # pipeline data
        src = self._pipeline_src

        processed.pulse.xgm.intensity = raw[src.name][src.property]

        for msg in err_msgs:
            raise ProcessingError('[XGM] ' + msg)
