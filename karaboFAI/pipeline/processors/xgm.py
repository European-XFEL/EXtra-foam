"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError
from ...utils import profiler
from ...database import Metadata as mt
from ...database import DATA_SOURCE_PROPERTIES


class XgmProcessor(_BaseProcessor):
    """Process XGM data."""

    def __init__(self):
        super().__init__()

        self._pipeline_src = None
        self._instrument_src = None

        self._pulse_slicer = slice(None, None)

    def update(self):
        srcs = self._meta.get_all_data_sources("XGM")

        # guard in case there is any src which is not unregistered properly
        assert(len(srcs) < 10)

        for src in srcs:
            if src.name.split(":")[-1] == "output":
                self._pipeline_src = src
            else:
                self._instrument_src = src

        self._pulse_slicer = self.str2slice(
            self._meta.get(mt.GLOBAL_PROC, 'selected_xgm_pulse_indices'))

    @profiler("XGM Processor")
    def process(self, data):
        """Process XGM data"""
        processed = data['processed']
        raw = data['raw']
        tid = processed.tid

        err_msgs = []

        # instrument data
        src = self._instrument_src
        if src:
            v, err = self._fetch_property_data(
                tid, raw, src.name, src.property)

            xgm_ppts = DATA_SOURCE_PROPERTIES["XGM"]
            processed.xgm.__dict__[xgm_ppts[src.property]] = v
            if err:
                err_msgs.append(err)

        # pipeline data
        src = self._pipeline_src
        if src:
            v, err = self._fetch_property_data(
                tid, raw, src.name, src.property)

            xgm_ppts = DATA_SOURCE_PROPERTIES["XGM:output"]
            # when streaming from files, it has a fixed length!!!
            processed.pulse.xgm.__dict__[xgm_ppts[src.property]] = v
            if err:
                err_msgs.append(err)

        for msg in err_msgs:
            raise ProcessingError(f'[XGM] {msg}')
