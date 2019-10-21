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


class XgmExtractor(_BaseProcessor):
    """XGM data extractor.

    Attributes:
        _sources (list): a list of SourceItems.
        _pulse_slicer (slice): a slice object which will be used to slice
            pulses for pulse-resolved pipeline data.
    """

    def __init__(self):
        super().__init__()

        self._sources = []
        self._pulse_slicer = slice(None, None)

    def update(self):
        self._sources = self._meta.get_all_data_sources("XGM")
        self._pulse_slicer = self.str2slice(
            self._meta.get(mt.GLOBAL_PROC, 'selected_xgm_pulse_indices'))

    @profiler("XGM Processor")
    def process(self, data):
        """Process XGM data"""
        processed = data['processed']
        raw = data['raw']
        tid = processed.tid

        instrument_srcs = []
        pipeline_srcs = []
        for src in self._sources:
            if src.name.split(":")[-1] == "output":
                pipeline_srcs.append(src)
            else:
                instrument_srcs.append(src)

        err_msgs = []

        # instrument data
        for src in instrument_srcs:
            v, err = self._fetch_property_data(
                tid, raw, src.name, src.property)

            if err:
                err_msgs.append(err)
            else:
                xgm_ppts = DATA_SOURCE_PROPERTIES["XGM"]
                processed.xgm.__dict__[xgm_ppts[src.property]] = v

        # pipeline data
        for src in pipeline_srcs:
            v, err = self._fetch_property_data(
                tid, raw, src.name, src.property)

            if err:
                err_msgs.append(err)
            else:
                xgm_ppts = DATA_SOURCE_PROPERTIES["XGM:output"]
                processed.pulse.xgm.__dict__[xgm_ppts[src.property]] = v[
                    self._pulse_slicer]

        for msg in err_msgs:
            raise ProcessingError(f'[XGM] {msg}')
