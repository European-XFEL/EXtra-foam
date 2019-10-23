"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import math

from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray
from ..exceptions import ProcessingError
from ...utils import profiler
from ...database import Metadata as mt
from ...database import DATA_SOURCE_PROPERTIES


class XgmProcessor(_BaseProcessor):
    """XGM data processor.

    Attributes:
        _sources (list): a list of SourceItems.
        _pulse_slicer (slice): a slice object which will be used to slice
            pulses for pulse-resolved pipeline data.
    """

    _intensity_ma = MovingAverageArray()
    _pulse_intensity_ma = MovingAverageArray()

    def __init__(self):
        super().__init__()

        self._sources = []
        self._pulse_slicer = slice(None, None)

        self._ma_window = 1

    def update(self):
        self._sources = self._meta.get_all_data_sources("XGM")

        cfg = self._meta.get_all(mt.GLOBAL_PROC)
        self._update_moving_average(cfg)

    def _update_moving_average(self, cfg):
        # update moving average
        if 'reset_ma_xgm' in cfg:
            # reset moving average
            del self._intensity_ma
            del self._pulse_intensity_ma
            self._meta.delete(mt.GLOBAL_PROC, 'reset_ma_xgm')

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self.__class__._intensity_ma.window = v
            self.__class__._pulse_intensity_ma.window = v

        self._ma_window = v

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
                self._intensity_ma = v
                processed.xgm.__dict__[xgm_ppts[src.property]] = \
                    self._intensity_ma

        # pipeline data
        for src in pipeline_srcs:
            v, err = self._fetch_property_data(
                tid, raw, src.name, src.property)

            if err:
                err_msgs.append(err)
            else:
                xgm_ppts = DATA_SOURCE_PROPERTIES["XGM:output"]
                self._pulse_intensity_ma = v[src.slicer]
                processed.pulse.xgm.__dict__[xgm_ppts[src.property]] = \
                    self._pulse_intensity_ma

                # apply filter
                lb, ub = src.vrange
                pidx = processed.pidx
                if not math.isinf(lb) and not math.isinf(ub):
                    for i, v in enumerate(self._pulse_intensity_ma):
                        if v > ub or v < lb:
                            pidx.mask(i)
                elif not math.isinf(lb):
                    for i, v in enumerate(self._pulse_intensity_ma):
                        if v < lb:
                            pidx.mask(i)
                elif not math.isinf(ub):
                    for i, v in enumerate(self._pulse_intensity_ma):
                        if v > ub:
                            pidx.mask(i)

        for msg in err_msgs:
            raise ProcessingError(f'[XGM] {msg}')
