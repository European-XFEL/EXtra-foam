"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _FomProcessor
from ..exceptions import ProcessingError
from ...ipc import process_logger as logger
from ...algorithms import SimplePairSequence, OneWayAccuPairSequence
from ...config import AnalysisType
from ...database import Metadata as mt
from ...utils import profiler


class CorrelationProcessor(_FomProcessor):
    """Add correlation information into processed data.

    Attributes:
        _idx (int): Index of correlation starting from 1.
        analysis_type (AnalysisType): analysis type.
        _pp_analysis_type (AnalysisType): pump-probe analysis type.
        _raw (SimplePairSequence): sequence which stores the history of
            (slow, FOM).
        _corr (_AbstractSequence): SimplePairSequence/OneWayAccuPairSequence
            which stores the master correlation data.
        _raw_slave (SimplePairSequence): sequence which stores the
            history of (slow, FOM slave).
        _corr_slave (_AbstractSequence): SimplePairSequence/OneWayAccuPairSequence
            which stores the slave correlation data.
        _source: source of slow data.
        _resolution: resolution of correlation.
        _reset: reset flag for correlation data.
        _auto_reset_ma: automatically reset moving average when the scan jumps to
            the next point in the scan mode. Only apply to correlation 1.
        _corr_pp (SimplePairSequence): sequence which stores the history of
            (slow, pump-probe FOM).
        _pp_fail_flag (int): a flag used to check whether pump-probe FOM is
            available
    """

    # 10 pulses/train * 60 seconds * 5 minutes = 3000
    _MAX_POINTS = 10 * 60 * 5

    def __init__(self, index):
        super().__init__("Correlation")

        self._idx = index

        self.analysis_type = AnalysisType.UNDEFINED

        self._raw = SimplePairSequence(max_len=self._MAX_POINTS)
        self._raw_slave = SimplePairSequence(max_len=self._MAX_POINTS)
        self._corr = self._raw
        self._corr_slave = self._raw_slave
        self._source = ""
        self._resolution = 0.0
        self._reset = False

        self._auto_reset_ma = False

        self._prepare_reset_ma = False

        self._corr_pp = SimplePairSequence(max_len=self._MAX_POINTS)

    def update(self):
        """Override."""
        g_cfg, cfg = self._meta.hget_all_multi(
            [mt.GLOBAL_PROC, mt.CORRELATION_PROC])
        if 'analysis_type' not in cfg:
            # CorrelationWindow not initialized
            return

        idx = self._idx

        if self._update_analysis(AnalysisType(int(cfg['analysis_type']))):
            self._reset = True

        src = cfg[f'source{idx}']
        if self._source != src:
            self._source = src
            self._reset = True

        resolution = float(cfg[f'resolution{idx}'])
        if self._resolution != resolution:
            if resolution == 0:
                self._corr = self._raw
                self._corr_slave = self._raw_slave
            else:
                self._corr = OneWayAccuPairSequence.from_array(
                    *self._raw.data(), resolution,
                    max_len=self._MAX_POINTS)
                self._corr_slave = OneWayAccuPairSequence.from_array(
                    *self._raw_slave.data(), resolution,
                    max_len=self._MAX_POINTS)
            self._resolution = resolution

        reset_key = f'reset{idx}'
        if reset_key in cfg:
            self._meta.hdel(mt.CORRELATION_PROC, reset_key)
            self._reset = True

        if self._idx == 1 and int(g_cfg['ma_window']) > 1:
            self._auto_reset_ma = cfg["auto_reset_ma"] == 'True'
        else:
            self._auto_reset_ma = False

    @profiler("Correlation Processor")
    def process(self, data):
        """Override."""
        self._process_general(data)
        if self._idx == 1:
            # process only once
            self._process_pump_probe(data)

    def _process_general(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        processed, raw = data['processed'], data['raw']

        if self.analysis_type == AnalysisType.PUMP_PROBE:
            pp_analysis_type = processed.pp.analysis_type
            if self._pp_analysis_type != pp_analysis_type:
                # reset if pump-pobe analysis type changes
                self._reset = True
                self._pp_analysis_type = pp_analysis_type

        if self._reset:
            self._raw.reset()
            self._raw_slave.reset()
            self._corr.reset()
            self._corr_slave.reset()
            self._prepare_reset_ma = False
            self._reset = False

        try:
            reset_ma = self._update_data_point(processed, raw)
            if self._auto_reset_ma:
                data['reset_ma'] = reset_ma

        except ProcessingError as e:
            logger.error(f"[{self._name}] {str(e)}!")

        out = processed.corr[self._idx - 1]
        out.x, out.y = self._corr.data()
        out.x_slave, out.y_slave = self._corr_slave.data()
        out.source = self._source
        out.resolution = self._resolution

    def _process_pump_probe(self, data):
        """Process the correlation in pump-probe analysis.

        This is completely decoupled from the general correlation analysis.
        It produces data for the correlation plot inside the pump-probe
        window.
        """
        processed = data['processed']
        tid = processed.tid
        pp = processed.pp

        if pp.reset:
            self._corr_pp.reset()

        if pp.fom is not None:
            self._corr_pp.append((tid, pp.fom))
        elif pp.analysis_type in self._ai_analysis_types:
            self._corr_pp.append((tid, self._extract_ai_fom(pp, pp.analysis_type)))

        c = processed.corr.pp
        c.x, c.y = self._corr_pp.data()

    def _update_data_point(self, processed, raw):
        _, fom, fom_slave = self._extract_fom(processed)
        if fom is None:
            return

        v, err = self._fetch_property_data(processed.tid, raw, self._source)

        if err:
            logger.error(err)

        reset_ma = False
        if v is not None:
            self._raw.append((v, fom))

            resolution = self._resolution
            if resolution > 0:
                if self._auto_reset_ma:
                    next_pos = self._corr.append_dry(v)
                    if next_pos and not self._prepare_reset_ma:
                        self._prepare_reset_ma = True
                        reset_ma = True
                    else:
                        self._prepare_reset_ma = False
                        self._corr.append((v, fom))
                else:
                    self._corr.append((v, fom))
                    self._prepare_reset_ma = False

            if fom_slave is not None:
                self._raw_slave.append((v, fom_slave))
                if resolution > 0 and not reset_ma:
                    self._corr_slave.append((v, fom_slave))

            # previous code block should not raise
            self._last_v = v

        return reset_ma
