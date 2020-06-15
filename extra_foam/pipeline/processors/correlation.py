"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError, UnknownParameterError
from ...ipc import process_logger as logger
from ...algorithms import SimplePairSequence, OneWayAccuPairSequence
from ...config import AnalysisType
from ...database import Metadata as mt
from ...utils import profiler


class CorrelationProcessor(_BaseProcessor):
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
        _corr_pp (SimplePairSequence): sequence which stores the history of
            (slow, pump-probe FOM).
        _pp_fail_flag (int): a flag used to check whether pump-probe FOM is
            available
    """

    # 10 pulses/train * 60 seconds * 5 minutes = 3000
    _MAX_POINTS = 10 * 60 * 5

    def __init__(self, index):
        super().__init__()

        self._idx = index

        self.analysis_type = AnalysisType.UNDEFINED
        self._pp_analysis_type = AnalysisType.UNDEFINED

        self._raw = SimplePairSequence(max_len=self._MAX_POINTS)
        self._raw_slave = SimplePairSequence(max_len=self._MAX_POINTS)
        self._corr = self._raw
        self._corr_slave = self._raw_slave
        self._source = ""
        self._resolution = 0.0
        self._reset = False

        self._corr_pp = SimplePairSequence(max_len=self._MAX_POINTS)
        self._pp_fail_flag = 0

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.CORRELATION_PROC)
        if not cfg:
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
            self._reset = False

        try:
            self._update_data_point(processed, raw)
        except ProcessingError as e:
            logger.error(f"[Correlation] {str(e)}!")

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

        c = processed.corr.pp
        c.x, c.y = self._corr_pp.data()

    def _update_data_point(self, processed, raw):
        analysis_type = self.analysis_type
        fom_slave = None
        if analysis_type == AnalysisType.PUMP_PROBE:
            fom = processed.pp.fom
            if fom is None:
                self._pp_fail_flag += 1
                # if on/off pulses are in different trains, pump-probe FOM is
                # only calculated every other train.
                if self._pp_fail_flag == 2:
                    self._pp_fail_flag = 0
                    raise ProcessingError("Pump-probe FOM is not available")
                return
            else:
                self._pp_fail_flag = 0
        elif analysis_type == AnalysisType.ROI_FOM:
            fom = processed.roi.fom
            fom_slave = processed.roi.fom_slave
            if fom is None:
                raise ProcessingError("ROI FOM is not available")
        elif analysis_type == AnalysisType.ROI_PROJ:
            fom = processed.roi.proj.fom
            if fom is None:
                raise ProcessingError("ROI projection FOM is not available")
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            fom = processed.ai.fom
            if fom is None:
                raise ProcessingError(
                    "Azimuthal integration FOM is not available")
        else:
            raise UnknownParameterError(
                f"[Correlation] Unknown analysis type: {self.analysis_type}")

        v, err = self._fetch_property_data(processed.tid, raw, self._source)

        if err:
            logger.error(err)

        if v is not None:
            self._raw.append((v, fom))
            if self._resolution > 0:
                self._corr.append((v, fom))
            if fom_slave is not None:
                self._raw_slave.append((v, fom_slave))
                if self._resolution > 0:
                    self._corr_slave.append((v, fom_slave))
