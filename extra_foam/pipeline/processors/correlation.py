"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import (
    _BaseProcessor, SimplePairSequence, OneWayAccuPairSequence
)
from ..exceptions import UnknownParameterError
from ...ipc import process_logger as logger
from ...config import AnalysisType
from ...database import Metadata as mt
from ...utils import profiler


class CorrelationProcessor(_BaseProcessor):
    """Add correlation information into processed data.

    Attributes:
        analysis_type (AnalysisType): analysis type.
        _pp_analysis_type (AnalysisType): pump-probe analysis type.
        _n_params (int): number of correlators.
        _correlations (list): a list of pair sequences (SimplePairSequence,
            OneWayAccuPairSequence) for storing the history of
            (correlator, FOM).
        _device_ids (list): a list of device ids for slow data correlators.
        _properties (list): a list of properties for slow data correlators.
        _resolutions (list): a list of resolutions for correlations.
        _resets (list): reset flags for correlation data.
        _correlation_pp (SimplePairSequence): store the history of
            (correlator, FOM) which is displayed in PumpProbeWindow.
        _pp_fail_flag (int): a flag used to check whether pump-probe FOM is
            available
    """

    # 10 pulses/train * 60 seconds * 5 minutes = 3000
    _MAX_POINTS = 10 * 60 * 5

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED
        self._pp_analysis_type = AnalysisType.UNDEFINED

        self._n_params = 2
        self._correlations = []
        for i in range(self._n_params):
            self._correlations.append(
                SimplePairSequence(max_len=self._MAX_POINTS))
        self._device_ids = [""] * self._n_params
        self._properties = [""] * self._n_params
        self._resolutions = [0.0] * self._n_params
        self._resets = [False] * self._n_params

        self._correlation_pp = SimplePairSequence(max_len=self._MAX_POINTS)
        self._pp_fail_flag = 0

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.CORRELATION_PROC)

        if self._update_analysis(AnalysisType(int(cfg['analysis_type']))):
            for i in range(self._n_params):
                self._resets[i] = True

        for i in range(len(self._correlations)):
            device_id = cfg[f'device_id{i+1}']
            if self._device_ids[i] != device_id:
                self._device_ids[i] = device_id
                self._resets[i] = True

            ppt = cfg[f'property{i+1}']
            if self._properties[i] != ppt:
                self._properties[i] = ppt
                self._resets[i] = True

            resolution = float(cfg[f'resolution{i+1}'])
            if self._resolutions[i] != 0 and resolution == 0:
                self._correlations[i] = SimplePairSequence(
                    max_len=self._MAX_POINTS)
            elif self._resolutions[i] == 0 and resolution != 0:
                self._correlations[i] = OneWayAccuPairSequence(
                    resolution, max_len=self._MAX_POINTS)
            elif self._resolutions[i] != resolution:
                self._resets[i] = True
            self._resolutions[i] = resolution

        if 'reset' in cfg:
            self._meta.hdel(mt.CORRELATION_PROC, 'reset')
            for i in range(self._n_params):
                self._resets[i] = True

    @profiler("Correlation Processor")
    def process(self, data):
        """Override."""
        self._process_general(data)
        self._process_pump_probe(data)

    def _process_general(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        processed = data['processed']

        if self.analysis_type == AnalysisType.PUMP_PROBE:
            pp_analysis_type = processed.pp.analysis_type
            if self._pp_analysis_type != pp_analysis_type:
                for i in range(self._n_params):
                    self._resets[i] = True
                self._pp_analysis_type = pp_analysis_type

        for i in range(self._n_params):
            if self._resets[i]:
                self._correlations[i].reset()
                self._resets[i] = False

        analysis_type = self.analysis_type
        if analysis_type == AnalysisType.PUMP_PROBE:
            fom = processed.pp.fom
            if fom is None:
                self._pp_fail_flag += 1
                # if on/off pulses are in different trains, pump-probe FOM is
                # only calculated every other train.
                if self._pp_fail_flag == 2:
                    logger.error(
                        "[Correlation] Pump-probe FOM is not available")
                    self._pp_fail_flag = 0
            else:
                self._pp_fail_flag = 0
        elif analysis_type == AnalysisType.ROI_FOM:
            fom = processed.roi.fom
            if fom is None:
                logger.error("[Correlation] ROI FOM is not available")
        elif analysis_type == AnalysisType.ROI_PROJ:
            fom = processed.roi.proj.fom
            if fom is None:
                logger.error(
                    "[Correlation] ROI projection FOM is not available")
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            fom = processed.ai.fom
            if fom is None:
                logger.error("[Correlation] Azimuthal integration FOM is not "
                             "available")
        else:
            raise UnknownParameterError(
                f"[Correlation] Unknown analysis type: {self.analysis_type}")

        if fom is not None:
            for i in range(self._n_params):
                v, err = self._fetch_property_data(
                    processed.tid, data['raw'],
                    self._device_ids[i], self._properties[i])

                if err:
                    logger.error(err)

                if v is not None:
                    self._correlations[i].append((v, fom))

        for i in range(self._n_params):
            out = processed.corr[i]
            c = self._correlations[i]
            out.x, out.y = c.data()
            out.device_id = self._device_ids[i]
            out.property = self._properties[i]
            out.resolution = self._resolutions[i]

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
            self._correlation_pp.reset()

        if pp.fom is not None:
            self._correlation_pp.append((tid, pp.fom))

        c = processed.corr.pp
        c.x, c.y = self._correlation_pp.data()
