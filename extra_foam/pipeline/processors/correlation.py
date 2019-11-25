"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError
from ...config import AnalysisType
from ...database import Metadata as mt
from ...utils import profiler


class CorrelationProcessor(_BaseProcessor):
    """Add correlation information into processed data.

    Attributes:
        analysis_type (AnalysisType): analysis type.
        _device_ids (list): device ids for slow data correlators.
        _properties (list): properties for slow data correlators
    """

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        n_params = 2
        self._device_ids = [""] * n_params
        self._properties = [""] * n_params
        self._resolutions = [0.0] * n_params

        # used to check whether pump-probe FOM is available
        self._pp_fail_flag = 0

        self._reset = False

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.CORRELATION_PROC)

        if self._update_analysis(AnalysisType(int(cfg['analysis_type']))):
            self._reset = True

        for i in range(len(self._device_ids)):
            self._device_ids[i] = cfg[f'device_id{i+1}']
            self._properties[i] = cfg[f'property{i+1}']
            self._resolutions[i] = float(cfg[f'resolution{i+1}'])

        if 'reset' in cfg:
            self._meta.hdel(mt.CORRELATION_PROC, 'reset')
            self._reset = True

    @profiler("Correlation Processor")
    def process(self, data):
        """Override."""
        processed = data['processed']

        processed.corr.correlation1.reset = self._reset
        processed.corr.correlation2.reset = self._reset
        processed.corr.correlation3.reset = self._reset
        processed.corr.correlation4.reset = self._reset
        self._reset = False

        err_msgs = []

        analysis_type = self.analysis_type
        if analysis_type == AnalysisType.PUMP_PROBE:
            fom = processed.pp.fom
            if fom is None:
                self._pp_fail_flag += 1
                # if on/off pulses are in different trains, pump-probe FOM is
                # only calculated every other train.
                if self._pp_fail_flag == 2:
                    err_msgs.append("Pump-probe result is not available")
                    self._pp_fail_flag = 0
            else:
                self._pp_fail_flag = 0
        elif analysis_type == AnalysisType.ROI1:
            fom = processed.roi.roi1.fom
            if fom is None:
                err_msgs.append("ROI1 FOM result is not available")
        elif analysis_type == AnalysisType.ROI2:
            fom = processed.roi.roi2.fom
            if fom is None:
                err_msgs.append("ROI2 FOM result is not available")
        elif analysis_type == AnalysisType.ROI1_SUB_ROI2:
            fom = processed.roi.roi1_sub_roi2.fom
            if fom is None:
                err_msgs.append("ROI1 - ROI2 FOM result is not available")
        elif analysis_type == AnalysisType.ROI1_ADD_ROI2:
            fom = processed.roi.roi1_add_roi2.fom
            if fom is None:
                err_msgs.append("ROI1 + ROI2 FOM result is not available")
        elif analysis_type == AnalysisType.PROJ_ROI1:
            fom = processed.roi.proj1.fom
            if fom is None:
                err_msgs.append("ROI1 projection result is not available")
        elif analysis_type == AnalysisType.PROJ_ROI2:
            fom = processed.roi.proj2.fom
            if fom is None:
                err_msgs.append("ROI2 projection result is not available")
        elif analysis_type == AnalysisType.PROJ_ROI1_SUB_ROI2:
            fom = processed.roi.proj1_sub_proj2.fom
            if fom is None:
                err_msgs.append(
                    "ROI1 - ROI2 projection result is not available")
        elif analysis_type == AnalysisType.PROJ_ROI1_ADD_ROI2:
            fom = processed.roi.proj1_add_proj2.fom
            if fom is None:
                err_msgs.append(
                    "ROI1 + ROI2 projection result is not available")
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            fom = processed.ai.fom
            if fom is None:
                err_msgs.append(
                    "Azimuthal integration result is not available")
        else:  # self.analysis_type == AnalysisType.UNDEFINED
            return

        # Note: do not return or raise even if FOM is None. Otherwise, the
        #       CorrelationDataItem will be reset.

        # update correlations

        correlations = [
            processed.corr.correlation1,
            processed.corr.correlation2,
            processed.corr.correlation3,
            processed.corr.correlation4
        ]

        for corr, dev_id, ppt, res in zip(correlations,
                                          self._device_ids,
                                          self._properties,
                                          self._resolutions):
            v, err = self._fetch_property_data(processed.tid, data['raw'], dev_id, ppt)
            if err:
                err_msgs.append(err)
            corr.update_params(v, fom, dev_id, ppt, res)

        for msg in err_msgs:
            raise ProcessingError(f'[Correlation] {msg}')
