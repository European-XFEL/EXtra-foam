"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

CorrelationProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import CompositeProcessor, _get_slow_data
from ..exceptions import ProcessingError
from ...config import config, AnalysisType
from ...metadata import Metadata as mt
from ...utils import profiler


class CorrelationProcessor(CompositeProcessor):
    """Add correlation information into processed data.

    Attributes:
        analysis_type (AnalysisType): analysis type.
        _device_ids (list): device ids for slow data correlators.
        _properties (list): properties for slow data correlators
    """

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        n_params = len(config["CORRELATION_COLORS"])
        self._device_ids = [None] * n_params
        self._properties = [None] * n_params

        self._reset = False

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.CORRELATION_PROC)

        if self._update_analysis(AnalysisType(int(cfg['analysis_type']))):
            self._reset = True

        for i in range(len(self._device_ids)):
            self._device_ids[i] = cfg[f'device_id{i+1}']
            self._properties[i] = cfg[f'property{i+1}']

    @profiler("Correlation Processor")
    def process(self, data):
        """Override."""
        processed = data['processed']

        processed.correlation.reset = self._reset
        self._reset = False

        if self.analysis_type == AnalysisType.PUMP_PROBE:
            fom = processed.pp.fom
            # Don't raise an Exception here if fom is None since it does not
            # work well if on- and off- pulses are in different trains.

        elif self.analysis_type in (AnalysisType.ROI1_ADD_ROI2,
                                    AnalysisType.ROI1_SUB_ROI2):
            fom1 = processed.roi.roi1_fom
            fom2 = processed.roi.roi2_fom

            if fom1 is None and fom2 is None:
                fom = None
            else:
                if fom1 is None:
                    fom1 = 0.0
                elif fom2 is None:
                    fom2 = 0.0

                if self.analysis_type == AnalysisType.ROI1_ADD_ROI2:
                    fom = fom1 + fom2
                else:  # self.analysis_type == AnalysisType.ROI1_SUB_ROI2:
                    fom = fom1 - fom2

        elif self.analysis_type == AnalysisType.TRAIN_AZIMUTHAL_INTEG:
            fom = processed.ai.intensity_fom
        else:
            return

        if fom is None:
            return

        # set the FOM and correlator values
        error_messages = []
        processed.correlation.fom = fom
        for i, (dev_id, ppt) in enumerate(zip(self._device_ids,
                                              self._properties)):
            if not dev_id or not ppt:
                continue

            try:
                ret = _get_slow_data(processed.tid, data['raw'], dev_id, ppt)
                setattr(processed.correlation, f'correlator{i+1}', ret)
            except ProcessingError as e:
                error_messages.append(f"[Correlation] {str(e)}")

        for msg in error_messages:
            raise ProcessingError(msg)
