"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

DataReductionProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError
from ...metadata import Metadata as mt
from ...config import AnalysisType
from ...utils import profiler


class DataReductionProcessor(_BaseProcessor):
    """DataReductionProcessor class.

    Apply data reduction based on the specified range of FOM.

    Attributes:
        analysis_type (AnalysisType): binning analysis type.
        _fom_range (tuple): if the FOM falls within this range, we will
            acknowledge the data.
        _pulse_resolved (bool): True for pulse resolved analysis
    """
    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED
        self._fom_range = (-np.inf, np.inf)

        self._pulse_resolved = True

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.DATA_REDUCTION_PROC)

        analysis_type = AnalysisType(int(cfg['analysis_type']))
        if self._pulse_resolved and analysis_type != AnalysisType.UNDEFINED:
            analysis_type = AnalysisType(analysis_type + AnalysisType.PULSE)
        self._update_analysis(analysis_type)

        self._fom_range = self.str2tuple(cfg['fom_range'])

    @profiler("Data reduction processor")
    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return
        processed = data['processed']

        if self._pulse_resolved:
            if self.analysis_type == AnalysisType.ROI1_PULSE:
                foms = processed.pulse.roi.roi1.fom
                if foms is None:
                    raise ProcessingError(
                        "[Data reduction]: "
                        "Pulse resolved ROI1 sum result is not available")
            elif self.analysis_type == AnalysisType.ROI2_PULSE:
                foms = processed.pulse.roi.roi2.fom
                if foms is None:
                    raise ProcessingError(
                        "[Data reduction]: "
                        "Pulse resolved ROI2 sum result is not available")
            else:
                raise NotImplementedError(
                    f'[Data reduction]: {repr(self.analysis_type)}')
        else:
            raise NotImplementedError(
                "f[Data reduction]: {repr(self.analysis_type)}")

        dropped = []  # a list of dropped indices
        lb, ub = self._fom_range[0], self._fom_range[1]
        for i, fom in enumerate(foms):
            if fom < lb or fom > ub:
                dropped.append(i)

        processed.image.dropped_indices = dropped
