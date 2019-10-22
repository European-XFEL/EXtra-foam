"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError
from ...database import Metadata as mt
from ...config import AnalysisType


class PostPulseFilter(_BaseProcessor):
    """PostPulseFilter class.

    Filter applied after pulse-resolved image analysis.

    Attributes:
        analysis_type (AnalysisType): analysis type.
        _fom_range (tuple): if the FOM falls within this range, we will
            acknowledge the data.
    """
    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED
        self._fom_range = (-np.inf, np.inf)

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.PULSE_FILTER_PROC)

        self._update_analysis(AnalysisType(int(cfg['analysis_type'])))
        self._fom_range = self.str2tuple(cfg['fom_range'])

    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        processed = data['processed']

        err_msgs = []

        if self.analysis_type == AnalysisType.ROI1_PULSE:
            foms = processed.pulse.roi.roi1.fom
            if foms is None:
                err_msgs.append("Pulse resolved ROI1 sum result is not available")
        elif self.analysis_type == AnalysisType.ROI2_PULSE:
            foms = processed.pulse.roi.roi2.fom
            if foms is None:
                err_msgs.append("Pulse resolved ROI2 sum result is not available")
        else:
            err_msgs.append("NotImplemented {repr(self.analysis_type)}")

        if err_msgs:
            raise ProcessingError(f"[Post pulse filter] {err_msgs[0]}")

        dropped = []  # a list of dropped indices
        lb, ub = self._fom_range
        for i, fom in enumerate(foms):
            if fom < lb or fom > ub:
                dropped.append(i)

        processed.image.dropped_indices.extend(dropped)
