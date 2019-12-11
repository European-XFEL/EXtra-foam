"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

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
        cfg = self._meta.hget_all(mt.PULSE_FILTER_PROC)

        self._update_analysis(AnalysisType(int(cfg['analysis_type'])))
        self._fom_range = self.str2tuple(cfg['fom_range'])

    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        processed = data['processed']

        err_msgs = []

        if self.analysis_type == AnalysisType.ROI_FOM_PULSE:
            foms = processed.pulse.roi.fom
            if foms is None:
                err_msgs.append("[Post pulse filter] "
                                "Pulse resolved ROI FOM is not available")
        else:
            err_msgs.append("NotImplemented {repr(self.analysis_type)}")

        if err_msgs:
            raise ProcessingError(f"[Post pulse filter] {err_msgs[0]}")

        pidx = processed.pidx
        lb, ub = self._fom_range
        for i, fom in enumerate(foms):
            if fom < lb or fom > ub:
                pidx.mask(i)
