"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError, UnknownParameterError
from ...database import Metadata as mt
from ...config import AnalysisType


class FomPulseFilter(_BaseProcessor):
    """FomPulseFilter class.

    Filter pulse by pulse-resolved FOM.

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
        cfg = self._meta.hget_all(mt.FOM_FILTER_PROC)
        if cfg["pulse_resolved"] != 'True':
            self._update_analysis(AnalysisType.UNDEFINED)
            return

        analysis_type = AnalysisType(int(cfg['analysis_type']))
        if analysis_type != AnalysisType.UNDEFINED:
            self._update_analysis(AnalysisType(analysis_type + 2700))
        self._fom_range = self.str2tuple(cfg['fom_range'])

    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        processed = data['processed']

        tag = "FOM pulse filter"
        if self.analysis_type == AnalysisType.ROI_FOM_PULSE:
            fom = processed.pulse.roi.fom
            if fom is None:
                raise ProcessingError(f"[{tag}] ROI FOM is not available")

        else:
            raise UnknownParameterError(
                f"[{tag}] Unknown analysis type: {self.analysis_type}")

        self.filter_pulse_by_vrange(fom, self._fom_range, processed.pidx, tag)


class FomTrainFilter(_BaseProcessor):
    """FomTrainFilter class.

    Filter train by train-resolved FOM.

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
        cfg = self._meta.hget_all(mt.FOM_FILTER_PROC)
        if cfg["pulse_resolved"] == 'True':
            self._update_analysis(AnalysisType.UNDEFINED)
            return

        self._update_analysis(AnalysisType(int(cfg['analysis_type'])))
        self._fom_range = self.str2tuple(cfg['fom_range'])

    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        processed = data['processed']

        tag = "FOM train filter"
        if self.analysis_type == AnalysisType.ROI_FOM:
            fom = processed.roi.fom
            if fom is None:
                raise ProcessingError(f"[{tag}] ROI FOM is not available")
        else:
            raise UnknownParameterError(
                f"[{tag}] Unknown analysis type: {self.analysis_type}")

        self.filter_train_by_vrange(fom, self._fom_range, tag)
