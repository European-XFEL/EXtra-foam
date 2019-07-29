"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

StatisticsProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque
import numpy as np

from .base_processor import CompositeProcessor
from ..exceptions import ProcessingError
from ...metadata import Metadata as mt
from ...config import AnalysisType
from ...utils import profiler


class StatisticsProcessor(CompositeProcessor):
    """StatisticsProcessor class.

    - Register the pulse resolved analysis in order to monitor the FOM of
    each pulse in a train.
    - Calculate histogram of FOMs.

    Attributes:
        analysis_type (AnalysisType): binning analysis type.
        _fom (deque): deque for storing FOMs accumulated over trains.
        _reset (bool): True for clearing _fom when analysis type changes
        _num_bins (int): number of bins
        _pulse_resolved (bool): True for pulse resolved analysis
    """
    _MAX_FOM_SIZE = 250000
    # 128 pulses/s * 60 seconds * 30 minutes = 153600
    # So 250000 is sufficient.
    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED
        self._fom = deque(maxlen=self._MAX_FOM_SIZE)
        self._reset = False
        self._num_bins = None
        self._pulse_resolved = False

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.STATISTICS_PROC)
        self._pulse_resolved = cfg['pulse_resolved'] == 'True'

        self._num_bins = int(cfg['n_bins'])

        analysis_type = AnalysisType(int(cfg['analysis_type']))
        if self._pulse_resolved and analysis_type != AnalysisType.UNDEFINED:
            analysis_type = AnalysisType(int(cfg['analysis_type']) + 2700)

        if self._update_analysis(analysis_type):
            self._reset = True

        if 'reset' in cfg:
            self._meta.delete(mt.STATISTICS_PROC, 'reset')
            self._reset = True

    @profiler("Statistics processor")
    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return
        processed = data['processed']

        if self._reset:
            self._fom.clear()
            self._reset = False

        foms = None
        if self._pulse_resolved:
            if self.analysis_type == AnalysisType.ROI1_PULSE:
                foms = processed.pulse.roi.roi1.fom
                if foms is None:
                    raise ProcessingError(
                        "Pulse resolved ROI1 sum result is not available")
            elif self.analysis_type == AnalysisType.AZIMUTHAL_INTEG_PULSE:
                foms = processed.pulse.ai.fom
                if foms is None:
                    raise ProcessingError(
                        "Pulse resolved azimuthal int. result is not available")
        else:
            if self.analysis_type == AnalysisType.ROI1:
                foms = processed.roi.roi1.fom
                if foms is None:
                    raise ProcessingError("ROI1 sum result is not available")
            elif self.analysis_type == AnalysisType.AZIMUTHAL_INTEG:
                foms = processed.ai.fom
                if foms is None:
                    raise ProcessingError(
                        "Azimuthal int. result is not available")

        if foms is not None:
            processed.st.fom_hist = foms if self._pulse_resolved else None
            if isinstance(foms, list):
                self._fom.extend(foms)
            else:
                self._fom.append(foms)

        hist, bins_edges = np.histogram(self._fom, bins=self._num_bins)
        bins_center = (bins_edges[1:]+bins_edges[:-1])/2.0
        processed.st.fom_bin_center = bins_center
        processed.st.fom_counts = hist

