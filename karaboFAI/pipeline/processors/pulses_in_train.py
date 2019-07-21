"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PulsesInTrainProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import CompositeProcessor
from ...metadata import Metadata as mt
from ...config import AnalysisType


class PulsesInTrainProcessor(CompositeProcessor):
    """PulsesInTrainProcessor class.

    Register the pulse resolved analysis in order to monitor the FOM of
    each pulse in a train.

    Attributes:
        analysis_type (AnalysisType): binning analysis type.
    """

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.PULSE_FOM_PROC)

        self._update_analysis(AnalysisType(int(cfg['analysis_type'])))
