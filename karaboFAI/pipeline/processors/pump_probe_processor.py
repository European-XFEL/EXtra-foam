"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeProcessors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import (
    LeafProcessor, CompositeProcessor, SharedProperty,
)
from ...config import AnalysisType
from ...metadata import Metadata as mt


class PumpProbeProcessor(CompositeProcessor):
    """PumpProbeImageExtractor.

    Extract the pump and probe images in a train.

    Attributes:
        analysis_type (AnalysisType): pump-probe analysis type.
        abs_difference (bool): True for calculating absolute different
            between on/off pulses.
        ma_window (int): moving average window size.
    """
    analysis_type = SharedProperty()
    abs_difference = SharedProperty()
    ma_window = SharedProperty()

    def __init__(self):
        super().__init__()

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.PUMP_PROBE_PROC)

        self._update_analysis(AnalysisType(int(cfg['analysis_type'])))
        self.ma_window = int(cfg['ma_window'])
        self.abs_difference = cfg['abs_difference'] == 'True'

    def process(self, data):
        processed = data['processed']

        processed.pp.analysis_type = self.analysis_type

        # setting processing parameters should come before setting data
        processed.pp.ma_window = self.ma_window
        processed.pp.abs_difference = self.abs_difference
