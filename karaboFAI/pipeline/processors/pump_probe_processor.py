"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeProcessors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import CompositeProcessor
from ...config import AnalysisType, PumpProbeMode
from ...metadata import Metadata as mt


class PumpProbeProcessor(CompositeProcessor):
    """PumpProbeImageExtractor.

    Extract the pump and probe images in a train.

    Attributes:
        analysis_type (AnalysisType): pump-probe analysis type.
        _abs_difference (bool): True for calculating absolute different
            between on/off pulses.
        _ma_window (int): moving average window size.
    """
    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        self._mode = PumpProbeMode.UNDEFINED

        self._abs_difference = False
        self._ma_window = 1

        self._reset = False

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.PUMP_PROBE_PROC)

        if self._update_analysis(AnalysisType(int(cfg['analysis_type'])),
                                 register=False):
            self._reset = True

        mode = PumpProbeMode(int(cfg['mode']))
        if mode != self._mode:
            self._reset = True
            self._mode = mode

        self._ma_window = int(cfg['ma_window'])

        abs_difference = cfg['abs_difference'] == 'True'
        if abs_difference != self._abs_difference:
            self._reset = True
            self._abs_difference = abs_difference

        if 'reset' in cfg:
            self._meta.delete(mt.PUMP_PROBE_PROC, 'reset')
            # reset when commanded by the GUI
            self._reset = True

    def process(self, data):
        processed = data['processed']

        processed.pp.reset = self._reset
        self._reset = False

        processed.pp.analysis_type = self.analysis_type

        # setting processing parameters should come before setting data
        processed.pp.ma_window = self._ma_window
        processed.pp.abs_difference = self._abs_difference
