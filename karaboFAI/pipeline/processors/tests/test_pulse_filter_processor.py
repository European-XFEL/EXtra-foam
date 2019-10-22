"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from unittest.mock import MagicMock

import numpy as np

from karaboFAI.pipeline.processors import PostPulseFilter
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.config import AnalysisType
from karaboFAI.pipeline.processors.tests import _BaseProcessorTest


class TestPulseFilters(_BaseProcessorTest):
    def testPostPulseFilter(self):
        proc = PostPulseFilter()

        # Note: sequence of the test should be the opposite of the sequence
        #       of "if elif else" in the 'process' method

        # AZIMUTHAL_INTEG
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.AZIMUTHAL_INTEG_PULSE
        with self.assertRaises(ProcessingError):
            proc.process(data)

        # ROI2
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.ROI1_PULSE
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertListEqual([], processed.pidx.dropped_indices(4).tolist())
        processed.pulse.roi.roi1.fom = [1, 2, 3, 4]
        proc.process(data)
        self.assertListEqual([], processed.pidx.dropped_indices(4).tolist())
        proc._fom_range = [0, 2.5]
        proc.process(data)
        self.assertListEqual([2, 3], processed.pidx.dropped_indices(4).tolist())

        # ROI1
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.ROI2_PULSE
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertListEqual([], processed.pidx.dropped_indices(4).tolist())
        processed.pulse.roi.roi2.fom = [4, 5, 6, 7]
        proc._fom_range = [0, 2.5]
        proc.process(data)
        self.assertListEqual([0, 1, 2, 3], processed.pidx.dropped_indices(4).tolist())

        # UNDEFINED
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.UNDEFINED
        proc.process(data)
        self.assertListEqual([], processed.pidx.dropped_indices(4).tolist())
