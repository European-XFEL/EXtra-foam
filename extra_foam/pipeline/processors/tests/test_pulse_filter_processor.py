"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest

from extra_foam.pipeline.processors import PostPulseFilter
from extra_foam.pipeline.exceptions import ProcessingError
from extra_foam.config import AnalysisType
from extra_foam.pipeline.processors.tests import _BaseProcessorTest


class TestPulseFilters(unittest.TestCase, _BaseProcessorTest):
    def testPostPulseFilter(self):
        proc = PostPulseFilter()

        # Note: sequence of the test should be the opposite of the sequence
        #       of "if elif else" in the 'process' method

        # AZIMUTHAL_INTEG
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.AZIMUTHAL_INTEG_PULSE
        with self.assertRaises(ProcessingError):
            proc.process(data)

        # ROI FOM
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.ROI_FOM_PULSE
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertListEqual([], processed.pidx.dropped_indices(4).tolist())
        processed.pulse.roi.fom = [1, 2, 3, 4]
        proc.process(data)
        self.assertListEqual([], processed.pidx.dropped_indices(4).tolist())
        proc._fom_range = [0, 2.5]
        proc.process(data)
        self.assertListEqual([2, 3], processed.pidx.dropped_indices(4).tolist())

        # UNDEFINED
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.UNDEFINED
        proc.process(data)
        self.assertListEqual([], processed.pidx.dropped_indices(4).tolist())
