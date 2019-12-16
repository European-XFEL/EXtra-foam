"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import pytest
from unittest import mock

import numpy as np

from extra_foam.pipeline.processors import CorrelationProcessor
from extra_foam.config import AnalysisType

from extra_foam.pipeline.processors.tests import _BaseProcessorTest


_analysis_types = [AnalysisType.PUMP_PROBE,
                   AnalysisType.ROI_FOM,
                   AnalysisType.ROI_PROJ,
                   AnalysisType.AZIMUTHAL_INTEG]


class TestCorrelationProcessor(_BaseProcessorTest):

    def _set_fom(self, processed, analysis_type, fom):
        if analysis_type == AnalysisType.PUMP_PROBE:
            processed.pp.fom = fom
        elif analysis_type == AnalysisType.ROI_FOM:
            processed.roi.fom = fom
        elif analysis_type == AnalysisType.ROI_PROJ:
            processed.roi.proj.fom = fom
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            processed.ai.fom = fom
        else:
            raise NotImplementedError

    @mock.patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("analysis_type", _analysis_types)
    def testGeneral(self, error, analysis_type):
        data, processed = self.simple_data(1001, (2, 2))
        data['raw'] = {'A': {'e': 1}}

        proc = CorrelationProcessor()
        proc.analysis_type = analysis_type
        proc._device_ids = ['A', '']
        proc._properties = ['e', '']
        proc._resolutions = [0.0, 0.0]

        empty_arr = np.array([], dtype=np.float64)

        fom_gt = 10.
        self._set_fom(processed, analysis_type, fom_gt)
        proc.process(data)

        corr = processed.corr
        assert 'A' == corr[0].device_id
        assert 'e' == corr[0].property
        assert 0.0 == corr[0].resolution
        np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[0].x)
        np.testing.assert_array_equal(np.array([10], dtype=np.float64), corr[0].y)

        assert '' == corr[1].device_id
        assert '' == corr[1].property
        assert 0.0 == corr[1].resolution
        np.testing.assert_array_equal(empty_arr, corr[1].x)
        np.testing.assert_array_equal(empty_arr, corr[1].y)

        if analysis_type == AnalysisType.PUMP_PROBE:
            assert '' == corr.pp.device_id
            assert '' == corr.pp.property
            assert 0.0 == corr.pp.resolution
            np.testing.assert_array_equal(np.array([1001]), corr.pp.x)
            np.testing.assert_array_equal(np.array([fom_gt]), corr.pp.y)

        # ---------------
        # new data arrive
        # ---------------
        proc._device_ids = ['A', 'B']
        proc._properties = ['e', 'f']
        proc._resolutions = [0.0, 0.0]

        fom_gt = 20.
        self._set_fom(processed, analysis_type, fom_gt)
        proc.process(data)
        error.assert_called_once()
        error.reset_mock()
        np.testing.assert_array_equal(np.array([1, 1], dtype=np.float64), corr[0].x)
        np.testing.assert_array_equal(np.array([10, 20], dtype=np.float64), corr[0].y)

        # ------------------------
        # set slow data source 'B'
        # ------------------------
        data['raw'] = {'A': {'e': 2}, 'B': {'f': 5}}
        proc.process(data)
        np.testing.assert_array_equal(np.array([1, 1, 2], dtype=np.float64), corr[0].x)
        np.testing.assert_array_equal(np.array([10, 20, 20], dtype=np.float64), corr[0].y)

        assert 'B' == corr[1].device_id
        assert 'f' == corr[1].property
        assert 0.0 == corr[1].resolution
        np.testing.assert_array_equal(np.array([5], dtype=np.float64), corr[1].x)
        np.testing.assert_array_equal(np.array([20], dtype=np.float64), corr[1].y)

        # -----------
        # FOM is None
        # -----------
        proc._resolutions = [1.0, 0.0]
        fom_gt = None
        self._set_fom(processed, analysis_type, fom_gt)
        proc.process(data)
        if analysis_type != AnalysisType.PUMP_PROBE:
            error.assert_called_once()
            error.reset_mock()
        else:
            assert 1 == proc._pp_fail_flag

        np.testing.assert_array_equal(np.array([1, 1, 2], dtype=np.float64), corr[0].x)
        np.testing.assert_array_equal(np.array([10, 20, 20], dtype=np.float64), corr[0].y)

        np.testing.assert_array_equal(np.array([5], dtype=np.float64), corr[1].x)
        np.testing.assert_array_equal(np.array([20], dtype=np.float64), corr[1].y)

        # again
        if analysis_type == AnalysisType.PUMP_PROBE:
            proc.process(data)
            error.assert_called_once()
            error.reset_mock()
            assert 0 == proc._pp_fail_flag
