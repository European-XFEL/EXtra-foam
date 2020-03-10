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

from extra_foam.pipeline.processors.correlation import (
    CorrelationProcessor, SimplePairSequence, OneWayAccuPairSequence
)

from extra_foam.config import AnalysisType

from extra_foam.pipeline.tests import _TestDataMixin


_analysis_types = [AnalysisType.PUMP_PROBE,
                   AnalysisType.ROI_FOM,
                   AnalysisType.ROI_PROJ,
                   AnalysisType.AZIMUTHAL_INTEG]


class TestCorrelationProcessor(_TestDataMixin):

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
    @pytest.mark.parametrize("index", [0, 1])
    def testGeneral(self, error, analysis_type, index):
        data, processed = self.simple_data(1001, (2, 2))
        corr = processed.corr

        slow_src = f'A{index} ppt'
        data['raw'] = {slow_src: 1}

        proc = CorrelationProcessor(index+1)
        proc.analysis_type = analysis_type

        proc._resolution = 0.0
        proc._correlation = SimplePairSequence()
        proc._correlation_slave = SimplePairSequence()

        # source is empty
        proc._source = ''
        empty_arr = np.array([], dtype=np.float64)
        proc.process(data)
        np.testing.assert_array_equal(empty_arr, corr[index].x)
        np.testing.assert_array_equal(empty_arr, corr[index].y)
        if analysis_type != AnalysisType.PUMP_PROBE:
            error.assert_called_once()
            error.reset_mock()

        # set FOM and source
        fom_gt = 10.
        self._set_fom(processed, analysis_type, fom_gt)
        proc._source = slow_src
        proc.process(data)
        assert slow_src == corr[index].source
        assert 0.0 == corr[index].resolution
        np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
        np.testing.assert_array_equal(np.array([10], dtype=np.float64), corr[index].y)

        if analysis_type == AnalysisType.PUMP_PROBE and index == 0:
            # _process_pump_probe is called only once
            assert '' == corr.pp.source
            assert 0.0 == corr.pp.resolution
            np.testing.assert_array_equal(np.array([1001]), corr.pp.x)
            np.testing.assert_array_equal(np.array([fom_gt]), corr.pp.y)

        # -------------------
        # slow data not found
        # -------------------
        data['raw'] = {}
        fom_gt = 20
        self._set_fom(processed, analysis_type, fom_gt)
        proc.process(data)
        np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
        np.testing.assert_array_equal(np.array([10], dtype=np.float64), corr[index].y)
        error.assert_called_once()
        error.reset_mock()

        # ---------------
        # new data arrive
        # ---------------
        data['raw'] = {slow_src: 2}
        fom_gt = 20
        self._set_fom(processed, analysis_type, fom_gt)
        proc.process(data)
        np.testing.assert_array_equal(np.array([1, 2], dtype=np.float64), corr[index].x)
        np.testing.assert_array_equal(np.array([10, 20], dtype=np.float64), corr[index].y)

        # -----------
        # FOM is None
        # -----------
        fom_gt = None
        self._set_fom(processed, analysis_type, fom_gt)
        proc.process(data)
        if analysis_type != AnalysisType.PUMP_PROBE:
            error.assert_called_once()
            error.reset_mock()
        else:
            assert 1 == proc._pp_fail_flag
        np.testing.assert_array_equal(np.array([1, 2], dtype=np.float64), corr[index].x)
        np.testing.assert_array_equal(np.array([10, 20], dtype=np.float64), corr[index].y)

        # again
        if analysis_type == AnalysisType.PUMP_PROBE:
            proc.process(data)
            error.assert_called_once()
            error.reset_mock()
            assert 0 == proc._pp_fail_flag

    @pytest.mark.parametrize("index", [0, 1])
    def testMaskSlave(self, index):
        data, processed = self.simple_data(1001, (2, 2))
        corr = processed.corr

        slow_src = f'A{index} ppt'
        data['raw'] = {slow_src: 1}

        proc = CorrelationProcessor(index+1)
        proc.analysis_type = AnalysisType.ROI_FOM

        # first data
        processed.roi.fom = 10
        proc._source = slow_src
        proc.process(data)
        np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
        np.testing.assert_array_equal(np.array([10], dtype=np.float64), corr[index].y)

        # second data
        processed.roi.fom = 20
        processed.roi.fom_slave = 1
        proc._source = slow_src
        proc.process(data)
        np.testing.assert_array_equal(np.array([1, 1], dtype=np.float64), corr[index].x)
        np.testing.assert_array_equal(np.array([10, 20], dtype=np.float64), corr[index].y)
        np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x_slave)
        np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].y_slave)

        # third data
        processed.roi.fom = 30
        processed.roi.fom_slave = 2
        proc._source = slow_src
        proc.process(data)
        np.testing.assert_array_equal(np.array([1, 1, 1], dtype=np.float64), corr[index].x)
        np.testing.assert_array_equal(np.array([10, 20, 30], dtype=np.float64), corr[index].y)
        np.testing.assert_array_equal(np.array([1, 1], dtype=np.float64), corr[index].x_slave)
        np.testing.assert_array_equal(np.array([1, 2], dtype=np.float64), corr[index].y_slave)

        # test reset
        proc._reset = True
        with mock.patch.object(proc._correlation_pp, "reset") as patched_reset:
            proc.process(data)
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
            np.testing.assert_array_equal(np.array([30], dtype=np.float64), corr[index].y)
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x_slave)
            np.testing.assert_array_equal(np.array([2], dtype=np.float64), corr[index].y_slave)
            # correlation_pp has another reset flag
            patched_reset.assert_not_called()
