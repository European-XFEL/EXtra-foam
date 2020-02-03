"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from extra_foam.pipeline.processors.tests import _BaseProcessorTest
from extra_foam.pipeline.processors.bin import _BinMixin, BinProcessor
from extra_foam.config import AnalysisType, BinMode


_analysis_types = [AnalysisType.PUMP_PROBE,
                   AnalysisType.ROI_FOM,
                   AnalysisType.ROI_PROJ,
                   AnalysisType.AZIMUTHAL_INTEG]

_bin_modes = [BinMode.ACCUMULATE, BinMode.AVERAGE]


class TestBinMixin:
    def testSearchSorted(self):
        proc = _BinMixin()
        assert -1 == proc.searchsorted([], 1)
        assert -1 == proc.searchsorted([0], 1)
        assert 0 == proc.searchsorted([0, 1], 1)
        assert 1 == proc.searchsorted([0, 1, 2], 1.5)
        assert 2 == proc.searchsorted([0, 1, 2, 3], 3)
        assert 0 == proc.searchsorted([0, 1, 2, 3], 0)


class TestBinProcessor(_BaseProcessorTest):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = BinProcessor()
        yield
        self._proc._clear_history()

    @pytest.mark.parametrize("mode", _bin_modes)
    def test1dBinning(self, mode):
        proc = self._proc
        proc._mode = mode

        fom_gt = np.random.randn(10)
        proc._fom.extend(fom_gt)

        proc._slow1.extend(np.arange(10))
        proc._n_bins1 = 4
        proc._range1 = [0, 8]

        proc._new_1d_binning()
        assert not proc._bin1d
        assert proc._bin2d
        assert 4 == len(proc._stats1)
        assert [2, 2, 2, 3] == proc._counts1.tolist()
        assert [0, 2, 4, 6, 8] == proc._edges1.tolist()
        assert proc._vfom is None
        assert proc._vfom_heat1 is None
        assert proc._vfom_x1 is None
        if mode == BinMode.AVERAGE:
            assert pytest.approx(fom_gt[:9].sum()) == \
                   pytest.approx(sum([v * c for v, c in zip(proc._stats1, proc._counts1)]))
        elif mode == BinMode.ACCUMULATE:
            assert pytest.approx(fom_gt[:9].sum()) == pytest.approx(proc._stats1.sum())

        # new outsider data point
        proc._update_1d_binning(0.1, None, 9)  # index 4
        assert [2, 2, 2, 3] == proc._counts1.tolist()

        # new valid data point
        new_fom = 0.1
        new_stats1 = proc._stats1.copy()
        if mode == BinMode.AVERAGE:
            new_stats1[1] = (2 * new_stats1[1] + new_fom) / 3
        elif mode == BinMode.ACCUMULATE:
            new_stats1[1] += new_fom
        proc._update_1d_binning(new_fom, None, 3)  # index 1
        assert [2, 3, 2, 3] == proc._counts1.tolist()
        np.testing.assert_array_almost_equal(new_stats1, proc._stats1)

    @pytest.mark.parametrize("mode", _bin_modes)
    def test1dBinningWithVFom(self, mode):
        proc = self._proc

        proc._mode = mode
        proc._n_bins1 = 4
        proc._range1 = [1, 3]

        vfom = np.random.randn(10)
        vfom_x = np.arange(10)
        proc._init_vfom_binning(vfom, vfom_x)

        assert (10, 4) == proc._vfom_heat1.shape
        assert (1, 10) == proc._vfom.data().shape
        assert [1., 1.5, 2., 2.5, 3.] == proc._edges1.tolist()
        assert [0, 0, 0, 0] == proc._counts1.tolist()

        # new VFOM falls into the 3rd bin
        new_vfom = np.random.randn(10)
        new_heat1 = proc._vfom_heat1.copy()
        new_heat1[:, 3] += new_vfom
        proc._update_1d_binning(0.1, new_vfom, 2.6)
        np.testing.assert_array_equal(new_heat1, proc._vfom_heat1)

        # another new VFOM falls into the 3rd bin
        new_vfom = np.random.randn(10)
        new_heat1 = proc._vfom_heat1.copy()
        if mode == BinMode.AVERAGE:
            new_heat1[:, 3] = (new_heat1[:, 3] + new_vfom) / 2
        elif mode == BinMode.ACCUMULATE:
            new_heat1[:, 3] += new_vfom
        proc._update_1d_binning(0.1, new_vfom, 2.6)
        np.testing.assert_array_almost_equal(new_heat1, proc._vfom_heat1)

    @pytest.mark.parametrize("mode", _bin_modes)
    def test2dBinning(self, mode):
        proc = self._proc
        proc._mode = mode

        fom_gt = np.random.randn(10)
        proc._fom.extend(fom_gt)

        proc._slow1.extend(np.arange(10))
        proc._slow2.extend(np.arange(10) + 1)
        proc._n_bins1 = 4
        proc._range1 = [1, 8]
        proc._n_bins2 = 2
        proc._range2 = [2, 6]

        proc._new_2d_binning()
        assert proc._bin1d
        assert not proc._bin2d

        assert (2, 4) == proc._heat.shape
        assert (2, 4) == proc._heat_count.shape
        assert [[2, 0], [0, 2], [0, 1], [0, 0]], proc._heat_count.tolist()
        np.testing.assert_array_equal([2, 4, 6], proc._edges2)
        assert proc._edges1 is None  # calculated in _new_1d_binning
        proc._new_1d_binning()
        np.testing.assert_array_equal([1., 2.75, 4.5, 6.25, 8.], proc._edges1)
        if mode == BinMode.AVERAGE:
            assert pytest.approx(fom_gt[1:6].sum()) == \
                   pytest.approx(np.sum([v * c for v, c in zip(proc._heat, proc._heat_count)]))
        elif mode == BinMode.ACCUMULATE:
            assert pytest.approx(fom_gt[1:6].sum()) == pytest.approx(np.sum(proc._heat))

        # new outsider data point
        proc._update_2d_binning(0.1, 7, 6.5)  # index 2
        assert [[2, 0, 0, 0], [0, 2, 1, 0]] == proc._heat_count.tolist()

        # new valid data point
        new_fom = 0.1
        new_heat = proc._heat.copy()
        if mode == BinMode.AVERAGE:
            new_heat[0, 0] = (2 * new_heat[0, 0] + new_fom) / 3
        elif mode == BinMode.ACCUMULATE:
            new_heat[0, 0] += new_fom
        proc._update_2d_binning(new_fom, 1, 2)  # index 0
        assert [[3, 0, 0, 0], [0, 2, 1, 0]] == proc._heat_count.tolist()
        np.testing.assert_array_almost_equal(new_heat, proc._heat)

    def testNotTriggered(self):
        proc = self._proc
        data, processed = self.simple_data(1001, (4, 2, 2))

        # nothing should happen
        proc._get_data_point = MagicMock()
        assert proc.analysis_type == AnalysisType.UNDEFINED
        proc.process(data)
        proc._get_data_point.assert_not_called()

        proc.analysis_type = _analysis_types[0]
        proc._has_param1 = False
        proc.process(data)
        proc._get_data_point.assert_not_called()

    def _set_ret(self, processed, analysis_type, fom, vfom, vfom_x):
        if analysis_type == AnalysisType.PUMP_PROBE:
            ret = processed.pp
        elif analysis_type == AnalysisType.ROI_FOM:
            ret = processed.roi
        elif analysis_type == AnalysisType.ROI_PROJ:
            ret = processed.roi.proj
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            ret = processed.ai
        else:
            raise NotImplementedError

        ret.fom = fom
        ret.x = vfom_x
        ret.y = vfom

    @patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("analysis_type", _analysis_types)
    def testProcess1DBinning(self, error, analysis_type):
        proc = self._proc
        proc._new_2d_binning = MagicMock()
        proc._update_2d_binning = MagicMock()

        data, processed = self.simple_data(1001, (4, 2, 2))
        data['raw'] = {'A ppt': 0.1, 'B ppt': 5}

        proc.analysis_type = analysis_type
        proc._n_bins1 = 4
        proc._range1 = [-1, 1]
        proc._source1 = 'A ppt'
        proc._has_param1 = True

        fom_gt = 10
        vfom_gt = [0.5, 2.0, 1.0]
        vfom_x_gt = [1, 2, 3]
        self._set_ret(processed, analysis_type, fom_gt, vfom_gt, vfom_x_gt)

        proc.process(data)

        def assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt):
            bin = processed.bin
            assert 'A ppt' == bin[0].source
            np.testing.assert_array_equal([-0.75, -0.25, 0.25, 0.75], bin[0].centers)
            np.testing.assert_array_equal([0, 0, 1, 0], bin[0].counts)
            np.testing.assert_array_equal([0, 0, fom_gt, 0], bin[0].stats)
            np.testing.assert_array_equal(vfom_x_gt, bin[0].x)
            heat_gt = np.zeros((len(vfom_gt), len(bin[0].centers)))
            heat_gt[:, 2] = vfom_gt
            np.testing.assert_array_equal(heat_gt, bin[0].heat)

        assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt)

        # -------------------------
        # Slow data cannot be found
        # -------------------------
        data['raw'] = {'AA ppt': 0.1, 'B ppt': 5}

        proc.process(data)

        error.assert_called_once()
        error.reset_mock()

        # Test that even if the slow data is not available, the existing data
        # will be posted.
        assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt)

        # -----------
        # FOM is None
        # -----------
        data['raw'] = {'A ppt': 0.1, 'B ppt': 5}
        self._set_ret(processed, analysis_type, None, vfom_gt, vfom_x_gt)
        proc.process(data)
        if analysis_type != AnalysisType.PUMP_PROBE:
            error.assert_called_once()
            error.reset_mock()
        else:
            assert 1 == proc._pp_fail_flag

        # Test that the existing data will be posted even if the new FOM is None
        assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt)

        # again
        if analysis_type == AnalysisType.PUMP_PROBE:
            proc.process(data)
            error.assert_called_once()
            error.reset_mock()
            assert 0 == proc._pp_fail_flag

        # ----------------------
        # Length of VFOM changes
        # ----------------------
        vfom_gt = [0.5, 2.0, 1.0, 5.0]
        vfom_x_gt = [1, 2, 3, 4]
        self._set_ret(processed, analysis_type, fom_gt, vfom_gt, vfom_x_gt)
        proc.process(data)

        assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt)

        proc._n_bins1 = 2
        proc._range1 = [-2, 2]
        proc._device_id1 = 'B ppt'
        proc._has_param1 = True

        # 2D binning function not called
        proc._new_2d_binning.assert_not_called()
        proc._update_2d_binning.assert_not_called()

    @patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("analysis_type", _analysis_types)
    def testProcess2DBinning(self, error, analysis_type):
        proc = self._proc
        proc._new_2d_binning = MagicMock()
        proc._update_2d_binning = MagicMock()

        data, processed = self.simple_data(1001, (4, 2, 2))
        data['raw'] = {'A ppt': 0.1, 'B ppt': 5}

        proc.analysis_type = analysis_type
        proc._n_bins1 = 4
        proc._range1 = [-1, 1]
        proc._source1 = 'A ppt'
        proc._has_param1 = True
        proc._n_bins2 = 2
        proc._range2 = [0, 1]
        proc._source2 = 'B ppt'
        proc._has_param2 = False

        fom_gt = 10
        vfom_gt = [0.5, 2.0, 1.0, 5.0]
        vfom_x_gt = [1, 2, 3, 4]
        self._set_ret(processed, analysis_type, fom_gt, vfom_gt, vfom_x_gt)
        proc.process(data)
        proc._new_2d_binning.assert_not_called()
        proc._update_2d_binning.assert_not_called()

        # test '_has_param2'
        proc._has_param2 = True
        proc.process(data)
        proc._new_2d_binning.assert_called_once()
        proc._update_2d_binning.assert_called_once()
