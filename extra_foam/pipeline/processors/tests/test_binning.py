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

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.processors.binning import _BinMixin, BinningProcessor
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


class TestBinningProcessor(_TestDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = BinningProcessor()
        yield
        self._proc._clear_history()

    @pytest.mark.parametrize("mode", _bin_modes)
    def test1dBinning(self, mode):
        proc = self._proc
        proc._mode = mode

        # bin1d with 10 data points
        fom_gt = np.random.randn(10)
        proc._fom.extend(fom_gt)
        proc._slow1.extend(np.arange(10))
        proc._n_bins1 = 4
        proc._actual_range1 = [0, 8]
        proc._new_1d_binning()
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
        assert [2, 2, 2, 3] == proc._counts1.tolist()  # remain unchanged

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

        vfom = np.random.randn(10)  # this is VFOM for a single point
        vfom_x = np.arange(10)
        # caveat: sequence
        proc._init_vfom_binning(vfom, vfom_x)
        proc._fom.append(0)  # any single FOM
        proc._slow1.append(2.2)  # index 2
        proc._n_bins1 = 4
        proc._actual_range1 = (1, 3)
        proc._new_1d_binning()
        assert (10, 4) == proc._vfom_heat1.shape
        assert (1, 10) == proc._vfom.data().shape
        assert [1., 1.5, 2., 2.5, 3.] == proc._edges1.tolist()
        assert [0, 0, 1, 0] == proc._counts1.tolist()

        # new VFOM
        vfom = np.random.randn(10)
        vfom_heat1_gt = proc._vfom_heat1.copy()
        vfom_heat1_gt[:, 3] += vfom
        fom, slow1 = 0.1, 2.6  # index 3
        proc._update_1d_binning(fom, vfom, slow1)
        np.testing.assert_array_equal(vfom_heat1_gt, proc._vfom_heat1)

        # another new VFOM
        vfom = np.random.randn(10)
        vfom_heat1_gt = proc._vfom_heat1.copy()
        if mode == BinMode.AVERAGE:
            vfom_heat1_gt[:, 3] = (vfom_heat1_gt[:, 3] + vfom) / 2
        elif mode == BinMode.ACCUMULATE:
            vfom_heat1_gt[:, 3] += vfom
        fom, slow1 = 0.2, 2.7  # index 3
        proc._update_1d_binning(fom, vfom, slow1)
        np.testing.assert_array_almost_equal(vfom_heat1_gt, proc._vfom_heat1)

    @pytest.mark.parametrize("mode", _bin_modes)
    def test2dBinning(self, mode):
        proc = self._proc
        proc._mode = mode

        # bin2d with 10 data points
        fom_gt = np.random.randn(10)
        proc._fom.extend(fom_gt)
        proc._slow1.extend(np.arange(10))
        proc._slow2.extend(np.arange(10) + 1)
        proc._n_bins1 = 4
        proc._actual_range1 = [1, 8]
        proc._n_bins2 = 2
        proc._actual_range2 = [2, 6]
        proc._new_2d_binning()
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
        fom, slow1, slow2 = 0.1, 7, 6.5   # index (4, 2)
        proc._update_2d_binning(fom, slow1, slow2)
        assert [[2, 0, 0, 0], [0, 2, 1, 0]] == proc._heat_count.tolist()

        # new valid data point
        fom, slow1, slow2 = 0.1, 1, 2  # index (0, 0)
        heat_gt = proc._heat.copy()
        if mode == BinMode.AVERAGE:
            heat_gt[0, 0] = (2 * heat_gt[0, 0] + fom) / 3
        elif mode == BinMode.ACCUMULATE:
            heat_gt[0, 0] += fom
        proc._update_2d_binning(fom, slow1, slow2)
        assert [[3, 0, 0, 0], [0, 2, 1, 0]] == proc._heat_count.tolist()
        np.testing.assert_array_almost_equal(heat_gt, proc._heat)

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

    def testUpdate(self):
        proc = self._proc
        proc._meta.hget_all = MagicMock(return_value={'analysis_type': "1"})

        assert not proc._reset
        proc._update_analysis = MagicMock(return_value=True)
        with pytest.raises(KeyError):
            proc.update()
        assert proc._reset
        proc._update_analysis = MagicMock(return_value=False)

        get_cfg = proc._meta.hget_all
        get_cfg.return_value.update({
            'mode': str(int(BinMode.AVERAGE)),
            'source1': 'A ppt',
            'n_bins1': '4',
            'bin_range1': '(-1, 1)',
            'source2': 'B ppt',
            'n_bins2': '10',
            'bin_range2': '(-inf, inf)',
        })
        proc.update()

        # reset when source1 changes
        get_cfg.return_value.update({"source1": 'A ppt 2'})
        proc._reset = False
        proc.update()
        assert proc._reset

        # reset when source2 changes
        get_cfg.return_value.update({"source2": 'B ppt 2'})
        proc._reset = False
        proc.update()
        assert proc._reset

        # only reset 2d binning when source2 was unset
        get_cfg.return_value.update({"source2": ''})
        proc._reset = False
        proc.update()
        assert not proc._reset
        assert proc._reset_bin2d

        # reset when the sub-analysis type of pump-probe of analysis changes
        proc.analysis_type = AnalysisType.PUMP_PROBE
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc._pp_analysis_type = AnalysisType.ROI_PROJ
        processed.pp.analysis_type = AnalysisType.AZIMUTHAL_INTEG
        with patch.object(proc, "_clear_history") as mk:
            proc.process(data)
            mk.assert_called_once()

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
        def assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt):
            bin = processed.bin
            assert 'A ppt' == bin[0].source
            np.testing.assert_array_equal([-0.75, -0.25, 0.25, 0.75], bin[0].centers)
            np.testing.assert_array_equal([0, 0, 1, 0], bin[0].counts)
            np.testing.assert_array_equal([0, 0, fom_gt, 0], bin[0].stats)
            np.testing.assert_array_equal(vfom_x_gt, bin[0].x)
            vfom_heat_gt = np.zeros((len(vfom_gt), len(bin[0].centers)))
            vfom_heat_gt[:, 2] = vfom_gt
            np.testing.assert_array_equal(vfom_heat_gt, bin[0].heat)
            assert 0.5 == bin[0].size

        proc = self._proc
        proc._new_2d_binning = MagicMock()
        proc._update_2d_binning = MagicMock()

        data, processed = self.simple_data(1001, (4, 2, 2))
        data['raw'] = {'A ppt': 0.1, 'B ppt': 5}

        proc._meta.hget_all = MagicMock()
        proc._update_analysis = MagicMock(return_value=False)
        get_cfg = proc._meta.hget_all

        get_cfg.return_value = {
            'mode': str(int(BinMode.AVERAGE)),
            'analysis_type': str(int(analysis_type)),
            'source1': 'A ppt',
            'n_bins1': '4',
            'bin_range1': '(-1, 1)',
            'source2': '',
            'n_bins2': '10',
            'bin_range2': '(-inf, inf)',
        }

        proc.analysis_type = analysis_type

        # new data point
        fom_gt = 10
        vfom_gt = [0.5, 2.0, 1.0]
        vfom_x_gt = [1, 2, 3]
        self._set_ret(processed, analysis_type, fom_gt, vfom_gt, vfom_x_gt)
        assert proc._actual_range1 is None
        # specify source 1
        proc.update()
        proc.process(data)
        assert proc._actual_range1 == proc._bin_range1  # actual range1 was calculated
        assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt)

        # slow data cannot be found
        data['raw'] = {'AA ppt': 0.1, 'B ppt': 5}
        proc.process(data)
        error.assert_called_once()
        error.reset_mock()
        # even if the slow data is not available, the existing data
        # will be posted.
        assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt)

        # FOM is None
        data['raw'] = {'A ppt': 0.1, 'B ppt': 5}
        self._set_ret(processed, analysis_type, None, vfom_gt, vfom_x_gt)
        proc.process(data)
        if analysis_type != AnalysisType.PUMP_PROBE:
            error.assert_called_once()
            error.reset_mock()
        else:
            assert 1 == proc._pp_fail_flag
        # the existing data will be posted even if the new FOM is None
        assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt)

        # again only for pump-probe analysis
        if analysis_type == AnalysisType.PUMP_PROBE:
            proc.process(data)
            error.assert_called_once()
            error.reset_mock()
            assert 0 == proc._pp_fail_flag

        # Length of VFOM changes
        vfom_gt = [0.5, 2.0, 1.0, 5.0]
        vfom_x_gt = [1, 2, 3, 4]
        self._set_ret(processed, analysis_type, fom_gt, vfom_gt, vfom_x_gt)
        proc.process(data)
        assert_ret1(processed, fom_gt, vfom_gt, vfom_x_gt)

        # 2D binning function should have not been called
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

        proc._meta.hget_all = MagicMock()
        proc._update_analysis = MagicMock(return_value=False)
        get_cfg = proc._meta.hget_all

        get_cfg.return_value = {
            'mode': str(int(BinMode.AVERAGE)),
            'analysis_type': str(int(analysis_type)),
            'source1': 'A ppt',
            'n_bins1': '4',
            'bin_range1': '(-1, 1)',
            'source2': '',
            'n_bins2': '2',
            'bin_range2': '(0, 1)',
        }
        fom_gt = 10
        vfom_gt = [0.5, 2.0, 1.0, 5.0]
        vfom_x_gt = [1, 2, 3, 4]
        self._set_ret(processed, analysis_type, fom_gt, vfom_gt, vfom_x_gt)

        # without param2, bin2d will not be processed
        proc.update()
        proc.process(data)
        assert proc._actual_range2 is None
        proc._new_2d_binning.assert_not_called()
        proc._update_2d_binning.assert_not_called()
        # the first time new 2d binning was created
        get_cfg.return_value.update({'source2': 'B ppt'})
        proc.update()
        proc.process(data)
        assert proc._actual_range2 == proc._bin_range2
        proc._new_2d_binning.assert_called_once()
        proc._update_2d_binning.assert_not_called()

        # the second time only the existing 2d binning was updated
        proc.process(data)
        proc._update_2d_binning.assert_called_once()

    @patch('extra_foam.ipc.ProcessLogger.error')
    def testAutoBinRange(self, error):
        proc = self._proc

        data, processed = self.simple_data(1001, (4, 2, 2))
        data['raw'] = {'A ppt': 0.1, 'B ppt': 5}

        proc._meta.hget_all = MagicMock()
        proc._update_analysis = MagicMock(return_value=False)
        get_cfg = proc._meta.hget_all

        get_cfg.return_value = {
            'mode': str(int(BinMode.AVERAGE)),
            'analysis_type': str(int(AnalysisType.AZIMUTHAL_INTEG)),
            'source1': 'A ppt',
            'n_bins1': '4',
            'bin_range1': '(-inf, inf)',
            'source2': '',
            'n_bins2': '10',
            'bin_range2': '(-inf, inf)',
        }

        proc.analysis_type = AnalysisType.AZIMUTHAL_INTEG

        # test if the number of data points is 0, it will not raise
        proc.update()
        proc.process(data)
        assert proc._auto_range1 == [True, True]

        # first data point, actual range1 was set
        fom_gt = 10
        self._set_ret(processed, proc.analysis_type, fom_gt, None, None)
        proc.process(data)
        assert proc._auto_range1 == [True, True]
        assert proc._actual_range1 == (-0.4, 0.6)

        # second data point, actual range1 was updated
        data['raw'] = {'A ppt': 0.2, 'B ppt': 5}
        proc.process(data)
        assert proc._auto_range1 == [True, True]
        assert proc._actual_range1 == (0.1, 0.2)

        # corner case 1
        get_cfg.return_value.update({
            'bin_range1': '(0.3, inf)',
        })
        proc.update()
        proc.process(data)
        assert proc._auto_range1 == [False, True]
        assert proc._actual_range1 == (0.3, 1.3)

        # corner case 2
        get_cfg.return_value.update({
            'bin_range1': '(-inf, 0.0)',
        })
        proc.update()
        proc.process(data)
        assert proc._auto_range1 == [True, False]
        assert proc._actual_range1 == (-1.0, 0.0)

        # source 2 was set
        get_cfg.return_value.update({
            'source2': 'B ppt',
            'n_bins2': '5',
            'bin_range2': '(-inf, inf)',
        })
        proc.update()
        proc.process(data)
        assert proc._auto_range2 == [True, True]
        assert proc._actual_range2 == (4.5, 5.5)

        # bin range 2 was changed
        get_cfg.return_value.update({'bin_range2': '(-inf, 10)'})
        data['raw'] = {'A ppt': 0.2, 'B ppt': 6}
        proc.update()
        proc.process(data)
        assert proc._auto_range2 == [True, False]
        assert proc._actual_range2 == (5.0, 10)

        # bin range 2 was changed again
        get_cfg.return_value.update({'bin_range2': '(7, 10)'})
        data['raw'] = {'A ppt': 0.2, 'B ppt': 8}
        proc.update()
        proc.process(data)
        assert proc._auto_range2 == [False, False]
        assert proc._actual_range2 == (7.0, 10)
