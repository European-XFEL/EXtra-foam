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

    def testResolutionSwitch(self):
        proc = CorrelationProcessor(1)

        with mock.patch.object(proc, "_update_analysis", return_value=False):
            with mock.patch.object(proc._meta, "hget_all_multi") as patched_cfg:
                g_cfg = {
                    "ma_window": '1',
                }
                cfg = {
                    "analysis_type": '1',
                    "source1": "ABC abc",
                    "resolution1": '0',
                    "auto_reset_ma": 'True',
                }
                patched_cfg.return_value = g_cfg, cfg
                proc.update()
                assert proc._corr is proc._raw
                assert proc._corr_slave is proc._raw_slave

                proc._reset = False

                proc._raw.extend([(0, 1), (1, 2), (2, 3)])
                proc._raw_slave.extend([(1, 2), (2, 3), (3, 4)])

                cfg['resolution1'] = '1'
                proc.update()
                np.testing.assert_array_equal([0, 1, 2], proc._raw.data()[0])
                np.testing.assert_array_equal([1, 2, 3], proc._raw.data()[1])
                np.testing.assert_array_equal([1, 2, 3], proc._raw_slave.data()[0])
                np.testing.assert_array_equal([2, 3, 4], proc._raw_slave.data()[1])
                assert isinstance(proc._corr, OneWayAccuPairSequence)
                assert isinstance(proc._corr_slave, OneWayAccuPairSequence)
                np.testing.assert_array_equal([0.5], proc._corr.data()[0])
                np.testing.assert_array_equal([1.5], proc._corr_slave.data()[0])

                cfg['resolution1'] = '4'
                proc.update()
                assert isinstance(proc._corr, OneWayAccuPairSequence)
                assert isinstance(proc._corr_slave, OneWayAccuPairSequence)
                np.testing.assert_array_equal([1.0], proc._corr.data()[0])
                np.testing.assert_array_equal([2.0], proc._corr_slave.data()[0])

                cfg['resolution1'] = '0'
                proc.update()
                assert proc._corr is proc._raw
                assert proc._corr_slave is proc._raw_slave
                np.testing.assert_array_equal([0, 1, 2], proc._corr.data()[0])
                np.testing.assert_array_equal([1, 2, 3], proc._corr.data()[1])
                np.testing.assert_array_equal([1, 2, 3], proc._corr_slave.data()[0])
                np.testing.assert_array_equal([2, 3, 4], proc._corr_slave.data()[1])

                assert not proc._reset

    @mock.patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("analysis_type", _analysis_types)
    @pytest.mark.parametrize("index", [0, 1])
    @pytest.mark.parametrize("resolution", [0, 2])
    def testProcess(self, error, analysis_type, index, resolution):
        data, processed = self.simple_data(1001, (2, 2))
        corr = processed.corr

        slow_src = f'A{index} ppt'
        data['raw'] = {slow_src: 1}

        proc = CorrelationProcessor(index+1)
        proc.analysis_type = analysis_type
        proc._resolution = resolution
        assert not proc._auto_reset_ma
        if resolution > 0:
            proc._corr = OneWayAccuPairSequence(resolution)
            proc._corr_slave = OneWayAccuPairSequence(resolution)

        # source is empty
        proc._source = ''
        proc.process(data)
        assert len(corr[index].x) == 0
        if resolution == 0:
            assert len(corr[index].y) == 0
        else:
            assert len(corr[index].y[0]) == 0
        if analysis_type != AnalysisType.PUMP_PROBE:
            error.assert_called_once()
            error.reset_mock()

        # set FOM and source
        fom_gt = 10.
        self._set_fom(processed, analysis_type, fom_gt)
        proc._source = slow_src
        proc.process(data)
        assert slow_src == corr[index].source
        if resolution == 0:
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
            np.testing.assert_array_equal(np.array([10], dtype=np.float64), corr[index].y)
        else:
            assert len(corr[index].x) == 0
            assert len(corr[index].y[0]) == 0

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
        if resolution == 0:
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
            np.testing.assert_array_equal(np.array([10], dtype=np.float64), corr[index].y)
        else:
            assert len(corr[index].x) == 0
            assert len(corr[index].y[0]) == 0
        error.assert_called_once()
        error.reset_mock()

        # ---------------
        # new data arrive
        # ---------------
        data['raw'] = {slow_src: 2}
        fom_gt = 20
        self._set_fom(processed, analysis_type, fom_gt)
        proc.process(data)
        if resolution == 0:
            np.testing.assert_array_equal(np.array([1, 2], dtype=np.float64), corr[index].x)
            np.testing.assert_array_equal(np.array([10, 20], dtype=np.float64), corr[index].y)
        else:
            np.testing.assert_array_equal(np.array([1.5], dtype=np.float64), corr[index].x)
            assert len(corr[index].y[0]) == 1

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
        if resolution == 0:
            np.testing.assert_array_equal(np.array([1, 2], dtype=np.float64), corr[index].x)
            np.testing.assert_array_equal(np.array([10, 20], dtype=np.float64), corr[index].y)
        else:
            np.testing.assert_array_equal(np.array([1.5], dtype=np.float64), corr[index].x)
            assert len(corr[index].y[0]) == 1

        # again
        if analysis_type == AnalysisType.PUMP_PROBE:
            proc.process(data)
            error.assert_called_once()
            error.reset_mock()
            assert 0 == proc._pp_fail_flag

    @pytest.mark.parametrize("index", [0, 1])
    @pytest.mark.parametrize("resolution", [0, 2])
    def testMasterSlave(self, index, resolution):
        data, processed = self.simple_data(1001, (2, 2))
        corr = processed.corr

        slow_src = f'A{index} ppt'
        data['raw'] = {slow_src: 1}

        proc = CorrelationProcessor(index+1)
        proc.analysis_type = AnalysisType.ROI_FOM
        proc._resolution = resolution
        if resolution > 0:
            proc._corr = OneWayAccuPairSequence(resolution)
            proc._corr_slave = OneWayAccuPairSequence(resolution)

        # first data
        processed.roi.fom = 10
        proc._source = slow_src
        proc.process(data)
        if resolution == 0:
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
            np.testing.assert_array_equal(np.array([10], dtype=np.float64), corr[index].y)
            assert len(corr[index].x_slave) == 0
            assert len(corr[index].y_slave) == 0
        else:
            assert len(corr[index].x) == 0
            assert len(corr[index].y[0]) == 0
            assert len(corr[index].x_slave) == 0
            assert len(corr[index].y_slave[0]) == 0

        # second data
        processed.roi.fom = 20
        processed.roi.fom_slave = 1
        proc._source = slow_src
        proc.process(data)
        if resolution == 0:
            np.testing.assert_array_equal(np.array([1, 1], dtype=np.float64), corr[index].x)
            np.testing.assert_array_equal(np.array([10, 20], dtype=np.float64), corr[index].y)
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x_slave)
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].y_slave)
        else:
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
            assert len(corr[index].y[0]) == 1
            assert len(corr[index].x_slave) == 0
            assert len(corr[index].y_slave[0]) == 0

        # third data
        processed.roi.fom = 30
        processed.roi.fom_slave = 2
        proc._source = slow_src
        proc.process(data)
        if resolution == 0:
            np.testing.assert_array_equal(np.array([1, 1, 1], dtype=np.float64), corr[index].x)
            np.testing.assert_array_equal(np.array([10, 20, 30], dtype=np.float64), corr[index].y)
            np.testing.assert_array_equal(np.array([1, 1], dtype=np.float64), corr[index].x_slave)
            np.testing.assert_array_equal(np.array([1, 2], dtype=np.float64), corr[index].y_slave)
        else:
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
            assert len(corr[index].y[0]) == 1
            np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x_slave)
            assert len(corr[index].y_slave[0]) == 1

        # test reset
        proc._prepare_reset_ma = True
        proc._reset = True
        with mock.patch.object(proc._corr_pp, "reset") as patched_reset:
            proc.process(data)
            if resolution == 0:
                np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x)
                np.testing.assert_array_equal(np.array([30], dtype=np.float64), corr[index].y)
                np.testing.assert_array_equal(np.array([1], dtype=np.float64), corr[index].x_slave)
                np.testing.assert_array_equal(np.array([2], dtype=np.float64), corr[index].y_slave)
            else:
                assert len(corr[index].x) == 0
                assert len(corr[index].y[0]) == 0
                assert len(corr[index].x_slave) == 0
                assert len(corr[index].y_slave[0]) == 0

            # correlation_pp has another reset flag
            patched_reset.assert_not_called()

            assert not proc._prepare_reset_ma

    @mock.patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("resolution", [0, 2])
    def testAutoResetMovingAverage(self, error, resolution):
        data, processed = self.simple_data(1001, (2, 2))
        analysis_type = AnalysisType.PUMP_PROBE
        slow_src = f'A0 ppt'

        proc = CorrelationProcessor(1)
        proc.analysis_type = analysis_type
        proc._source = slow_src
        proc._resolution = resolution
        proc._auto_reset_ma = True
        if resolution > 0:
            proc._corr = OneWayAccuPairSequence(resolution)
            proc._corr_slave = OneWayAccuPairSequence(resolution)

        # 1st
        data['raw'] = {slow_src: 1}
        self._set_fom(processed, analysis_type, 10)
        proc.process(data)
        x, _ = proc._corr.data()
        if resolution == 0:
            assert not data['reset_ma']
        else:
            np.testing.assert_array_equal(np.array([], dtype=np.float64), x)
            assert data['reset_ma']

        # 2nd
        data['raw'] = {slow_src: 2}
        self._set_fom(processed, analysis_type, 20)
        proc.process(data)
        x, _ = proc._corr.data()
        assert not data['reset_ma']
        if resolution != 0:
            # the first (previous) data is dropped
            np.testing.assert_array_equal(np.array([], dtype=np.float64), x)

        # 3rd
        data['raw'] = {slow_src: 5}
        self._set_fom(processed, analysis_type, 50)
        proc.process(data)
        x, _ = proc._corr.data()
        if resolution == 0:
            assert not data['reset_ma']
        else:
            np.testing.assert_array_equal(np.array([], dtype=np.float64), x)
            assert data['reset_ma']

        # 4th, 5th
        data['raw'] = {slow_src: 6}
        self._set_fom(processed, analysis_type, 60)
        proc.process(data)
        data['raw'] = {slow_src: 7}
        self._set_fom(processed, analysis_type, 70)
        proc.process(data)
        x, _ = proc._corr.data()
        assert not data['reset_ma']
        if resolution != 0:
            np.testing.assert_array_equal(np.array([6.5], dtype=np.float64), x)

        # 6th
        data['raw'] = {slow_src: 8}
        self._set_fom(processed, analysis_type, 80)
        proc.process(data)
        x, _ = proc._corr.data()
        assert not data['reset_ma']
        if resolution != 0:
            np.testing.assert_array_equal(np.array([7], dtype=np.float64), x)

        # 7th
        data['raw'] = {slow_src: 1}
        self._set_fom(processed, analysis_type, 10)
        proc.process(data)
        if resolution == 0:
            assert not data['reset_ma']
        else:
            np.testing.assert_array_equal(np.array([7], dtype=np.float64), x)
            assert data['reset_ma']
