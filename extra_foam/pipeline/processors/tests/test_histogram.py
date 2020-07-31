"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Ebad Kamil <ebad.kamil@xfel.eu>, Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from unittest.mock import patch
import pytest

import numpy as np

from extra_foam.pipeline.processors.histogram import HistogramProcessor, ProcessingError
from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.config import AnalysisType

np.warnings.filterwarnings("ignore", category=RuntimeWarning)

_analysis_types = [
    AnalysisType.PUMP_PROBE,
    AnalysisType.ROI_FOM
]


class TestHistogramProcessor(_TestDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = HistogramProcessor()
        self._proc._n_bins = 5

    def _set_fom_pulse(self, processed, analysis_type, fom):
        if analysis_type == AnalysisType.ROI_FOM:
            processed.pulse.roi.fom = fom
        else:
            raise NotImplementedError

    def _set_fom_train(self, processed, analysis_type, fom):
        if analysis_type == AnalysisType.PUMP_PROBE:
            processed.pp.fom = fom
        elif analysis_type == AnalysisType.ROI_FOM:
            processed.roi.fom = fom
        else:
            raise NotImplementedError

    def testUndefined(self):
        # Test Undefined analysis type
        self._proc.analysis_type = AnalysisType.UNDEFINED

        data, processed = self.simple_data(1001, (5, 2, 2))
        self._proc.process(data)

        assert processed.hist.hist is None
        assert processed.hist.bin_centers is None
        assert processed.pulse.hist.pulse_foms is None
        with pytest.raises(KeyError):
            processed.pulse.hist[0]

    @patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("analysis_type", _analysis_types)
    def testFomHistogramPr(self, error, analysis_type):
        proc = self._proc
        proc._pulse_resolved = True

        # -----------
        # FOM is None
        # -----------

        proc.analysis_type = analysis_type + AnalysisType.PULSE
        data, processed = self.simple_data(1001, (5, 2, 2))

        if analysis_type != AnalysisType.PUMP_PROBE:
            proc.process(data)
            error.assert_called_once()
            assert "is not available" in error.call_args[0][0]
            # It also test that the code can deal with empty FOM array
            error.reset_mock()
        else:
            with pytest.raises(ProcessingError, match="not supported"):
                proc.process(data)

        if analysis_type == AnalysisType.PUMP_PROBE:
            return

        fom_gt1 = [10, 20, 30, 40, 20]
        fom_gt2 = [30, 20, 10, 40, 10]
        # test histogram
        self._set_fom_pulse(processed, analysis_type, fom_gt1.copy())
        proc.process(data)
        self._set_fom_pulse(processed, analysis_type, fom_gt2.copy())
        proc.process(data)

        np.testing.assert_array_equal(fom_gt1 + fom_gt2, proc._fom)
        np.testing.assert_array_almost_equal([3, 3, 0, 2, 2], processed.hist.hist)
        np.testing.assert_array_almost_equal([13.,  19.,  25.,  31.,  37.], processed.hist.bin_centers)

        # test POI histogram
        assert 5 == processed.n_pulses
        assert [0, 0] == processed.image.poi_indices
        np.testing.assert_array_equal([1, 0, 0, 0, 1],
                                      processed.pulse.hist[0].hist)  # histogram of [10, 30]
        np.testing.assert_array_equal([12., 16., 20., 24., 28.],
                                      processed.pulse.hist[0].bin_centers)
        with pytest.raises(KeyError):
            processed.pulse.hist[1]

        # change POI indices
        fom_gt3 = [25] * 5
        processed.pulse.roi.fom = fom_gt3.copy()
        processed.image.poi_indices = [0, 3]
        proc.process(data) # assume that the same data comes again
        np.testing.assert_array_equal(fom_gt1 + fom_gt2 + fom_gt3, proc._fom)
        np.testing.assert_array_almost_equal([3, 3, 5, 2, 2], processed.hist.hist)
        np.testing.assert_array_almost_equal([13.,  19.,  25.,  31.,  37.],
                                             processed.hist.bin_centers)

        # test POI histogram
        np.testing.assert_array_equal([1, 0, 0, 1, 1],
                                      processed.pulse.hist[0].hist)  # histogram of [10, 25, 30]
        np.testing.assert_array_equal([12, 16, 20, 24, 28],
                                      processed.pulse.hist[0].bin_centers)
        np.testing.assert_array_equal([1, 0, 0, 0, 2],
                                      processed.pulse.hist[3].hist)  # histogram of [40, 40, 25]
        np.testing.assert_array_equal([26.5, 29.5, 32.5, 35.5, 38.5],
                                      processed.pulse.hist[3].bin_centers)

    @patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("analysis_type", _analysis_types)
    def testFomHistogramTr(self, error, analysis_type):
        proc = self._proc
        proc._pulse_resolved = False

        proc.analysis_type = analysis_type
        data, processed = self.simple_data(1001, (2, 2))

        # -----------
        # FOM is None
        # -----------

        proc.process(data)
        if analysis_type != AnalysisType.PUMP_PROBE:
            error.assert_called_once()
            assert "is not available" in error.call_args[0][0]
            error.reset_mock()
        else:
            assert 1 == proc._pp_fail_flag

        # again
        if analysis_type == AnalysisType.PUMP_PROBE:
            proc.process(data)
            error.assert_called_once()
            error.reset_mock()
            assert 0 == proc._pp_fail_flag

        # ---------------
        # New data arrive
        # ---------------

        fom_gt = [10, 20, 30, 40, 20, 30, 20, 10, 40, 10]
        for _fom in fom_gt:
            self._set_fom_train(processed, analysis_type, _fom)
            proc.process(data)

        assert processed.pulse.hist.pulse_foms is None
        np.testing.assert_array_equal(fom_gt, proc._fom)
        np.testing.assert_array_almost_equal([13.,  19.,  25.,  31.,  37.], processed.hist.bin_centers)
        np.testing.assert_array_almost_equal([3, 3, 0, 2, 2], processed.hist.hist)
        with pytest.raises(KeyError):
            processed.pulse.hist[0]

        # the same data comes again
        for _fom in fom_gt:
            self._set_fom_train(processed, analysis_type, _fom)
            proc.process(data)

        np.testing.assert_array_equal(fom_gt * 2, proc._fom)
        np.testing.assert_array_almost_equal([13.,  19.,  25.,  31.,  37.], processed.hist.bin_centers)
        np.testing.assert_array_almost_equal([6, 6, 0, 4, 4], processed.hist.hist)
