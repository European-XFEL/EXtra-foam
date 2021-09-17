from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.data_model import AzimuthalIntegrationData, PumpProbeData
from extra_foam.pipeline.processors import (
    AzimuthalIntegProcessorTrain, AzimuthalIntegProcessorPulse
)
from extra_foam.pipeline.exceptions import ProcessingError
from extra_foam.algorithms import mask_image_data
from extra_foam.config import AnalysisType, list_azimuthal_integ_methods


_analysis_types = [AnalysisType.AZIMUTHAL_INTEG,
                   AnalysisType.AZIMUTHAL_INTEG_PEAK,
                   AnalysisType.AZIMUTHAL_INTEG_PEAK_Q,
                   AnalysisType.AZIMUTHAL_INTEG_COM]


@contextmanager
def patch_proc(proc, analysis_type):
    with patch.object(proc._meta, 'has_analysis', side_effect=lambda x: analysis_type == x) as one, \
         patch.object(proc._meta, 'has_any_analysis', side_effect=lambda x: analysis_type in x) as two:
        yield (one, two)


class TestAzimuthalIntegProcessorTrain(_TestDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        proc = AzimuthalIntegProcessorTrain()

        proc._sample_dist = 0.2
        proc._pixel1 = 2e-4
        proc._pixel2 = 2e-4
        proc._poni1 = 0
        proc._poni2 = 0
        proc._wavelength = 5e-10

        proc._integ_method = 'BBox'
        proc._integ_range = (0, 0.2)
        proc._integ_points = 64

        proc._fom_integ_range = (-np.inf, np.inf)

        self._proc = proc

    def create_test_data(self, row_mask=np.s_[:]):
        shape = (4, 128, 64)
        image_mask = np.zeros(shape[-2:], dtype=bool)
        image_mask[row_mask, ::2] = True
        threshold_mask = (0, 0.5)
        data, processed = self.data_with_assembled(1001, shape,
                                                   image_mask=image_mask,
                                                   threshold_mask=threshold_mask)

        return data, processed, image_mask, threshold_mask

    def extract_fom(self, ai, analysis_type):
        if analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            return ai.fom
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG_PEAK:
            return ai.max_peak
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG_PEAK_Q:
            return ai.max_peak_q
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG_COM:
            return ai.center_of_mass
        else:
            raise NotImplementedError

    @pytest.mark.parametrize("analysis_type", _analysis_types)
    @pytest.mark.parametrize("method", list_azimuthal_integ_methods("LPD"))
    def testAzimuthalIntegration(self, method, analysis_type):
        proc = self._proc
        proc._integ_method = method

        data, processed, _, _ = self.create_test_data()

        with patch_proc(proc, analysis_type):
            proc.process(data)

            ai = processed.ai
            fom = self.extract_fom(ai, analysis_type)

            assert len(ai.x) == proc._integ_points
            assert all([not np.isnan(v) for v in ai.y])
            assert len(ai.y) == proc._integ_points
            assert all([not np.isnan(v) for v in ai.y])
            assert fom is not None and fom != 0
            # assert shape[-2:] == ai.q_map.shape
            assert ai.peaks is None

            # test peak finding
            with patch("extra_foam.pipeline.processors.azimuthal_integration.find_peaks_1d") as mocked_find_peaks:
                proc._find_peaks = True
                proc._peak_prominence = 10
                proc._peak_slicer = slice(1, -1)
                mocked_find_peaks.return_value = np.arange(10), object()
                proc.process(data)
                np.testing.assert_array_equal(np.arange(1, 9), ai.peaks)

                # test too many peaks found
                mocked_find_peaks.return_value = np.arange(proc._MAX_N_PEAKS + 3), object()
                proc.process(data)
                assert ai.peaks is None

    @pytest.mark.parametrize("analysis_type", _analysis_types)
    def testAzimuthalIntegrationPp(self, analysis_type):
        proc = self._proc

        data, processed, image_mask, threshold_mask = self.create_test_data(row_mask=np.s_[::2])

        image_on = np.nanmean(data['assembled']['sliced'][::2, ...], axis=0)
        mask_on = np.zeros_like(image_mask)
        mask_image_data(image_on,
                        image_mask=image_mask,
                        threshold_mask=threshold_mask,
                        out=mask_on)

        image_off = np.nanmean(data['assembled']['sliced'][1::2, ...], axis=0)
        mask_off = np.zeros_like(image_mask)
        mask_image_data(image_off,
                        image_mask=image_mask,
                        threshold_mask=threshold_mask,
                        out=mask_off)

        pp = processed.pp
        pp.analysis_type = analysis_type
        pp.image_on = image_on
        pp.image_off = image_off
        pp.on.mask = mask_on
        pp.off.mask = mask_off

        with patch_proc(proc, analysis_type):
            proc.process(data)

            fom = self.extract_fom(pp, analysis_type)

            assert len(pp.y_on) == proc._integ_points
            assert all([not np.isnan(v) for v in pp.y_on])
            assert len(pp.y_off) == proc._integ_points
            assert all([not np.isnan(v) for v in pp.y_off])
            assert len(pp.x) == proc._integ_points
            assert len(pp.y) == proc._integ_points
            assert fom is not None and fom != 0

    def testComputeFom(self):
        proc = self._proc

        ai = AzimuthalIntegrationData()
        ai.x = np.arange(0, 10)
        ai.y = np.arange(-5, 5)

        with patch_proc(proc, AnalysisType.AZIMUTHAL_INTEG) as (has_analysis, _):
            proc._process_fom(ai)

            # Sanity test
            has_analysis.assert_called()
            assert np.sum(np.abs(ai.y)) == ai.fom

        with patch_proc(proc, AnalysisType.AZIMUTHAL_INTEG_COM):
            proc._process_fom(ai)

            # Negative CoMs should be discarded
            assert ai.center_of_mass == (np.nan, np.nan)

        # Attempting to find peak FoMs should fail if peak finding is disabled
        proc._find_peaks = False
        with patch_proc(proc, AnalysisType.AZIMUTHAL_INTEG_PEAK), pytest.raises(ProcessingError):
            proc._process_fom(ai)
        with patch_proc(proc, AnalysisType.AZIMUTHAL_INTEG_PEAK_Q), pytest.raises(ProcessingError):
            proc._process_fom(ai)

        proc._find_peaks = True

        # If there are no peaks, we should get NaNs
        with patch_proc(proc, AnalysisType.AZIMUTHAL_INTEG_PEAK):
            proc._process_fom(ai)
            assert np.isnan(ai.max_peak)
        with patch_proc(proc, AnalysisType.AZIMUTHAL_INTEG_PEAK_Q):
            proc._process_fom(ai)
            assert np.isnan(ai.max_peak_q)

        # Check behaviour when pump-probe analysis is requested
        ai = PumpProbeData()
        ai.x = np.arange(0, 10)
        ai.y = np.arange(-5, 5)

        # Sanity test
        ai.analysis_type = AnalysisType.AZIMUTHAL_INTEG
        proc._process_fom(ai)
        assert np.sum(np.abs(ai.y)) == ai.fom

        # abs_difference should be respected
        ai.abs_difference = False
        proc._process_fom(ai)
        assert np.sum(ai.y) == ai.fom
