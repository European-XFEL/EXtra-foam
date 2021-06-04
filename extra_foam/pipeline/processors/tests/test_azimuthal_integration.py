from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.processors import (
    AzimuthalIntegProcessorTrain, AzimuthalIntegProcessorPulse
)
from extra_foam.algorithms import mask_image_data
from extra_foam.config import AnalysisType, list_azimuthal_integ_methods


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

    @pytest.mark.parametrize("method", list_azimuthal_integ_methods("LPD"))
    def testAzimuthalIntegration(self, method):
        proc = self._proc
        proc._integ_method = method

        shape = (4, 128, 64)
        image_mask = np.zeros(shape[-2:], dtype=bool)
        image_mask[:, ::2] = True
        data, processed = self.data_with_assembled(1001, shape,
                                                   image_mask=image_mask,
                                                   threshold_mask=(0, 0.5))
        with patch.object(proc._meta, 'has_analysis',
                          side_effect=lambda x: True if x == AnalysisType.AZIMUTHAL_INTEG else False):
            proc.process(data)

            ai = processed.ai
            assert len(ai.x) == proc._integ_points
            assert all([not np.isnan(v) for v in ai.y])
            assert len(ai.y) == proc._integ_points
            assert all([not np.isnan(v) for v in ai.y])
            assert ai.fom is not None and ai.fom != 0
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

    def testAzimuthalIntegrationPp(self):
        proc = self._proc

        shape = (4, 128, 64)
        image_mask = np.zeros(shape[-2:], dtype=bool)
        image_mask[::2, ::2] = True
        threshold_mask = (0, 0.5)
        data, processed = self.data_with_assembled(1001, shape,
                                                   image_mask=image_mask,
                                                   threshold_mask=threshold_mask)

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
        pp.analysis_type = AnalysisType.AZIMUTHAL_INTEG
        pp.image_on = image_on
        pp.image_off = image_off
        pp.on.mask = mask_on
        pp.off.mask = mask_off

        with patch.object(proc._meta, 'has_analysis',
                          side_effect=lambda x: True if x == AnalysisType.AZIMUTHAL_INTEG else False):
            proc.process(data)

            assert len(pp.y_on) == proc._integ_points
            assert all([not np.isnan(v) for v in pp.y_on])
            assert len(pp.y_off) == proc._integ_points
            assert all([not np.isnan(v) for v in pp.y_off])
            assert len(pp.x) == proc._integ_points
            assert len(pp.y) == proc._integ_points
            assert pp.fom is not None and pp.fom != 0
