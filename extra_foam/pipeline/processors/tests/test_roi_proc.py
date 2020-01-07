"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import random
from unittest.mock import MagicMock, patch

import pytest

import numpy as np

from extra_foam.pipeline.processors import RoiProcessorTrain, RoiProcessorPulse
from extra_foam.config import AnalysisType, Normalizer, RoiCombo, RoiFom
from extra_foam.pipeline.processors.tests import _BaseProcessorTest


_handlers = {
    RoiFom.SUM: np.sum,
    RoiFom.MEAN: np.mean,
    RoiFom.MEDIAN: np.median,
    RoiFom.MAX: np.max,
    RoiFom.MIN: np.min,
}


class TestRoiProcessorPulse(_BaseProcessorTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        proc = RoiProcessorPulse()
        proc._geom1 = [0, 1, 2, 3]
        proc._geom2 = [1, 0, 1, 3]
        proc._geom3 = [1, 2, 2, 3]
        proc._geom4 = [3, 2, 3, 4]
        self._proc = proc

    def _get_data(self):
        return self.data_with_assembled(1001, (4, 20, 20))

    def _get_roi_slice(self, geom):
        return slice(geom[1], geom[1] + geom[3]), slice(geom[0], geom[0] + geom[2])

    def testRoiGeom(self):
        proc = self._proc

        data, processed = self._get_data()
        proc.process(data)

        roi = processed.roi
        assert list(roi.geom1.geometry) == proc._geom1
        assert list(roi.geom2.geometry) == proc._geom2
        assert list(roi.geom3.geometry) == proc._geom3
        assert list(roi.geom4.geometry) == proc._geom4

    @pytest.mark.parametrize("norm_type, fom_handler", [(k, v) for k, v in _handlers.items()])
    def testRoiNorm(self, norm_type, fom_handler):
        proc = self._proc

        for combo, geom in zip([RoiCombo.ROI3, RoiCombo.ROI4], ['_geom3', '_geom4']):
            data, processed = self._get_data()
            proc._norm_combo = combo
            proc._norm_type = norm_type
            proc.process(data)
            s = self._get_roi_slice(getattr(proc, geom))
            fom_gt = fom_handler(data['detector']['assembled'][:, s[0], s[1]], axis=(-1, -2))
            np.testing.assert_array_equal(fom_gt, processed.pulse.roi.norm)

        for norm_combo in [RoiCombo.ROI3_SUB_ROI4, RoiCombo.ROI3_ADD_ROI4]:
            data, processed = self._get_data()
            proc._norm_combo = norm_combo
            proc._norm_type = norm_type
            proc.process(data)
            s3 = self._get_roi_slice(proc._geom3)
            fom3_gt = fom_handler(data['detector']['assembled'][:, s3[0], s3[1]], axis=(-1, -2))
            s4 = self._get_roi_slice(proc._geom4)
            fom4_gt = fom_handler(data['detector']['assembled'][:, s4[0], s4[1]], axis=(-1, -2))
            if norm_combo == RoiCombo.ROI3_SUB_ROI4:
                np.testing.assert_array_equal(fom3_gt - fom4_gt, processed.pulse.roi.norm)
            else:
                np.testing.assert_array_equal(fom3_gt + fom4_gt, processed.pulse.roi.norm)

    @pytest.mark.parametrize("fom_type, fom_handler", [(k, v) for k, v in _handlers.items()])
    def testRoiFom(self, fom_type, fom_handler):
        proc = self._proc

        for combo, geom in zip([RoiCombo.ROI1, RoiCombo.ROI2], ['_geom1', '_geom2']):
            data, processed = self._get_data()
            proc._fom_combo = combo
            proc._fom_type = fom_type
            proc._fom_norm = Normalizer.UNDEFINED
            proc.process(data)
            s = self._get_roi_slice(getattr(proc, geom))
            fom_gt = fom_handler(data['detector']['assembled'][:, s[0], s[1]], axis=(-1, -2))
            np.testing.assert_array_equal(fom_gt, processed.pulse.roi.fom)

        for fom_combo in [RoiCombo.ROI1_SUB_ROI2, RoiCombo.ROI1_ADD_ROI2]:
            data, processed = self._get_data()
            proc._fom_combo = fom_combo
            proc._fom_type = fom_type
            proc._fom_norm = Normalizer.UNDEFINED
            proc.process(data)
            s1 = self._get_roi_slice(proc._geom1)
            fom1_gt = fom_handler(data['detector']['assembled'][:, s1[0], s1[1]], axis=(-1, -2))
            s2 = self._get_roi_slice(proc._geom2)
            fom2_gt = fom_handler(data['detector']['assembled'][:, s2[0], s2[1]], axis=(-1, -2))
            if fom_combo == RoiCombo.ROI1_SUB_ROI2:
                np.testing.assert_array_equal(fom1_gt - fom2_gt, processed.pulse.roi.fom)
            else:
                np.testing.assert_array_equal(fom1_gt + fom2_gt, processed.pulse.roi.fom)


class TestRoiProcessorTrain(_BaseProcessorTest):

    @pytest.fixture(autouse=True)
    def setUp(self):
        proc = RoiProcessorTrain()
        proc._reset_roi_moving_average()
        proc._set_roi_moving_average_window(1)

        proc._auc_range = (0, 1000)
        proc._fom_integ_range = (0, 1000)

        proc._meta.has_analysis = MagicMock(return_value=False)
        proc._meta.has_any_analysis = MagicMock(return_value=False)

        self._proc = proc

    def _get_data(self):
        shape = (20, 20)
        data, processed = self.data_with_assembled(1001, shape)
        proc = RoiProcessorPulse()
        proc._geom1 = [0, 1, 2, 3]
        proc._geom2 = [1, 0, 2, 3]
        proc._geom3 = [1, 2, 2, 3]
        proc._geom4 = [3, 2, 3, 4]
        # set processed.roi.geom{1, 2, 3, 4}
        proc.process(data)
        processed.pp.image_on = np.random.randn(*shape).astype(np.float32)
        processed.pp.image_off = np.random.randn(*shape).astype(np.float32)
        return data, processed

    def _get_roi_slice(self, geom):
        return slice(geom[1], geom[1] + geom[3]), slice(geom[0], geom[0] + geom[2])

    @pytest.mark.parametrize("norm_type, fom_handler", [(k, v) for k, v in _handlers.items()])
    def testRoiNorm(self, norm_type, fom_handler):
        proc = self._proc

        for combo, geom in zip([RoiCombo.ROI3, RoiCombo.ROI4], ['geom3', 'geom4']):
            data, processed = self._get_data()
            proc._norm_combo = combo
            proc._norm_type = norm_type
            proc.process(data)
            s = self._get_roi_slice(getattr(processed.roi, geom).geometry)
            assert fom_handler(processed.image.masked_mean[s[0], s[1]]) == processed.roi.norm

        for norm_combo in [RoiCombo.ROI3_SUB_ROI4, RoiCombo.ROI3_ADD_ROI4]:
            data, processed = self._get_data()
            proc._norm_combo = norm_combo
            proc._norm_type = norm_type
            proc.process(data)
            s3 = self._get_roi_slice(processed.roi.geom3.geometry)
            fom3_gt = fom_handler(processed.image.masked_mean[s3[0], s3[1]])
            s4 = self._get_roi_slice(processed.roi.geom4.geometry)
            fom4_gt = fom_handler(processed.image.masked_mean[s4[0], s4[1]])
            if norm_combo == RoiCombo.ROI3_SUB_ROI4:
                assert fom3_gt - fom4_gt == processed.roi.norm
            else:
                assert fom3_gt + fom4_gt == processed.roi.norm

    @pytest.mark.parametrize("fom_type, fom_handler", [(k, v) for k, v in _handlers.items()])
    def testRoiFom(self, fom_type, fom_handler):
        proc = self._proc

        # We do not test all the combinations of parameters.

        for combo, geom in zip([RoiCombo.ROI1, RoiCombo.ROI2], ['geom1', 'geom2']):
            data, processed = self._get_data()
            proc._fom_combo = combo
            proc._fom_type = fom_type
            proc._fom_norm = Normalizer.UNDEFINED
            proc.process(data)
            s = self._get_roi_slice(getattr(processed.roi, geom).geometry)
            assert fom_handler(processed.image.masked_mean[s[0], s[1]]) == processed.roi.fom

        for fom_combo in [RoiCombo.ROI1_SUB_ROI2, RoiCombo.ROI1_ADD_ROI2]:
            data, processed = self._get_data()
            proc._fom_combo = fom_combo
            proc._fom_type = fom_type
            proc._fom_norm = Normalizer.UNDEFINED
            proc.process(data)
            s1 = self._get_roi_slice(processed.roi.geom1.geometry)
            fom1_gt = fom_handler(processed.image.masked_mean[s1[0], s1[1]])
            s2 = self._get_roi_slice(processed.roi.geom2.geometry)
            fom2_gt = fom_handler(processed.image.masked_mean[s2[0], s2[1]])
            if fom_combo == RoiCombo.ROI1_SUB_ROI2:
                assert fom1_gt - fom2_gt == processed.roi.fom
            else:
                assert fom1_gt + fom2_gt == processed.roi.fom

    @patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("direct, axis", [('x', -2), ('y', -1)])
    def testProjFom(self, error, direct, axis):
        proc = self._proc

        for combo, geom in zip([RoiCombo.ROI1, RoiCombo.ROI2], ['geom1', 'geom2']):
            data, processed = self._get_data()
            proc._proj_combo = combo
            proc._proj_direct = direct
            proc._proj_norm = Normalizer.UNDEFINED
            proc.process(data)
            s = self._get_roi_slice(getattr(processed.roi, geom).geometry)
            y_gt = processed.image.masked_mean[s[0], s[1]].sum(axis=axis)
            np.testing.assert_array_equal(np.arange(len(y_gt)), processed.roi.proj.x)
            np.testing.assert_array_equal(y_gt, processed.roi.proj.y)
            assert np.sum(y_gt) == processed.roi.proj.fom

        for proj_combo in [RoiCombo.ROI1_SUB_ROI2, RoiCombo.ROI1_ADD_ROI2]:
            data, processed = self._get_data()
            proc._proj_combo = proj_combo
            proc._proj_direct = direct
            proc._proj_norm = Normalizer.UNDEFINED
            proc.process(data)
            s1 = self._get_roi_slice(processed.roi.geom1.geometry)
            y1_gt = processed.image.masked_mean[s1[0], s1[1]].sum(axis=axis)
            s2 = self._get_roi_slice(processed.roi.geom2.geometry)
            y2_gt = processed.image.masked_mean[s2[0], s2[1]].sum(axis=axis)
            if proj_combo == RoiCombo.ROI1_SUB_ROI2:
                y_gt = y1_gt - y2_gt
            else:
                y_gt = y1_gt + y2_gt
            np.testing.assert_array_equal(np.arange(len(y_gt)), processed.roi.proj.x)
            np.testing.assert_array_equal(y_gt, processed.roi.proj.y)
            np.testing.assert_array_equal(np.sum(y_gt), processed.roi.proj.fom)

            # test when ROI2 has different shape from ROI1
            processed.roi.geom2.geometry = [1, 0, 1, 3]
            proc.process(data)
            error.assert_called_once()
            error.reset_mock()

    def testGeneralPumpProbe(self):
        proc = self._proc
        proc._process_norm_pump_probe = MagicMock()
        proc._process_fom_pump_probe = MagicMock()
        proc._process_proj_pump_probe = MagicMock()

        data, processed = self._get_data()
        processed.pp.analysis_type = AnalysisType.UNDEFINED
        proc.process(data)
        proc._process_norm_pump_probe.assert_not_called()
        proc._process_fom_pump_probe.assert_not_called()
        proc._process_proj_pump_probe.assert_not_called()

        processed.pp.analysis_type = AnalysisType.ROI_PROJ
        proc.process(data)
        proc._process_norm_pump_probe.assert_called_once()
        proc._process_norm_pump_probe.reset_mock()
        proc._process_fom_pump_probe.assert_not_called()
        proc._process_proj_pump_probe.assert_called_once()
        proc._process_proj_pump_probe.reset_mock()

        processed.pp.analysis_type = AnalysisType.ROI_FOM
        proc.process(data)
        proc._process_norm_pump_probe.assert_called_once()
        proc._process_norm_pump_probe.reset_mock()
        proc._process_fom_pump_probe.assert_called_once()
        proc._process_fom_pump_probe.reset_mock()
        proc._process_proj_pump_probe.assert_not_called()

        processed.pp.image_on = None
        proc.process(data)
        proc._process_norm_pump_probe.assert_not_called()
        proc._process_fom_pump_probe.assert_not_called()
        proc._process_proj_pump_probe.assert_not_called()

    @pytest.mark.parametrize("norm_type, fom_handler", [(k, v) for k, v in _handlers.items()])
    def testRoiNormPumpProbe(self, norm_type, fom_handler):
        proc = self._proc

        for combo, geom in zip([RoiCombo.ROI3, RoiCombo.ROI4], ['geom3', 'geom4']):
            data, processed = self._get_data()
            processed.pp.analysis_type = random.choice([AnalysisType.ROI_FOM, AnalysisType.ROI_PROJ])
            proc._norm_combo = combo
            proc._norm_type = norm_type
            proc.process(data)
            s = self._get_roi_slice(getattr(processed.roi, geom).geometry)
            assert fom_handler(processed.pp.image_on[s[0], s[1]]) == processed.pp.roi_norm_on
            assert fom_handler(processed.pp.image_off[s[0], s[1]]) == processed.pp.roi_norm_off

        for norm_combo in [RoiCombo.ROI3_SUB_ROI4, RoiCombo.ROI3_ADD_ROI4]:
            data, processed = self._get_data()
            processed.pp.analysis_type = random.choice([AnalysisType.ROI_FOM, AnalysisType.ROI_PROJ])
            proc._norm_combo = norm_combo
            proc._norm_type = norm_type
            proc.process(data)
            s3 = self._get_roi_slice(processed.roi.geom3.geometry)
            fom3_on_gt = fom_handler(processed.pp.image_on[s3[0], s3[1]])
            fom3_off_gt = fom_handler(processed.pp.image_off[s3[0], s3[1]])
            s4 = self._get_roi_slice(processed.roi.geom4.geometry)
            fom4_on_gt = fom_handler(processed.pp.image_on[s4[0], s4[1]])
            fom4_off_gt = fom_handler(processed.pp.image_off[s4[0], s4[1]])
            if norm_combo == RoiCombo.ROI3_SUB_ROI4:
                fom_on_gt = fom3_on_gt - fom4_on_gt
                fom_off_gt = fom3_off_gt - fom4_off_gt
            else:
                fom_on_gt = fom3_on_gt + fom4_on_gt
                fom_off_gt = fom3_off_gt + fom4_off_gt
            assert fom_on_gt == processed.pp.roi_norm_on
            assert fom_off_gt == processed.pp.roi_norm_off

    @pytest.mark.parametrize("fom_type, fom_handler", [(k, v) for k, v in _handlers.items()])
    def testRoiFomPumpProbe(self, fom_type, fom_handler):
        proc = self._proc

        for combo, geom in zip([RoiCombo.ROI1, RoiCombo.ROI2], ['geom1', 'geom2']):
            data, processed = self._get_data()
            processed.pp.analysis_type = AnalysisType.ROI_FOM
            proc._fom_combo = combo
            proc._fom_type = fom_type
            proc.process(data)
            s = self._get_roi_slice(getattr(processed.roi, geom).geometry)
            fom_on_gt = fom_handler(processed.pp.image_on[s[0], s[1]])
            fom_off_gt = fom_handler(processed.pp.image_off[s[0], s[1]])
            assert fom_on_gt - fom_off_gt == processed.pp.fom

        for fom_combo in [RoiCombo.ROI1_SUB_ROI2, RoiCombo.ROI1_ADD_ROI2]:
            data, processed = self._get_data()
            processed.pp.analysis_type = AnalysisType.ROI_FOM
            proc._fom_combo = fom_combo
            proc._fom_type = fom_type
            proc.process(data)
            s1 = self._get_roi_slice(processed.roi.geom1.geometry)
            fom1_on_gt = fom_handler(processed.pp.image_on[s1[0], s1[1]])
            fom1_off_gt = fom_handler(processed.pp.image_off[s1[0], s1[1]])
            s2 = self._get_roi_slice(processed.roi.geom2.geometry)
            fom2_on_gt = fom_handler(processed.pp.image_on[s2[0], s2[1]])
            fom2_off_gt = fom_handler(processed.pp.image_off[s2[0], s2[1]])
            if fom_combo == RoiCombo.ROI1_SUB_ROI2:
                fom_on_gt = fom1_on_gt - fom2_on_gt
                fom_off_gt = fom1_off_gt - fom2_off_gt
            else:
                fom_on_gt = fom1_on_gt + fom2_on_gt
                fom_off_gt = fom1_off_gt + fom2_off_gt
            assert fom_on_gt - fom_off_gt == processed.pp.fom

    @patch('extra_foam.ipc.ProcessLogger.error')
    @pytest.mark.parametrize("direct, axis", [('x', -2), ('y', -1)])
    def testRoiProjPumpProbe(self, error, direct, axis):
        proc = self._proc

        for combo, geom in zip([RoiCombo.ROI1, RoiCombo.ROI2], ['geom1', 'geom2']):
            data, processed = self._get_data()
            processed.pp.analysis_type = AnalysisType.ROI_PROJ
            proc._proj_combo = combo
            proc._proj_direct = direct
            proc._proj_norm = Normalizer.UNDEFINED
            processed.pp.abs_difference = True
            proc.process(data)
            s = self._get_roi_slice(getattr(processed.roi, geom).geometry)
            y_on_gt = processed.pp.image_on[s[0], s[1]].sum(axis=axis)
            y_off_gt = processed.pp.image_off[s[0], s[1]].sum(axis=axis)
            np.testing.assert_array_equal(y_on_gt, processed.pp.y_on)
            np.testing.assert_array_equal(y_off_gt, processed.pp.y_off)
            np.testing.assert_array_equal(y_on_gt - y_off_gt, processed.pp.y)
            assert np.abs(y_on_gt - y_off_gt).sum() == processed.pp.fom
            # test abs_difference == False
            processed.pp.abs_difference = False
            proc.process(data)
            assert (y_on_gt - y_off_gt).sum() == processed.pp.fom

        for proj_combo in [RoiCombo.ROI1_SUB_ROI2, RoiCombo.ROI1_ADD_ROI2]:
            data, processed = self._get_data()
            processed.pp.analysis_type = AnalysisType.ROI_PROJ
            proc._proj_combo = proj_combo
            proc._proj_direct = direct
            proc._proj_norm = Normalizer.UNDEFINED
            processed.pp.abs_difference = True
            proc.process(data)
            s1 = self._get_roi_slice(processed.roi.geom1.geometry)
            y1_on_gt = processed.pp.image_on[s1[0], s1[1]].sum(axis=axis)
            y1_off_gt = processed.pp.image_off[s1[0], s1[1]].sum(axis=axis)
            s2 = self._get_roi_slice(processed.roi.geom2.geometry)
            y2_on_gt = processed.pp.image_on[s2[0], s2[1]].sum(axis=axis)
            y2_off_gt = processed.pp.image_off[s2[0], s2[1]].sum(axis=axis)
            if proj_combo == RoiCombo.ROI1_SUB_ROI2:
                y_on_gt = y1_on_gt - y2_on_gt
                y_off_gt = y1_off_gt - y2_off_gt
            else:
                y_on_gt = y1_on_gt + y2_on_gt
                y_off_gt = y1_off_gt + y2_off_gt
            np.testing.assert_array_equal(y_on_gt, processed.pp.y_on)
            np.testing.assert_array_equal(y_off_gt, processed.pp.y_off)
            np.testing.assert_array_equal(y_on_gt - y_off_gt, processed.pp.y)
            assert np.abs(y_on_gt - y_off_gt).sum() == processed.pp.fom
            # test abs_difference == False
            processed.pp.abs_difference = False
            proc.process(data)
            assert (y_on_gt - y_off_gt).sum() == processed.pp.fom
            # test when ROI2 has different shape from ROI1
            processed.roi.geom2.geometry = [1, 0, 1, 3]
            proc.process(data)
            error.assert_called_once()
            error.reset_mock()
