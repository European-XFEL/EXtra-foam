"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from unittest.mock import MagicMock, patch
import pytest

from extra_foam.pipeline.processors import FomPulseFilter, FomTrainFilter
from extra_foam.pipeline.exceptions import ProcessingError
from extra_foam.config import AnalysisType
from extra_foam.pipeline.tests import _TestDataMixin


_analysis_types_pulse = [AnalysisType.ROI_FOM_PULSE]
_analysis_types_train = [AnalysisType.ROI_FOM]


class TestFomPulseFilter(_TestDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = FomPulseFilter()
        self._proc.filter_pulse_by_vrange = MagicMock()

    def _set_fom(self, processed, analysis_type, fom):
        if analysis_type == AnalysisType.ROI_FOM_PULSE:
            processed.pulse.roi.fom = fom
        else:
            raise NotImplementedError

    def testUndefined(self):
        proc = self._proc

        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.UNDEFINED
        proc.process(data)
        proc.filter_pulse_by_vrange.assert_not_called()

    @pytest.mark.parametrize("analysis_type", _analysis_types_pulse)
    def testFomPulseFilter(self, analysis_type):
        proc = self._proc
        proc._fom_range = (-5, 5)

        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = analysis_type
        with pytest.raises(ProcessingError):
            proc.process(data)  # FOM is not available

        self._set_fom(processed, analysis_type, [1, 2, 3])
        proc.process(data)
        proc.filter_pulse_by_vrange.assert_called_with(
            [1, 2, 3], (-5, 5), processed.pidx)


class TestFomTrainFilter(_TestDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = FomTrainFilter()
        self._proc.filter_train_by_vrange = MagicMock()

    def _set_fom(self, processed, analysis_type, fom):
        if analysis_type == AnalysisType.ROI_FOM:
            processed.roi.fom = fom
        else:
            raise NotImplementedError

    def testUndefined(self):
        proc = self._proc

        data, processed = self.simple_data(1001, (4, 2, 2))
        proc.analysis_type = AnalysisType.UNDEFINED
        proc.process(data)
        proc.filter_train_by_vrange.assert_not_called()

    @pytest.mark.parametrize("analysis_type", _analysis_types_train)
    def testFomTrainFilter(self, analysis_type):
        proc = self._proc
        proc._fom_range = (-10, 10)

        data, processed = self.simple_data(1001, (2, 2))
        proc.analysis_type = analysis_type
        with pytest.raises(ProcessingError):
            proc.process(data)  # FOM is not available

        self._set_fom(processed, analysis_type, 2)
        proc.process(data)
        proc.filter_train_by_vrange.assert_called_with(
            2, (-10, 10), 'FOM train filter')
