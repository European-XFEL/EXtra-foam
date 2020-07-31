from collections import Counter

import pytest
import numpy as np

from extra_foam.pipeline.tests import _TestDataMixin

from extra_foam.special_suite import logger, mkQApp
from extra_foam.pipeline.exceptions import ProcessingError
from extra_foam.special_suite.vector_view import (
    VectorViewProcessor, VectorViewWindow, VectorPlot, VectorCorrelationPlot,
    InTrainVectorCorrelationPlot
)

from . import _SpecialSuiteWindowTestBase, _SpecialSuiteProcessorTestBase


app = mkQApp()

logger.setLevel('INFO')


class TestCamViewWindow(_SpecialSuiteWindowTestBase):
    @classmethod
    def setUpClass(cls):
        cls._win = VectorViewWindow('SCS')

    @classmethod
    def tearDownClass(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    @staticmethod
    def data4visualization():
        """Override."""
        return {
            "vector1": np.arange(10),
            "vector2": np.arange(10) + 5,
            "vector1_full": np.arange(100),
            "vector2_full": np.arange(100) + 5,
        }

    def testWindow(self):
        win = self._win

        self.assertEqual(3, len(win._plot_widgets_st))
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[VectorPlot])
        self.assertEqual(1, counter[InTrainVectorCorrelationPlot])
        self.assertEqual(1, counter[VectorCorrelationPlot])

        self._check_update_plots()

    def testCtrl(self):
        win = self._win
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values
        self.assertEqual('XGM intensity', proc._vector1)
        self.assertEqual('', proc._vector2)

        # test set new values
        widget = ctrl_widget.vector1_cb
        widget.setCurrentText("ROI FOM")
        self.assertEqual("ROI FOM", proc._vector1)

        widget = ctrl_widget.vector2_cb
        widget.setCurrentText("Digitizer pulse integral")
        self.assertEqual("Digitizer pulse integral", proc._vector2)


class TestVectorViewProcessor(_TestDataMixin, _SpecialSuiteProcessorTestBase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = VectorViewProcessor(object(), object())
        self._proc._vector1 = "XGM intensity"

    def testProcessing(self):
        data, processed = self.simple_data(1001, (4, 2, 2))
        proc = self._proc

        with pytest.raises(ProcessingError, match="XGM intensity is not available"):
            proc.process(data)

        processed.pulse.xgm.intensity = np.random.rand(4)
        ret = proc.process(data)
        self._check_processed_data_structure(ret)

        self._proc._vector2 = "ROI FOM"
        processed.pulse.roi.fom = np.random.rand(5)
        with pytest.raises(ProcessingError, match="Vectors have different lengths"):
            proc.process(data)
        processed.pulse.roi.fom = np.random.rand(4)
        proc.process(data)

        self._proc._vector2 = "Digitizer pulse integral"
        processed.pulse.digitizer.ch_normalizer = 'B'
        processed.pulse.digitizer['B'].pulse_integral = np.random.rand(4)
        proc.process(data)

    def _check_processed_data_structure(self, ret):
        """Override."""
        data_gt = TestCamViewWindow.data4visualization().keys()
        assert set(ret.keys()) == set(data_gt)
