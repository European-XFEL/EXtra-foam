from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

from extra_foam.pipeline.tests import _RawDataMixin

from extra_foam.special_suite import logger, mkQApp
from extra_foam.special_suite.multicam_view_proc import MultiCamViewProcessor
from extra_foam.special_suite.multicam_view_w import (
    MultiCamViewWindow, CameraView
)

from . import _SpecialSuiteWindowTestBase, _SpecialSuiteProcessorTestBase

app = mkQApp()
window_type = MultiCamViewWindow

logger.setLevel('INFO')


class TestMultiCamViewWindow(_SpecialSuiteWindowTestBase):
    @staticmethod
    def data4visualization():
        """Override."""
        return {
            "channels": {0: "camera1", 1: None, 2: None, 3: "camera2"},
            "images": {0: np.ones((4, 5)), 1: None, 2: None, 3: np.ones((5, 6))}
        }

    def testWindow(self, win, check_update_plots):
        assert 4 == len(win._plot_widgets_st)
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        assert 4 == counter[CameraView]

        check_update_plots()

    def testCtrl(self, win):
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values

        # test set new values
        widgets = ctrl_widget.output_channels
        for i, widget in enumerate(widgets):
            widget.clear()
            QTest.keyClicks(widget, f"new/output/channel{i}")
            QTest.keyPress(widget, Qt.Key_Enter)
            assert f"new/output/channel{i}" == proc._output_channels[i]

        widgets = ctrl_widget.properties
        for i, widget in enumerate(widgets):
            widget.clear()
            QTest.keyClicks(widget, f"new/property{i}")
            QTest.keyPress(widget, Qt.Key_Enter)
            assert f"new/property{i}" == proc._properties[i]


class TestMultiCamViewProcessor(_RawDataMixin, _SpecialSuiteProcessorTestBase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = MultiCamViewProcessor(object(), object())

    def testProcessing(self):
        proc = self._proc

        proc._output_channels = [
            "", "camera1:output", "camera2:output", "camera3:output"
        ]
        proc._properties = [
            "data.pixel", "", "data.pixel", "data.adc"
        ]

        data = self._gen_data(1234, {
            "camera1:output": [("data.pixel", np.ones((2, 2)))],
            "camera2:output": [("data.pixel", np.ones((3, 3)))],
            "camera3:output": [("data.adc", np.ones((4, 4, 1)))]
        })

        processed = proc.process(data)
        self._check_processed_data_structure(processed)

        for i, gt in enumerate(proc._output_channels):
            assert gt == processed["channels"][i]

        assert processed["images"][0] is None
        assert processed["images"][1] is None
        np.testing.assert_array_equal(np.ones((3, 3)), processed["images"][2])
        np.testing.assert_array_equal(np.ones((4, 4)), processed["images"][3])
        assert np.float32 == processed["images"][3].dtype

    def _check_processed_data_structure(self, ret):
        """Override."""
        data_gt = TestMultiCamViewWindow.data4visualization().keys()
        assert set(ret.keys()) == set(data_gt)
