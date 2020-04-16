import unittest
from unittest.mock import patch
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

from extra_foam.logger import logger_suite as logger
from extra_foam.gui import mkQApp
from extra_foam.special_suite.multicam_view_proc import MultiCamViewProcessor
from extra_foam.special_suite.multicam_view_w import (
    MultiCamViewWindow, CameraView
)
from extra_foam.pipeline.tests import _RawDataMixin

app = mkQApp()

logger.setLevel('CRITICAL')


class TestMultiCamView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._win = MultiCamViewWindow('SCS')

    @classmethod
    def tearDownClass(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    def testWindow(self):
        win = self._win

        self.assertEqual(4, len(win._plot_widgets_st))
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        self.assertEqual(4, counter[CameraView])

        win.updateWidgetsST()

    def testCtrl(self):

        win = self._win
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values

        # test set new values
        widgets = ctrl_widget.output_channels
        for i, widget in enumerate(widgets):
            widget.clear()
            QTest.keyClicks(widget, f"new/output/channel{i}")
            QTest.keyPress(widget, Qt.Key_Enter)
            self.assertEqual(f"new/output/channel{i}", proc._output_channels[i])

        widgets = ctrl_widget.properties
        for i, widget in enumerate(widgets):
            widget.clear()
            QTest.keyClicks(widget, f"new/property{i}")
            QTest.keyPress(widget, Qt.Key_Enter)
            self.assertEqual(f"new/property{i}", proc._properties[i])


class TestMultiCamViewProcessor(_RawDataMixin):
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
        for i, gt in enumerate(proc._output_channels):
            assert gt == processed["channels"][i]

        assert processed["images"][0] is None
        assert processed["images"][1] is None
        np.testing.assert_array_equal(np.ones((3, 3)), processed["images"][2])
        np.testing.assert_array_equal(np.ones((4, 4)), processed["images"][3])
        assert np.float32 == processed["images"][3].dtype
