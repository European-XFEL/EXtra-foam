import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import time

import numpy as np

from PyQt5.QtTest import QTest, QSignalSpy
from PyQt5.QtCore import Qt, QPoint

from karaboFAI.algorithms import mask_image
from karaboFAI.config import config, _Config, ConfigWrapper
from karaboFAI.gui import mkQApp
from karaboFAI.gui.windows import ImageToolWindow
from karaboFAI.gui.windows.image_tool import _SimpleImageData
from karaboFAI.logger import logger
from karaboFAI.pipeline.data_model import ImageData, ProcessedData
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.processes import wait_until_redis_shutdown
from karaboFAI.services import FAI

app = mkQApp()

logger.setLevel('CRITICAL')


class TestSimpleImageData(unittest.TestCase):
    @patch.dict(config._data, {"PIXEL_SIZE": 1e-3})
    def testGeneral(self):
        with self.assertRaises(TypeError):
            _SimpleImageData([1, 2, 3])

        gt_data = np.arange(9).reshape(3, 3)
        img_data = _SimpleImageData.from_array(gt_data)

        img_data.background = 1
        np.testing.assert_array_equal(gt_data - 1, img_data.masked)
        img_data.background = 0
        np.testing.assert_array_equal(gt_data, img_data.masked)

        img_data.threshold_mask = (3, 6)
        np.testing.assert_array_equal(mask_image(gt_data, threshold_mask=(3, 6)),
                                      img_data.masked)
        img_data.background = 3
        np.testing.assert_array_equal(mask_image(gt_data-3, threshold_mask=(3, 6)),
                                      img_data.masked)

        self.assertEqual(1.0e-3, img_data.pixel_size)

    @patch.dict(config._data, {"PIXEL_SIZE": 1e-3})
    def testInstantiateFromArray(self):
        gt_data = np.ones((2, 2, 2))

        image_data = _SimpleImageData.from_array(gt_data)

        np.testing.assert_array_equal(np.ones((2, 2)), image_data.masked)
        self.assertEqual(1e-3, image_data.pixel_size)
        self.assertEqual(0, image_data.background)
        self.assertEqual(None, image_data.threshold_mask)


class TestImageTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()  # ensure file
        config.load('LPD')

        cls.fai = FAI().init()
        cls.gui = cls.fai._gui
        cls.scheduler = cls.fai.scheduler
        cls.image_worker = cls.fai.image_worker

        actions = cls.gui._tool_bar.actions()
        cls._action = actions[2]

        # close the ImageToolWindow opened together with the MainGUI
        window = list(cls.gui._windows.keys())[-1]
        assert isinstance(window, ImageToolWindow)
        window.close()

    @classmethod
    def tearDownClass(cls):
        cls.fai.terminate()

        wait_until_redis_shutdown()

    def setUp(self):
        ImageToolWindow.reset()
        self._action.trigger()
        self.image_tool = list(self.gui._windows.keys())[-1]

        self.view = self.image_tool._image_view
        self.view.setImageData(None)
        self.view._image = None

    def testDefaultValues(self):
        # This must be the first test method in order to check that the
        # default values are set correctly
        proc = self.scheduler._roi_proc
        widget = self.image_tool._roi_ctrl_widget

        proc.update()

        for i, ctrl in enumerate(widget._roi_ctrls, 1):
            roi_region = [int(ctrl._px_le.text()),
                          int(ctrl._py_le.text()),
                          int(ctrl._width_le.text()),
                          int(ctrl._height_le.text())]
            self.assertListEqual(roi_region, getattr(proc, f"_roi{i}").rect)

    def testRoiCtrlWidget(self):
        widget = self.image_tool._roi_ctrl_widget
        roi_ctrls = widget._roi_ctrls
        proc = self.scheduler._roi_proc
        self.assertEqual(4, len(roi_ctrls))

        for ctrl in roi_ctrls:
            self.assertFalse(ctrl.activate_cb.isChecked())
            self.assertFalse(ctrl.lock_cb.isChecked())
            self.assertFalse(ctrl._width_le.isEnabled())
            self.assertFalse(ctrl._height_le.isEnabled())
            self.assertFalse(ctrl._px_le.isEnabled())
            self.assertFalse(ctrl._py_le.isEnabled())

        roi1_ctrl = roi_ctrls[0]
        roi1 = self.view._rois[0]
        self.assertIs(roi1_ctrl._roi, roi1)

        # activate ROI1 ctrl
        QTest.mouseClick(roi1_ctrl.activate_cb, Qt.LeftButton,
                         pos=QPoint(2, roi1_ctrl.activate_cb.height()/2))
        self.assertTrue(roi1_ctrl.activate_cb.isChecked())
        proc.update()
        self.assertTrue(proc._roi1._activated)

        # test default values
        self.assertTupleEqual((float(roi1_ctrl._width_le.text()),
                               float(roi1_ctrl._height_le.text())),
                              tuple(roi1.size()))
        self.assertTupleEqual((float(roi1_ctrl._px_le.text()),
                               float(roi1_ctrl._py_le.text())),
                              tuple(roi1.pos()))

        # use keyClicks to test that the QLineEdit is enabled
        roi1_ctrl._width_le.clear()
        QTest.keyClicks(roi1_ctrl._width_le, "10")
        QTest.keyPress(roi1_ctrl._width_le, Qt.Key_Enter)
        roi1_ctrl._height_le.clear()
        QTest.keyClicks(roi1_ctrl._height_le, "30")
        QTest.keyPress(roi1_ctrl._height_le, Qt.Key_Enter)
        self.assertTupleEqual((10, 30), tuple(roi1.size()))

        # ROI can be outside of the image
        roi1_ctrl._px_le.clear()
        QTest.keyClicks(roi1_ctrl._px_le, "-1")
        QTest.keyPress(roi1_ctrl._px_le, Qt.Key_Enter)
        roi1_ctrl._py_le.clear()
        QTest.keyClicks(roi1_ctrl._py_le, "-3")
        QTest.keyPress(roi1_ctrl._py_le, Qt.Key_Enter)
        self.assertTupleEqual((-1, -3), tuple(roi1.pos()))

        proc.update()
        self.assertListEqual([-1, -3, 10, 30], proc._roi1.rect)

        # lock ROI ctrl
        QTest.mouseClick(roi1_ctrl.lock_cb, Qt.LeftButton,
                         pos=QPoint(2, roi1_ctrl.lock_cb.height()/2))
        self.assertTrue(roi1_ctrl.activate_cb.isChecked())
        self.assertTrue(roi1_ctrl.lock_cb.isChecked())
        self.assertFalse(roi1_ctrl._width_le.isEnabled())
        self.assertFalse(roi1_ctrl._height_le.isEnabled())
        self.assertFalse(roi1_ctrl._px_le.isEnabled())
        self.assertFalse(roi1_ctrl._py_le.isEnabled())

        # deactivate ROI ctrl
        QTest.mouseClick(roi1_ctrl.activate_cb, Qt.LeftButton,
                         pos=QPoint(2, roi1_ctrl.activate_cb.height()/2))
        self.assertFalse(roi1_ctrl.activate_cb.isChecked())
        self.assertTrue(roi1_ctrl.lock_cb.isChecked())
        self.assertFalse(roi1_ctrl._width_le.isEnabled())
        self.assertFalse(roi1_ctrl._height_le.isEnabled())
        self.assertFalse(roi1_ctrl._px_le.isEnabled())
        self.assertFalse(roi1_ctrl._py_le.isEnabled())

    def testMovingAverageQLineEdit(self):
        widget = self.image_tool._image_action
        # moving average is disabled for pulse-resolved data
        self.assertFalse(widget.moving_avg_le.isEnabled())

    @patch("karaboFAI.gui.plot_widgets.image_view.ImageAnalysis."
           "onThresholdMaskChange")
    @patch("karaboFAI.gui.mediator.Mediator.onImageThresholdMaskChange")
    def testThresholdMask(self, on_mask_mediator, on_mask):
        widget = self.image_tool._image_action

        widget.threshold_mask_le.clear()
        QTest.keyClicks(widget.threshold_mask_le, "1, 10")
        QTest.keyPress(widget.threshold_mask_le, Qt.Key_Enter)
        on_mask.assert_called_once_with((1, 10))
        on_mask_mediator.assert_called_once_with((1, 10))

    @patch("karaboFAI.gui.plot_widgets.image_view.ImageAnalysis."
           "onBkgChange")
    @patch("karaboFAI.gui.mediator.Mediator.onImageBackgroundChange")
    def testBackground(self, on_bkg_mediator, on_bkg):
        widget = self.image_tool._image_action

        widget.bkg_le.clear()
        QTest.keyClicks(widget.bkg_le, "1.1")
        QTest.keyPress(widget.bkg_le, Qt.Key_Enter)
        on_bkg.assert_called_once_with(1.1)
        on_bkg_mediator.assert_called_once_with(1.1)

    def testAutoLevel(self):
        widget = self.image_tool._image_ctrl_widget

        spy = QSignalSpy(self.image_tool._mediator.reset_image_level_sgn)
        widget.auto_level_btn.clicked.emit()
        self.assertEqual(1, len(spy))

    def testSetAndRemoveReference(self):
        widget = self.image_tool._image_ctrl_widget
        proc = self.image_worker._image_proc_pulse

        data = self._get_data()

        widget.set_ref_btn.clicked.emit()
        proc.process(data)
        self.assertIsNone(proc._reference)

        self.view._image = 2 * np.ones((10, 10), np.float32)
        widget.set_ref_btn.clicked.emit()
        # FIXME: the test could fail randomly
        time.sleep(0.2)
        proc.process(data)
        np.testing.assert_array_equal(self.view._image, proc._reference)

        widget.remove_ref_btn.clicked.emit()
        proc.process(data)
        self.assertIsNone(proc._reference)

        with self.assertRaises(ProcessingError):
            self.view._image = np.ones((2, 2), np.float32)
            widget.set_ref_btn.clicked.emit()
            time.sleep(0.2)
            proc.process(data)

    def testDrawMask(self):
        pass

    def _get_data(self):
        return {'assembled': np.ones((4, 10, 10), np.float32),
                'processed': ProcessedData(1001)}


class TestImageToolTs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()  # ensure file
        config.load('JungFrau')

        cls.fai = FAI().init()
        cls.gui = cls.fai._gui
        cls.scheduler = cls.fai.scheduler

        actions = cls.gui._tool_bar.actions()
        cls._action = actions[2]

        # close the ImageToolWindow opened together with the MainGUI
        window = list(cls.gui._windows.keys())[-1]
        assert isinstance(window, ImageToolWindow)
        window.close()

    @classmethod
    def tearDownClass(cls):
        cls.fai.terminate()

        wait_until_redis_shutdown()

    def setUp(self):
        ImageToolWindow.reset()
        self._action.trigger()
        self.image_tool = list(self.gui._windows.keys())[-1]

    @patch("karaboFAI.gui.mediator.Mediator.onImageMaWindowChange")
    def testImageAction1(self, on_ma_mediator):
        widget = self.image_tool._image_action

        widget.moving_avg_le.clear()
        QTest.keyClicks(widget.moving_avg_le, "10")
        QTest.keyPress(widget.moving_avg_le, Qt.Key_Enter)
        on_ma_mediator.assert_called_once_with(10)
