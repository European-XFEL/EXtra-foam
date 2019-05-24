import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile

import numpy as np

from PyQt5 import QtCore
from PyQt5.QtTest import QTest, QSignalSpy
from PyQt5.QtCore import Qt

from karaboFAI.config import _Config, ConfigWrapper
from karaboFAI.services import FAI
from karaboFAI.pipeline.data_model import ImageData
from karaboFAI.gui.windows import ImageToolWindow
from karaboFAI.gui.windows.image_tool import _SimpleImageData


class TestSimpleImageData(unittest.TestCase):
    def testGeneral(self):
        with self.assertRaises(TypeError):
            _SimpleImageData([1, 2, 3])

        with self.assertRaises(ValueError):
            _SimpleImageData(np.ones([2, 2, 2]))

        orig_data = np.arange(9).reshape(3, 3)
        img_data = _SimpleImageData(orig_data)

        img_data.background = 1
        np.testing.assert_array_equal(orig_data - 1, img_data.masked)
        img_data.background = 0
        np.testing.assert_array_equal(orig_data, img_data.masked)

        with self.assertRaises(TypeError):
            img_data.threshold_mask = 1

        with self.assertRaises(ValueError):
            img_data.threshold_mask = [1, 2, 3]

        img_data.threshold_mask = (3, 6)
        np.testing.assert_array_equal(np.clip(orig_data, 3, 6), img_data.masked)
        img_data.background = 3
        np.testing.assert_array_equal(np.clip(orig_data-3, 3, 6), img_data.masked)

    def testImageWithNan(self):
        orig_data = np.array([[1, 0], [np.nan, np.nan]])
        img_data = _SimpleImageData(orig_data)

        img_data.background = -1
        np.testing.assert_array_equal(np.array([[2, 1], [np.nan, np.nan]]),
                                      img_data.masked)

        img_data.threshold_mask = (1.1, 1.6)
        np.testing.assert_array_almost_equal(
            np.array([[1.6, 1.1], [np.nan, np.nan]]), img_data.masked)


class TestImageTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()  # ensure file

        fai = FAI('LPD')
        fai.init()
        cls.gui = fai.gui
        cls.app = fai.app
        cls.fai = fai

        cls._imagetool_action = cls.gui._tool_bar.actions()[2]
        cls._imagetool_action.trigger()

        cls.window = list(cls.gui._windows.keys())[-1]
        cls.view = cls.window._image_view

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()
        ImageToolWindow._reset()

        del cls.fai

    def setUp(self):
        ImageData.clear()
        self.view.setImageData(None)
        self.view._image = None

    def testRoiCtrl(self):
        roi_widget = self.window._roi_ctrl_widget
        roi_ctrls = roi_widget._roi_ctrls
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
                         pos=QtCore.QPoint(2, roi1_ctrl.activate_cb.height()/2))
        self.assertTrue(roi1_ctrl.activate_cb.isChecked())

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

        # lock ROI ctrl
        QTest.mouseClick(roi1_ctrl.lock_cb, Qt.LeftButton,
                         pos=QtCore.QPoint(2, roi1_ctrl.lock_cb.height()/2))
        self.assertTrue(roi1_ctrl.activate_cb.isChecked())
        self.assertTrue(roi1_ctrl.lock_cb.isChecked())
        self.assertFalse(roi1_ctrl._width_le.isEnabled())
        self.assertFalse(roi1_ctrl._height_le.isEnabled())
        self.assertFalse(roi1_ctrl._px_le.isEnabled())
        self.assertFalse(roi1_ctrl._py_le.isEnabled())

        # deactivate ROI ctrl
        QTest.mouseClick(roi1_ctrl.activate_cb, Qt.LeftButton,
                         pos=QtCore.QPoint(2, roi1_ctrl.activate_cb.height()/2))
        self.assertFalse(roi1_ctrl.activate_cb.isChecked())
        self.assertTrue(roi1_ctrl.lock_cb.isChecked())
        self.assertFalse(roi1_ctrl._width_le.isEnabled())
        self.assertFalse(roi1_ctrl._height_le.isEnabled())
        self.assertFalse(roi1_ctrl._px_le.isEnabled())
        self.assertFalse(roi1_ctrl._py_le.isEnabled())

    @patch("karaboFAI.gui.plot_widgets.image_view.ImageAnalysis."
           "onMovingAverageWindowChange")
    @patch("karaboFAI.gui.mediator.Mediator.onImageMaWindowChange")
    def testImageAction(self, on_ma_mediator, on_ma):
        widget = self.window._image_action

        widget.moving_avg_le.clear()
        QTest.keyClicks(widget.moving_avg_le, "10")
        QTest.keyPress(widget.moving_avg_le, Qt.Key_Enter)
        on_ma.assert_called_once_with(10)
        on_ma_mediator.assert_called_once_with(10)

    @patch("karaboFAI.gui.plot_widgets.image_view.ImageAnalysis."
           "onThresholdMaskChange")
    @patch("karaboFAI.gui.mediator.Mediator.onImageThresholdMaskChange")
    def testImageAction2(self, on_mask_mediator, on_mask):
        widget = self.window._image_action

        widget.threshold_mask_le.clear()
        QTest.keyClicks(widget.threshold_mask_le, "1, 10")
        QTest.keyPress(widget.threshold_mask_le, Qt.Key_Enter)
        on_mask.assert_called_once_with((1, 10))
        on_mask_mediator.assert_called_once_with((1, 10))

    @patch("karaboFAI.gui.plot_widgets.image_view.ImageAnalysis."
           "onBkgChange")
    @patch("karaboFAI.gui.mediator.Mediator.onImageBackgroundChange")
    def testImageAction3(self, on_bkg_mediator, on_bkg):
        widget = self.window._image_action

        widget.bkg_le.clear()
        QTest.keyClicks(widget.bkg_le, "1.1")
        QTest.keyPress(widget.bkg_le, Qt.Key_Enter)
        on_bkg.assert_called_once_with(1.1)
        on_bkg_mediator.assert_called_once_with(1.1)

    def testImageCtrl(self):
        widget = self.window._image_ctrl_widget

        spy = QSignalSpy(self.window._mediator.reset_image_level_sgn)
        widget.auto_level_btn.clicked.emit()
        self.assertEqual(1, len(spy))
