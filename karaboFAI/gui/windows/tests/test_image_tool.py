import unittest
import os
import tempfile

import numpy as np

from PyQt5 import QtCore
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from karaboFAI.config import _Config, ConfigWrapper
from karaboFAI.services import Fai
from karaboFAI.gui import mkQApp
from karaboFAI.pipeline.data_model import ImageData, ProcessedData


class TestImageTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()  # ensure file

        cls.fai = Fai('LPD')
        cls.fai.init()
        cls.gui = cls.fai.gui
        cls.app = mkQApp()
        cls._tid = 0

        cls._imagetool_action = cls.gui._tool_bar.actions()[2]
        cls._imagetool_action.trigger()

        cls.window = list(cls.gui._windows.keys())[-1]
        cls.view = cls.window._image_view

    @classmethod
    def tearDownClass(cls):
        cls.fai.shutdown()
        cls.gui.close()

    def setUp(self):
        self.__class__._tid += 1
        ImageData.clear()
        self.view.setImageData(None)
        self.view._image = None

    def testImageCtrl(self):
        proc_data = ProcessedData(self._tid,
                                  np.arange(2*128*128).reshape(2, 128, 128))
        self.window._data.set(proc_data)
        self.window.update()

        # ctrl_widget = self.window._image_ctrl
        # spy = QSignalSpy(ctrl_widget._moving_avg_le)
        # self.assertEqual(int(ctrl_widget._moving_avg_le.text()),
        #                  proc_data.image.ma_window)
        # QTest.keyClicks(ctrl_widget._moving_avg_le, "100")
        # QTest.keyPress(ctrl_widget._moving_avg_le, Qt.Key_Enter)
        # self.app.processEvents()
        # self.assertEqual(1, len(spy))
        # proc_data.image.update()
        # self.assertEqual(100, proc_data.image.ma_window)
        #
        # mask_ctrl = self.window._mask_ctrl
        # mask_ctrl._min_pixel_le.setText("1")
        # # self.assertEqual(1, )
        #
        # mask_ctrl._max_pixel_le.setText("100")
        # self.assertEqual()

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

    def testImageOperation(self):
        imgs = np.arange(2*128*128).reshape(2, 128, 128)
        img_mean = np.mean(imgs, axis=0)
        proc_data = ProcessedData(self._tid, imgs)
        self.window._data.set(proc_data)

        self.assertEqual(None, self.view._image)
        self.window.update()
        np.testing.assert_array_equal(img_mean, self.view._image)

        # test 'update_image_btn'
        imgs = 2 * np.arange(2*128*128).reshape(2, 128, 128)
        img_mean = np.mean(imgs, axis=0)
        proc_data = ProcessedData(self._tid, imgs)
        self.window._data.set(proc_data)
        self.window._image_proc_widget.update_image_btn.clicked.emit()
        np.testing.assert_array_equal(img_mean, self.view._image)
