import unittest

import numpy as np

from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from karaboFAI.services import FaiServer
from karaboFAI.pipeline.data_model import ImageData, ProcessedData


class TestImageTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = FaiServer('LPD')._gui
        cls._tid = 0

        cls._imagetool_action = cls.gui._tool_bar.actions()[2]
        cls._imagetool_action.trigger()

        cls.window = list(cls.gui._windows.keys())[-1]
        cls.view = cls.window._image_view

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def setUp(self):
        self.__class__._tid += 1
        ImageData.clear()

    def test_imageOperation(self):
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

    def test_roiOperation(self):
        # test clear ROI history
        widget = self.window._roi_ctrl_widget
        QTest.mouseClick(widget._clear_roi_hist_btn, Qt.LeftButton)
