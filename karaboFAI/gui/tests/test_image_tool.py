import unittest

import numpy as np

from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from karaboFAI.gui.main_gui import MainGUI

from karaboFAI.pipeline.data_model import ImageData, ProcessedData

from . import mkQApp
app = mkQApp()


class TestImageTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = MainGUI('LPD')
        cls.proc = cls.gui._proc_worker
        cls._tid = 0

        cls._imagetool_action = cls.gui._tool_bar.actions()[2]

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def setUp(self):
        self.__class__._tid += 1
        ImageData.reset()

    def test_imagetoolwindow(self):
        n_registered = len(self.gui._windows)
        self._imagetool_action.trigger()

        window = list(self.gui._windows.keys())[-1]
        view = window._image_view
        # TODO: ImageToolWindow is a SingletonWindow
        # self.assertIsInstance(window, ImageToolWindow)
        self.assertEqual(n_registered + 1, len(self.gui._windows))

        imgs = np.arange(4*128*128).reshape(4, 128, 128)
        img_mean = np.mean(imgs, axis=0)
        proc_data = ProcessedData(self._tid, imgs)
        window._data.set(proc_data)

        self.assertEqual(None, view._image)
        window.update()
        np.testing.assert_array_equal(img_mean, view._image)
