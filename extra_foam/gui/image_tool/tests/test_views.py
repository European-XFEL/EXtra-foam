import unittest
from unittest.mock import MagicMock, patch

from PyQt5.QtWidgets import QMainWindow, QWidget

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.data_model import ProcessedData
from extra_foam.gui import mkQApp
from extra_foam.logger import logger


app = mkQApp()

logger.setLevel('CRITICAL')


class TestViews(_TestDataMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gui = QMainWindow()  # dummy MainGUI
        cls.gui.createCtrlWidget = MagicMock(return_value=QWidget())

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testCorrectedView(self):
        from extra_foam.gui.image_tool.corrected_view import CorrectedView

        # pulse-resolved

        view = CorrectedView(pulse_resolved=True, parent=self.gui)
        # empty data
        data = ProcessedData(1)
        view._corrected.updateF(data)
        view._roi_hist.updateF(data)
        # non-empty data
        data = self.processed_data(1001, (2, 2), roi_histogram=True)
        view._roi_hist.updateF(data)

        # train-resolved

        view = CorrectedView(pulse_resolved=False, parent=self.gui)
        # empty data
        data = ProcessedData(1)
        view._roi_hist.updateF(data)
        # non-empty data
        data = self.processed_data(1001, (2, 2), roi_histogram=True)
        view._roi_hist.updateF(data)
