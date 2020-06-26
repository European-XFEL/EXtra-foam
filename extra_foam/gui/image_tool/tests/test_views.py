import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

from PyQt5.QtWidgets import QMainWindow, QWidget

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.data_model import ProcessedData
from extra_foam.gui import mkQApp
from extra_foam.gui.image_tool.corrected_view import CorrectedView
from extra_foam.gui.image_tool.azimuthal_integ_1d_view import AzimuthalInteg1dView
from extra_foam.logger import logger


app = mkQApp()

logger.setLevel('CRITICAL')


class TestViews(_TestDataMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gui = QMainWindow()  # dummy MainGUI
        widget = QWidget()
        widget.setRois = MagicMock()
        cls.gui.createCtrlWidget = MagicMock(return_value=widget)

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testCorrectedView(self):

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

    def testAzimuthalIntegrationView(self):

        # pulse-resolved

        view = AzimuthalInteg1dView(pulse_resolved=True, parent=self.gui)
        # empty data
        data = ProcessedData(1)
        view._corrected.updateF(data)
        view._azimuthal_integ_1d_curve.updateF(data)
        view._q_view.updateF(data)

        # non-empty data
        data = self.processed_data(1001, (2, 2))
        data.ai.x = np.arange(10)
        data.ai.y = np.arange(10)
        view._azimuthal_integ_1d_curve.updateF(data)
        data.ai.peaks = np.arange(10)
        with patch.object(view._azimuthal_integ_1d_curve, "setAnnotationList") as mocked:
            view._azimuthal_integ_1d_curve.updateF(data)
            mocked.assert_called_once()
