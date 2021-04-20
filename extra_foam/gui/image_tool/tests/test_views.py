import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.data_model import ProcessedData
from extra_foam.gui import mkQApp
from extra_foam.gui.image_tool.corrected_view import CorrectedView
from extra_foam.gui.image_tool.azimuthal_integ_1d_view import AzimuthalInteg1dView
from extra_foam.gui.image_tool.transform_view import TransformView, ImageTransformCtrlWidget
from extra_foam.config import ImageTransformType
from extra_foam.logger import logger


app = mkQApp()

logger.setLevel('CRITICAL')


class TestViews(_TestDataMixin, unittest.TestCase):

    class MockedImageToolWindow(QMainWindow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ctrl_widgets = []

        def createCtrlWidget(self, widget_class, *args, **kwargs):
            widget = widget_class(*args, pulse_resolved=True, require_geometry=True, **kwargs)
            self._ctrl_widgets.append(widget)
            return widget

    @classmethod
    def setUpClass(cls):
        cls.gui = cls.MockedImageToolWindow()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testCorrectedView(self):

        # pulse-resolved

        view = CorrectedView(pulse_resolved=True, parent=self.gui)

        # empty data
        data = ProcessedData(1)
        view.updateF(data, True)

        # non-empty data
        data = self.processed_data(1001, (2, 2), roi_histogram=True)
        view.updateF(data, True)

        # train-resolved

        view = CorrectedView(pulse_resolved=False, parent=self.gui)
        # empty data
        data = ProcessedData(1)
        view.updateF(data, True)

        # non-empty data
        data = self.processed_data(1001, (2, 2), roi_histogram=True)
        view.updateF(data, True)

    def testAzimuthalIntegrationView(self):

        # pulse-resolved

        view = AzimuthalInteg1dView(pulse_resolved=True, parent=self.gui)
        # empty data
        data = ProcessedData(1)
        view.updateF(data, True)

        # non-empty data
        data = self.processed_data(1001, (2, 2))
        data.ai.x = np.arange(10)
        data.ai.y = np.arange(10)
        view.updateF(data, True)
        data.ai.peaks = np.arange(10)
        with patch.object(view._azimuthal_integ_1d_curve, "setAnnotationList") as mocked:
            view._azimuthal_integ_1d_curve.updateF(data)
            mocked.assert_called_once()

        with patch.object(view._azimuthal_integ_1d_curve, "setFitted") as mocked:
            view._onCurveFit()
            mocked.assert_called_once()

    @patch("extra_foam.gui.mediator.Mediator.onItTransformTypeChange")
    def testTransformedView(self, mocked):

        view = TransformView(pulse_resolved=True, parent=self.gui)
        view.onActivated()
        opt_tab = view._ctrl_widget._opt_tab

        # ------------------------------
        # test concentric ring detection
        # ------------------------------
        self.assertEqual(int(ImageTransformType.CONCENTRIC_RINGS), opt_tab.currentIndex())
        self.assertEqual(ImageTransformType.CONCENTRIC_RINGS, view._transform_type)
        opt_tab.currentChanged.emit(opt_tab.currentIndex())
        self.assertTrue(view._ring_item.isVisible())

        ctrl_widget = view._ctrl_widget._concentric_rings

        with patch.object(ctrl_widget, "extractFeature") as mockedExtractFeature:
            with patch.object(view._ring_item, "setGeometry") as mockedSetGeometry:

                data = self.processed_data(1001, (2, 2))

                # empty data
                data.image.masked_mean = None
                view.updateF(data, True)
                QTest.mouseClick(ctrl_widget.detect_btn, Qt.LeftButton)

                # non-empty data
                image_data = np.random.randn(2, 2).astype(float)
                data.image.transform_type = ImageTransformType.CONCENTRIC_RINGS
                data.image.masked_mean = image_data
                data.image.transformed = None
                view.updateF(data, True)
                np.testing.assert_array_equal(image_data, view._corrected.image)
                np.testing.assert_array_equal(image_data, view._transformed.image)

                mockedExtractFeature.return_value = (1, 2, [50, 100])
                QTest.mouseClick(ctrl_widget.detect_btn, Qt.LeftButton)
                mockedExtractFeature.assert_called_once_with(image_data)
                mockedSetGeometry.assert_called_once_with(1, 2, [50, 100])

        # ------------------------------
        # test fourier transform
        # ------------------------------
        view._ctrl_widget._opt_tab.setCurrentIndex(int(ImageTransformType.FOURIER_TRANSFORM))
        self.assertEqual(ImageTransformType.FOURIER_TRANSFORM, view._transform_type)
        self.assertFalse(view._ring_item.isVisible())

        # ------------------------------
        # test edge detection
        # ------------------------------
        view._ctrl_widget._opt_tab.setCurrentIndex(int(ImageTransformType.EDGE_DETECTION))
        self.assertEqual(ImageTransformType.EDGE_DETECTION, view._transform_type)
        self.assertFalse(view._ring_item.isVisible())

        # ------------------------------
        # back to concentric rings
        # ------------------------------
        view._ctrl_widget._opt_tab.setCurrentIndex(int(ImageTransformType.CONCENTRIC_RINGS))
        self.assertEqual(ImageTransformType.CONCENTRIC_RINGS, view._transform_type)
        self.assertTrue(view._ring_item.isVisible())
