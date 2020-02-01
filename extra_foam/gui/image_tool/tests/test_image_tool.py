import unittest
from unittest.mock import MagicMock, patch
import math
import os
import tempfile
import time

import numpy as np

from PyQt5.QtTest import QTest, QSignalSpy
from PyQt5.QtCore import Qt, QPoint

from extra_foam.config import (
    AnalysisType, config, Normalizer, RoiCombo, RoiFom
)
from extra_foam.gui import mkQApp
from extra_foam.gui.image_tool import ImageToolWindow
from extra_foam.logger import logger
from extra_foam.pipeline.data_model import ImageData, ProcessedData, RectRoiGeom
from extra_foam.pipeline.exceptions import ImageProcessingError
from extra_foam.pipeline.processors.tests import _BaseProcessorTest
from extra_foam.processes import wait_until_redis_shutdown
from extra_foam.services import Foam
from extra_foam.database import Metadata, MetaProxy

app = mkQApp()

logger.setLevel('CRITICAL')

_tmp_cfg_dir = tempfile.mkdtemp()


def setup_module(module):
    from extra_foam import config
    module._backup_ROOT_PATH = config.ROOT_PATH
    config.ROOT_PATH = _tmp_cfg_dir


def teardown_module(module):
    os.rmdir(_tmp_cfg_dir)
    from extra_foam import config
    config.ROOT_PATH = module._backup_ROOT_PATH


class TestImageTool(unittest.TestCase, _BaseProcessorTest):
    @classmethod
    def setUpClass(cls):
        config.load('LPD', 'FXE')

        cls.foam = Foam().init()
        cls.gui = cls.foam._gui
        cls.train_worker = cls.foam.train_worker
        cls.pulse_worker = cls.foam.pulse_worker

        cls._meta = MetaProxy()

        actions = cls.gui._tool_bar.actions()
        cls._action = actions[3]
        assert("Image tool" == cls._action.text())

    @classmethod
    def tearDownClass(cls):
        cls.foam.terminate()

        wait_until_redis_shutdown()

        os.remove(config.config_file)

    def setUp(self):
        # construct a fresh ImageToolWindow for each test
        self.gui._image_tool = ImageToolWindow(queue=self.gui._queue,
                                               pulse_resolved=self.gui._pulse_resolved,
                                               parent=self.gui)
        self.image_tool = self.gui._image_tool

        self.view = self.image_tool._corrected_view.imageView
        self.view.setImageData(None)
        self.view._image = None

    def testGeneral(self):
        self.assertEqual(9, len(self.image_tool._ctrl_widgets))
        self.assertTrue(self.image_tool._pulse_resolved)
        self.assertTrue(self.image_tool._image_ctrl_widget._pulse_resolved)

    def testUpdateImage(self):
        widget = self.image_tool._image_ctrl_widget

        # test default values
        self.assertFalse(widget.update_image_btn.isEnabled())
        self.assertTrue(widget.auto_update_cb.isChecked())
        self.assertTrue(self.image_tool._auto_update)

        # test enabled and disable "update image" button
        widget.auto_update_cb.setChecked(False)
        self.assertTrue(widget.update_image_btn.isEnabled())
        self.assertFalse(self.image_tool._auto_update)
        widget.auto_update_cb.setChecked(True)
        self.assertFalse(widget.update_image_btn.isEnabled())
        self.assertTrue(self.image_tool._auto_update)

        # test update image manually
        self.image_tool.updateWidgets = MagicMock()
        widget.auto_update_cb.setChecked(False)
        widget.update_image_btn.clicked.emit()
        self.image_tool.updateWidgets.assert_called_once_with(True)

    def testRoiCtrlWidget(self):
        roi_ctrls = self.image_tool._corrected_view._roi_ctrl_widget._roi_ctrls
        proc = self.pulse_worker._roi_proc
        self.assertEqual(4, len(roi_ctrls))

        proc.update()

        for i, ctrl in enumerate(roi_ctrls, 1):
            # test real ROI position and size matches the numbers in the GUI
            self.assertListEqual([int(ctrl._px_le.text()), int(ctrl._py_le.text())],
                                 list(ctrl._roi.pos()))
            self.assertListEqual([int(ctrl._width_le.text()), int(ctrl._height_le.text())],
                                 list(ctrl._roi.size()))
            # test default values
            self.assertListEqual(RectRoiGeom.INVALID, getattr(proc, f"_geom{i}"))

        for ctrl in roi_ctrls:
            self.assertFalse(ctrl._activate_cb.isChecked())
            self.assertFalse(ctrl._lock_cb.isChecked())
            self.assertFalse(ctrl._width_le.isEnabled())
            self.assertFalse(ctrl._height_le.isEnabled())
            self.assertFalse(ctrl._px_le.isEnabled())
            self.assertFalse(ctrl._py_le.isEnabled())

        roi1_ctrl = roi_ctrls[0]
        roi1 = self.view._rois[0]
        self.assertIs(roi1_ctrl._roi, roi1)

        # activate ROI1 ctrl
        QTest.mouseClick(roi1_ctrl._activate_cb, Qt.LeftButton,
                         pos=QPoint(2, roi1_ctrl._activate_cb.height()/2))
        self.assertTrue(roi1_ctrl._activate_cb.isChecked())
        proc.update()

        self.assertTupleEqual((int(roi1_ctrl._width_le.text()), int(roi1_ctrl._height_le.text())),
                              tuple(roi1.size()))
        self.assertTupleEqual((int(roi1_ctrl._px_le.text()), int(roi1_ctrl._py_le.text())),
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
        self.assertListEqual([-1, -3, 10, 30], proc._geom1)

        # lock ROI ctrl
        QTest.mouseClick(roi1_ctrl._lock_cb, Qt.LeftButton,
                         pos=QPoint(2, roi1_ctrl._lock_cb.height()/2))
        self.assertTrue(roi1_ctrl._activate_cb.isChecked())
        self.assertTrue(roi1_ctrl._lock_cb.isChecked())
        self.assertFalse(roi1_ctrl._width_le.isEnabled())
        self.assertFalse(roi1_ctrl._height_le.isEnabled())
        self.assertFalse(roi1_ctrl._px_le.isEnabled())
        self.assertFalse(roi1_ctrl._py_le.isEnabled())

        # deactivate ROI ctrl
        QTest.mouseClick(roi1_ctrl._activate_cb, Qt.LeftButton,
                         pos=QPoint(2, roi1_ctrl._activate_cb.height()/2))
        self.assertFalse(roi1_ctrl._activate_cb.isChecked())
        self.assertTrue(roi1_ctrl._lock_cb.isChecked())
        self.assertFalse(roi1_ctrl._width_le.isEnabled())
        self.assertFalse(roi1_ctrl._height_le.isEnabled())
        self.assertFalse(roi1_ctrl._px_le.isEnabled())
        self.assertFalse(roi1_ctrl._py_le.isEnabled())

    def testMovingAverageQLineEdit(self):
        # TODO: remove it in the future
        widget = self.image_tool._image_ctrl_widget
        # moving average is disabled
        self.assertFalse(widget.moving_avg_le.isEnabled())

    @patch("extra_foam.gui.plot_widgets.image_views.ImageAnalysis."
           "onThresholdMaskChange")
    @patch("extra_foam.gui.mediator.Mediator.onImageThresholdMaskChange")
    def testThresholdMask(self, on_mask_mediator, on_mask):
        widget = self.image_tool._image_ctrl_widget

        widget.threshold_mask_le.clear()
        QTest.keyClicks(widget.threshold_mask_le, "1, 10")
        QTest.keyPress(widget.threshold_mask_le, Qt.Key_Enter)
        on_mask.assert_called_once_with((1, 10))
        on_mask_mediator.assert_called_once_with((1, 10))

    def testAutoLevel(self):
        widget = self.image_tool._image_ctrl_widget

        spy = QSignalSpy(self.image_tool._mediator.reset_image_level_sgn)
        widget.auto_level_btn.clicked.emit()
        self.assertEqual(1, len(spy))

    def testReferenceCtrlWidget(self):
        widget = self.image_tool._reference_view._ctrl_widget
        corrected = self.image_tool._reference_view._corrected
        proc = self.pulse_worker._image_proc

        data, _ = self.data_with_assembled(1001, (4, 10, 10))

        # test setting reference (no image)
        QTest.mouseClick(widget._set_ref_btn, Qt.LeftButton)
        ref = proc._ref_sub.update(proc._reference)
        self.assertIsNone(ref)

        # test setting reference
        corrected._image = 2 * np.ones((10, 10), np.float32)
        QTest.mouseClick(widget._set_ref_btn, Qt.LeftButton)
        ref = proc._ref_sub.update(corrected.image.copy())
        np.testing.assert_array_equal(corrected.image, ref)

        # test setting reference multiple times
        for i in range(5):
            corrected._image = np.random.rand(10, 10).astype(np.float32)
            QTest.mouseClick(widget._set_ref_btn, Qt.LeftButton)
        ref = proc._ref_sub.update(None)
        np.testing.assert_array_equal(corrected.image, ref)

        # test removing reference
        QTest.mouseClick(widget._remove_ref_btn, Qt.LeftButton)
        ref = proc._ref_sub.update(corrected.image.copy())
        self.assertIsNone(ref)

        # ------------------------------
        # test load and remove reference
        # ------------------------------

        # Here we test that "proc._ref_sub.update()" works properly. The rest
        # is done in the unittests of ImageProcessor.

        ref_gt = np.ones([2, 2], dtype=np.float32)

        def _read_image_side_effect(fn):
            if fn == "reference/file/path":
                return ref_gt

        # caveat: first establish the connection
        proc._cal_sub.update(None, None)

        with patch('extra_foam.gui.ctrl_widgets.ref_image_ctrl_widget.read_image',
                   side_effect=_read_image_side_effect):
            with patch('extra_foam.gui.ctrl_widgets.ref_image_ctrl_widget.QFileDialog.getOpenFileName',
                       return_value=["reference/file/path"]):
                QTest.mouseClick(widget._load_ref_btn, Qt.LeftButton)
                self.assertEqual("reference/file/path", widget._ref_fp_le.text())
                ref = proc._ref_sub.update(None)
                np.testing.assert_array_equal(ref, ref_gt)

                QTest.mouseClick(widget._remove_ref_btn, Qt.LeftButton)
                self.assertEqual("", widget._ref_fp_le.text())
                ref = proc._ref_sub.update(ref_gt)
                self.assertIsNone(ref)

    def testDrawMask(self):
        # TODO: test by really drawing something on ImageTool
        from extra_foam.ipc import ImageMaskPub

        pub = ImageMaskPub()
        proc = self.pulse_worker._image_proc
        data, _ = self.data_with_assembled(1001, (4, 10, 10))

        # trigger the lazily evaluated subscriber
        proc.process(data)
        self.assertIsNone(proc._image_mask)

        mask_gt = np.zeros(data['assembled']['data'].shape[-2:], dtype=np.bool)

        pub.add((0, 0, 2, 3))
        mask_gt[0:3, 0:2] = True

        # test adding mask
        n_attempts = 0
        # repeat to prevent random failure
        while n_attempts < 10:
            n_attempts += 1

            proc.process(data)
            # np.testing.assert_array_equal(mask_gt, proc._image_mask)
            if (mask_gt == proc._image_mask).all():
                break
            time.sleep(0.001)

        # add one more mask region
        pub.add((1, 1, 2, 3))
        proc.process(data)
        mask_gt[1:4, 1:3] = True
        np.testing.assert_array_equal(mask_gt, proc._image_mask)

        # test erasing mask
        pub.erase((2, 2, 3, 3))
        proc.process(data)
        mask_gt[2:5, 2:5] = False
        np.testing.assert_array_equal(mask_gt, proc._image_mask)

        # test trashing mask
        action = self.image_tool._tool_bar.actions()[2]
        action.trigger()
        proc.process(data)
        self.assertIsNone(proc._image_mask)

        # test set mask
        pub.set(mask_gt)
        proc.process(data)
        np.testing.assert_array_equal(mask_gt, proc._image_mask)

        # test set a mask which has a different shape from the image
        mask_gt = np.zeros((2, 2), dtype=np.bool)
        pub.set(mask_gt)
        with self.assertRaises(ImageProcessingError):
            proc.process(data)

    def testBulletinView(self):
        processed = ProcessedData(1357)

        processed.image = ImageData.from_array(np.ones((10, 4, 4), np.float32))
        processed.image.dark_count = 99
        processed.image.n_dark_pulses = 10
        processed.pidx.mask([1, 3, 5, 6])
        self.gui._queue.append(processed)
        self.image_tool.updateWidgetsF()

        view = self.image_tool._bulletin_view
        self.assertEqual(1357, int(view._latest_tid.intValue()))
        self.assertEqual(10, int(view._n_total_pulses.intValue()))
        self.assertEqual(6, int(view._n_kept_pulses.intValue()))
        self.assertEqual(99, int(view._dark_train_counter.intValue()))
        self.assertEqual(10, int(view._n_dark_pulses.intValue()))

    def testCalibrationCtrlWidget(self):
        widget = self.image_tool._gain_offset_view._ctrl_widget

        proc = self.pulse_worker._image_proc

        proc.update()
        self.assertTrue(proc._correct_gain)
        self.assertTrue(proc._correct_offset)
        self.assertEqual(slice(None), proc._gain_slicer)
        self.assertEqual(slice(None), proc._offset_slicer)
        self.assertTrue(proc._dark_as_offset)
        self.assertFalse(proc._recording_dark)

        widget._correct_gain_cb.setChecked(False)
        widget._correct_offset_cb.setChecked(False)
        widget._gain_slicer_le.setText(":70")
        widget._offset_slicer_le.setText("2:120:4")
        widget._dark_as_offset_cb.setChecked(False)
        QTest.mouseClick(widget._record_dark_btn, Qt.LeftButton)
        proc.update()
        self.assertFalse(proc._correct_gain)
        self.assertFalse(proc._correct_offset)
        self.assertEqual(slice(None, 70), proc._gain_slicer)
        self.assertEqual(slice(2, 120, 4), proc._offset_slicer)
        self.assertFalse(proc._dark_as_offset)
        self.assertTrue(proc._recording_dark)

        # test stop dark recording
        QTest.mouseClick(widget._record_dark_btn, Qt.LeftButton)
        proc.update()
        self.assertFalse(proc._recording_dark)

        # test remove dark
        data = np.ones((10, 10), dtype=np.float32)
        proc._dark = data
        proc._dark_mean = data
        QTest.mouseClick(widget._remove_dark_btn, Qt.LeftButton)
        proc.update()
        self.assertIsNone(proc._dark)
        self.assertIsNone(proc._dark_mean)

        # --------------------------------
        # test load and remove gain/offset
        # --------------------------------

        # Here we test that "proc._cal_sub.update()" works properly. The rest
        # is done in the unittests of ImageProcessor.

        const_gt = np.ones([2, 2])

        def _read_constants_side_effect(fn):
            if fn in ["gain/file/path", "offset/file/path"]:
                return const_gt

        # caveat: first establish the connection
        proc._cal_sub.update(None, None)

        with patch('extra_foam.ipc.read_cal_constants', side_effect=_read_constants_side_effect):
            with patch('extra_foam.gui.ctrl_widgets.calibration_ctrl_widget.QFileDialog.getOpenFileName',
                       return_value=["gain/file/path"]):
                QTest.mouseClick(widget._load_gain_btn, Qt.LeftButton)
                time.sleep(0.1)  # wait to write into redis
                self.assertEqual("gain/file/path", widget._gain_fp_le.text())

                n_attempts = 0
                # repeat to prevent random failure at Travis
                while n_attempts < 10:
                    n_attempts += 1
                    new_gain, gain, new_offset, offset = proc._cal_sub.update(None, None)
                    if new_gain:
                        break
                    time.sleep(0.001)

                np.testing.assert_array_equal(gain, const_gt)
                self.assertFalse(new_offset)
                self.assertIsNone(offset)

                QTest.mouseClick(widget._remove_gain_btn, Qt.LeftButton)
                self.assertEqual("", widget._gain_fp_le.text())
                new_gain, gain, new_offset, offset = proc._cal_sub.update(const_gt, None)
                self.assertTrue(new_gain)
                self.assertIsNone(gain)
                self.assertFalse(new_offset)
                self.assertIsNone(offset)

            with patch('extra_foam.gui.ctrl_widgets.calibration_ctrl_widget.QFileDialog.getOpenFileName',
                       return_value=["offset/file/path"]):
                proc._gain = const_gt

                QTest.mouseClick(widget._load_offset_btn, Qt.LeftButton)
                time.sleep(0.1)  # wait to write data into redis
                self.assertEqual("offset/file/path", widget._offset_fp_le.text())
                new_gain, gain, new_offset, offset = proc._cal_sub.update(const_gt, None)
                self.assertFalse(new_gain)
                np.testing.assert_array_equal(gain, const_gt)
                self.assertTrue(new_offset)
                np.testing.assert_array_equal(offset, const_gt)

                QTest.mouseClick(widget._remove_offset_btn, Qt.LeftButton)
                self.assertEqual("", widget._offset_fp_le.text())
                new_gain, gain, new_offset, offset = proc._cal_sub.update(const_gt, const_gt)
                self.assertFalse(new_gain)
                np.testing.assert_array_equal(gain, const_gt)
                self.assertTrue(new_offset)
                self.assertIsNone(offset)

    def testAzimuthalInteg1dCtrlWidget(self):
        from extra_foam.pipeline.processors.azimuthal_integration import energy2wavelength
        from extra_foam.gui.ctrl_widgets.azimuthal_integ_ctrl_widget import \
            _DEFAULT_AZIMUTHAL_INTEG_POINTS

        widget = self.image_tool._azimuthal_integ_1d_view._ctrl_widget
        avail_norms = {value: key for key, value in widget._available_norms.items()}
        train_worker = self.train_worker
        proc = train_worker._ai_proc

        proc.update()

        self.assertAlmostEqual(config['SAMPLE_DISTANCE'], proc._sample_dist)
        self.assertAlmostEqual(0.001 * energy2wavelength(config['PHOTON_ENERGY']), proc._wavelength)
        self.assertEqual(AnalysisType.UNDEFINED, proc.analysis_type)
        default_integ_method = 'BBox'
        self.assertEqual(default_integ_method, proc._integ_method)
        default_normalizer = Normalizer.UNDEFINED
        self.assertEqual(default_normalizer, proc._normalizer)
        self.assertEqual(_DEFAULT_AZIMUTHAL_INTEG_POINTS, proc._integ_points)
        self.assertTupleEqual((0, math.inf), proc._auc_range)
        self.assertTupleEqual((0, math.inf), proc._fom_integ_range)
        default_pixel_size = config["PIXEL_SIZE"]
        self.assertEqual(default_pixel_size, proc._pixel1)
        self.assertEqual(default_pixel_size, proc._pixel2)
        self.assertEqual(0, proc._poni1)
        self.assertEqual(0, proc._poni2)

        widget._photon_energy_le.setText("12.4")
        widget._sample_dist_le.setText("0.3")
        widget._integ_method_cb.setCurrentText('nosplit_csr')
        widget._norm_cb.setCurrentText(avail_norms[Normalizer.ROI])
        widget._integ_pts_le.setText(str(1024))
        widget._integ_range_le.setText("0.1, 0.2")
        widget._auc_range_le.setText("0.2, 0.3")
        widget._fom_integ_range_le.setText("0.3, 0.4")
        widget._px_le.setText("0.000001")
        widget._py_le.setText("0.000002")
        widget._cx_le.setText("-1000")
        widget._cy_le.setText("1000")
        proc.update()
        self.assertAlmostEqual(1e-10, proc._wavelength)
        self.assertAlmostEqual(0.3, proc._sample_dist)
        self.assertEqual('nosplit_csr', proc._integ_method)
        self.assertEqual(Normalizer.ROI, proc._normalizer)
        self.assertEqual(1024, proc._integ_points)
        self.assertTupleEqual((0.1, 0.2), proc._integ_range)
        self.assertTupleEqual((0.2, 0.3), proc._auc_range)
        self.assertTupleEqual((0.3, 0.4), proc._fom_integ_range)
        self.assertEqual(0.000001, proc._pixel2)
        self.assertEqual(0.000002, proc._pixel1)
        self.assertEqual(-1000 * 0.000001, proc._poni2)
        self.assertEqual(1000 * 0.000002, proc._poni1)

    def testRoiFomCtrlWidget(self):
        widget = self.image_tool._corrected_view._roi_fom_ctrl_widget
        avail_norms = {value: key for key, value in widget._available_norms.items()}
        avail_combos = {value: key for key, value in widget._available_combos.items()}
        avail_types = {value: key for key, value in widget._available_types.items()}

        proc = self.train_worker._roi_proc
        proc.update()

        # test default reconfigurable values
        self.assertEqual(RoiCombo.ROI1, proc._fom_combo)
        self.assertEqual(RoiFom.SUM, proc._fom_type)
        self.assertEqual(Normalizer.UNDEFINED, proc._fom_norm)

        # test setting new values
        widget._combo_cb.setCurrentText(avail_combos[RoiCombo.ROI1_SUB_ROI2])
        widget._type_cb.setCurrentText(avail_types[RoiFom.MEDIAN])
        widget._norm_cb.setCurrentText(avail_norms[Normalizer.ROI])

        proc.update()

        self.assertEqual(RoiCombo.ROI1_SUB_ROI2, proc._fom_combo)
        self.assertEqual(RoiFom.MEDIAN, proc._fom_type)
        self.assertEqual(Normalizer.ROI, proc._fom_norm)

    def testRoiNormCtrlWidget(self):
        widget = self.image_tool._corrected_view._roi_norm_ctrl_widget
        avail_combos = {value: key for key, value in widget._available_combos.items()}
        avail_types = {value: key for key, value in widget._available_types.items()}

        proc = self.train_worker._roi_proc
        proc.update()

        # test default reconfigurable values
        self.assertEqual(RoiCombo.ROI3, proc._norm_combo)
        self.assertEqual(RoiFom.SUM, proc._norm_type)

        # test setting new values
        widget._combo_cb.setCurrentText(avail_combos[RoiCombo.ROI3_ADD_ROI4])
        widget._type_cb.setCurrentText(avail_types[RoiFom.MEDIAN])

        proc.update()

        self.assertEqual(RoiCombo.ROI3_ADD_ROI4, proc._norm_combo)
        self.assertEqual(RoiFom.MEDIAN, proc._norm_type)

    def testRoiProjCtrlWidget(self):
        widget = self.image_tool._corrected_view._roi_proj_ctrl_widget
        avail_norms = {value: key for key, value in widget._available_norms.items()}
        avail_combos = {value: key for key, value in widget._available_combos.items()}

        proc = self.train_worker._roi_proc
        proc.update()

        # test default reconfigurable values
        self.assertEqual(RoiCombo.ROI1, proc._proj_combo)
        self.assertEqual('x', proc._proj_direct)
        self.assertEqual(Normalizer.UNDEFINED, proc._proj_norm)
        self.assertEqual((0, math.inf), proc._proj_fom_integ_range)
        self.assertEqual((0, math.inf), proc._proj_auc_range)

        # test setting new values
        widget._combo_cb.setCurrentText(avail_combos[RoiCombo.ROI1_SUB_ROI2])
        widget._direct_cb.setCurrentText('y')
        widget._norm_cb.setCurrentText(avail_norms[Normalizer.ROI])
        widget._fom_integ_range_le.setText("10, 20")
        widget._auc_range_le.setText("30, 40")
        proc.update()
        self.assertEqual(RoiCombo.ROI1_SUB_ROI2, proc._proj_combo)
        self.assertEqual('y', proc._proj_direct)
        self.assertEqual(Normalizer.ROI, proc._proj_norm)
        self.assertEqual((10, 20), proc._proj_fom_integ_range)
        self.assertEqual((30, 40), proc._proj_auc_range)

    def testGeometryCtrlWidget(self):
        from karabo_data.geometry2 import LPD_1MGeometry

        cw = self.image_tool._views_tab
        view = self.image_tool._geometry_view
        self.assertTrue(cw.isTabEnabled(cw.indexOf(view)))
        widget = view._ctrl_widget

        pulse_worker = self.pulse_worker

        widget._geom_file_le.setText(config["GEOMETRY_FILE"])
        self.assertTrue(widget.updateMetaData())
        pulse_worker._assembler.update()
        self.assertIsInstance(pulse_worker._assembler._geom, LPD_1MGeometry)

        widget._with_geometry_cb.setChecked(False)
        self.assertTrue(widget.updateMetaData())
        pulse_worker._assembler.update()
        self.assertIsNone(pulse_worker._assembler._geom)

    def testViewTabSwitching(self):
        tab = self.image_tool._views_tab
        self.assertEqual(0, tab.currentIndex())

        TabIndex = self.image_tool.TabIndex

        # switch to "gain / offset"
        record_btn = self.image_tool._gain_offset_view._ctrl_widget._record_dark_btn
        tab.tabBarClicked.emit(TabIndex.GAIN_OFFSET)
        tab.setCurrentIndex(TabIndex.GAIN_OFFSET)
        QTest.mouseClick(record_btn, Qt.LeftButton)  # start recording
        self.assertTrue(record_btn.isChecked())

        # switch to "reference"
        tab.tabBarClicked.emit(TabIndex.REFERENCE)
        tab.setCurrentIndex(TabIndex.REFERENCE)
        # test automatically stop dark recording when switching tab
        self.assertFalse(record_btn.isChecked())

        # switch to "azimuthal integration 1D"
        self.assertEqual('0', self._meta.hget(Metadata.ANALYSIS_TYPE, AnalysisType.AZIMUTHAL_INTEG))
        tab.tabBarClicked.emit(TabIndex.AZIMUTHAL_INTEG_1D)
        tab.setCurrentIndex(TabIndex.AZIMUTHAL_INTEG_1D)
        self.assertEqual('1', self._meta.hget(Metadata.ANALYSIS_TYPE, AnalysisType.AZIMUTHAL_INTEG))

        # switch to "geometry"
        tab.tabBarClicked.emit(TabIndex.GEOMETRY)
        tab.setCurrentIndex(TabIndex.GEOMETRY)


class TestImageToolTs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('JungFrau', 'FXE')

        cls.foam = Foam().init()
        cls.gui = cls.foam._gui
        cls.train_worker = cls.foam.train_worker

        actions = cls.gui._tool_bar.actions()
        cls._action = actions[3]
        assert("Image tool" == cls._action.text())

    @classmethod
    def tearDownClass(cls):
        cls.foam.terminate()

        wait_until_redis_shutdown()

        os.remove(config.config_file)

    def setUp(self):
        # construct a fresh ImageToolWindow for each test
        self.gui._image_tool = ImageToolWindow(queue=self.gui._queue,
                                               pulse_resolved=self.gui._pulse_resolved,
                                               parent=self.gui)
        self.image_tool = self.gui._image_tool

    def testGeneral(self):
        self.assertFalse(self.image_tool._pulse_resolved)
        self.assertFalse(self.image_tool._image_ctrl_widget._pulse_resolved)

    def testMovingAverageQLineEdit(self):
        # TODO: remove it in the future
        widget = self.image_tool._image_ctrl_widget
        # moving average is disabled
        self.assertFalse(widget.moving_avg_le.isEnabled())

    def testGeometryCtrlWidget(self):
        cw = self.image_tool._views_tab
        view = self.image_tool._geometry_view
        self.assertFalse(cw.isTabEnabled(cw.indexOf(view)))


if __name__ == '__main__':
    unittest.main()
