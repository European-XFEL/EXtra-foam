import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import math
import os
import tempfile
import time

import numpy as np

from PyQt5.QtTest import QTest, QSignalSpy
from PyQt5.QtCore import Qt, QPoint

from extra_foam.config import (
    AnalysisType, config, ImageTransformType, Normalizer, RoiCombo, RoiFom, RoiProjType
)
from extra_foam.gui import mkQApp
from extra_foam.gui.image_tool import ImageToolWindow
from extra_foam.logger import logger
from extra_foam.pipeline.data_model import ImageData, ProcessedData, RectRoiGeom
from extra_foam.pipeline.exceptions import ImageProcessingError
from extra_foam.pipeline.processors import ImageProcessor, ImageRoiPulse
from extra_foam.pipeline.tests import _TestDataMixin
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


class TestImageTool(unittest.TestCase, _TestDataMixin):
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
                                               require_geometry=self.gui._require_geometry,
                                               parent=self.gui)
        self.image_tool = self.gui._image_tool

        self.view = self.image_tool._corrected_view.imageView
        self.view._image = None

        self.pulse_worker._image_proc = ImageProcessor()
        self.pulse_worker._image_roi = ImageRoiPulse()

    def testGeneral(self):
        self.assertTrue(self.image_tool._pulse_resolved)
        self.assertTrue(self.image_tool._require_geometry)
        self.assertTrue(self.image_tool._image_ctrl_widget._pulse_resolved)
        self.assertTrue(self.image_tool._geometry_view._ctrl_widget._require_geometry)

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
        proc = self.pulse_worker._image_roi
        self.assertEqual(4, len(roi_ctrls))

        # test default

        proc.update()

        for i, ctrl in enumerate(roi_ctrls, 1):
            # test real ROI position and size matches the numbers in the GUI
            self.assertListEqual([int(ctrl._px_le.text()), int(ctrl._py_le.text())],
                                 list(ctrl._roi.pos()))
            self.assertListEqual([int(ctrl._width_le.text()), int(ctrl._height_le.text())],
                                 list(ctrl._roi.size()))
            self.assertListEqual(RectRoiGeom.INVALID, getattr(proc, f"_geom{i}"))

        for ctrl in roi_ctrls:
            self.assertFalse(ctrl._activate_cb.isChecked())
            self.assertFalse(ctrl._lock_cb.isChecked())
            self.assertFalse(ctrl._width_le.isEnabled())
            self.assertFalse(ctrl._height_le.isEnabled())
            self.assertFalse(ctrl._px_le.isEnabled())
            self.assertFalse(ctrl._py_le.isEnabled())

        # test activating ROI

        for i, item in enumerate(zip(roi_ctrls, self.view._rois), 1):
            ctrl, roi = item
            self.assertIs(ctrl._roi, roi)

            QTest.mouseClick(ctrl._activate_cb, Qt.LeftButton,
                             pos=QPoint(2, ctrl._activate_cb.height()/2))
            self.assertTrue(ctrl._activate_cb.isChecked())
            proc.update()
            w_gt, h_gt = int(ctrl._width_le.text()), int(ctrl._height_le.text())
            self.assertTupleEqual((w_gt, h_gt), tuple(roi.size()))
            x_gt, y_gt = int(ctrl._px_le.text()), int(ctrl._py_le.text())
            self.assertTupleEqual((x_gt, y_gt), tuple(roi.pos()))
            self.assertListEqual([x_gt, y_gt, w_gt, h_gt], getattr(proc, f"_geom{i}"))

            # use keyClicks to test that the QLineEdit is enabled
            ctrl._width_le.clear()
            QTest.keyClicks(ctrl._width_le, "10")
            QTest.keyPress(ctrl._width_le, Qt.Key_Enter)
            ctrl._height_le.clear()
            QTest.keyClicks(ctrl._height_le, "30")
            QTest.keyPress(ctrl._height_le, Qt.Key_Enter)
            self.assertTupleEqual((10, 30), tuple(roi.size()))

            # ROI can be outside of the image
            ctrl._px_le.clear()
            QTest.keyClicks(ctrl._px_le, "-1")
            QTest.keyPress(ctrl._px_le, Qt.Key_Enter)
            ctrl._py_le.clear()
            QTest.keyClicks(ctrl._py_le, "-3")
            QTest.keyPress(ctrl._py_le, Qt.Key_Enter)
            self.assertTupleEqual((-1, -3), tuple(roi.pos()))
            proc.update()
            self.assertListEqual([-1, -3, 10, 30], getattr(proc, f"_geom{i}"))

            # lock ROI ctrl
            QTest.mouseClick(ctrl._lock_cb, Qt.LeftButton,
                             pos=QPoint(2, ctrl._lock_cb.height()/2))
            self.assertTrue(ctrl._activate_cb.isChecked())
            self.assertTrue(ctrl._lock_cb.isChecked())
            self.assertFalse(ctrl._width_le.isEnabled())
            self.assertFalse(ctrl._height_le.isEnabled())
            self.assertFalse(ctrl._px_le.isEnabled())
            self.assertFalse(ctrl._py_le.isEnabled())

            # deactivate ROI ctrl
            QTest.mouseClick(ctrl._activate_cb, Qt.LeftButton,
                             pos=QPoint(2, ctrl._activate_cb.height()/2))
            self.assertFalse(ctrl._activate_cb.isChecked())
            self.assertTrue(ctrl._lock_cb.isChecked())
            self.assertFalse(ctrl._width_le.isEnabled())
            self.assertFalse(ctrl._height_le.isEnabled())
            self.assertFalse(ctrl._px_le.isEnabled())
            self.assertFalse(ctrl._py_le.isEnabled())
            proc.update()
            self.assertListEqual(RectRoiGeom.INVALID, getattr(proc, f"_geom{i}"))

    def testMovingAverageQLineEdit(self):
        # TODO: remove it in the future
        widget = self.image_tool._image_ctrl_widget
        # moving average is disabled
        self.assertFalse(widget.moving_avg_le.isEnabled())

    def testImageCtrlWidget(self):
        widget = self.image_tool._image_ctrl_widget

        spy = QSignalSpy(self.image_tool._mediator.reset_image_level_sgn)
        widget.auto_level_btn.clicked.emit()
        self.assertEqual(1, len(spy))

    def testMaskCtrlWidget(self):
        win = self.image_tool
        widget = win._mask_ctrl_widget
        view = win._corrected_view._corrected
        proc = self.pulse_worker._image_proc
        assembler = proc._assembler

        self.assertTrue(config["MASK_TILE_EDGE"])
        self.assertFalse(config["MASK_ASIC_EDGE"])
        # test default
        self.assertTrue(widget.mask_tile_cb.isChecked())
        self.assertFalse(widget.mask_asic_cb.isChecked())
        proc.update()
        self.assertEqual((-1e5, 1e5), proc._threshold_mask)
        self.assertTrue(assembler._mask_tile)
        self.assertFalse(assembler._mask_asic)
        self.assertFalse(view._mask_save_in_modules)

        # test set new value
        widget.threshold_mask_le.setText("1, 10")
        widget.mask_tile_cb.setChecked(False)
        widget.mask_asic_cb.setChecked(True)
        widget.mask_save_in_modules_cb.setChecked(True)
        proc.update()
        self.assertEqual((1, 10), proc._threshold_mask)
        self.assertFalse(assembler._mask_tile)
        self.assertTrue(assembler._mask_asic)
        self.assertTrue(view._mask_save_in_modules)

        # test save/load mask
        with patch.object(win._corrected_view._corrected, "saveImageMask") as patched:
            QTest.mouseClick(widget.save_btn, Qt.LeftButton)
            patched.assert_called_once()
        with patch.object(win._corrected_view._corrected, "loadImageMask") as patched:
            QTest.mouseClick(widget.load_btn, Qt.LeftButton)
            patched.assert_called_once()

        # test loading meta data
        mediator = widget._mediator
        mask_tile_state = widget.mask_tile_cb.isChecked()
        mask_asic_state = widget.mask_asic_cb.isChecked()
        mask_save_in_modules_state = widget.mask_save_in_modules_cb.isChecked()
        mediator.onImageThresholdMaskChange((-100, 10000))
        mediator.onImageMaskTileEdgeChange(not mask_tile_state)
        mediator.onImageMaskAsicEdgeChange(not mask_asic_state)
        mediator.onImageMaskSaveInModulesToggled(not mask_save_in_modules_state)
        widget.loadMetaData()
        self.assertEqual("-100, 10000", widget.threshold_mask_le.text())
        self.assertEqual(not mask_tile_state, widget.mask_tile_cb.isChecked())
        self.assertEqual(mask_asic_state, widget.mask_asic_cb.isChecked())
        self.assertEqual(not mask_save_in_modules_state, widget.mask_save_in_modules_cb.isChecked())

        with patch.dict(config._data, {"MASK_ASIC_EDGE": "True"}):
            widget.loadMetaData()
            self.assertEqual(not mask_asic_state, widget.mask_asic_cb.isChecked())

    @patch("extra_foam.pipeline.processors.ImageProcessor._require_geom",
           new_callable=PropertyMock, create=True, return_value=False)
    @patch("extra_foam.pipeline.processors.image_assembler.ImageAssemblerFactory.BaseAssembler.process",
           side_effect=lambda x: x)
    def testDrawMask(self, patched_process, require_geometry):
        # TODO: test by really drawing something on ImageTool
        from extra_foam.ipc import ImageMaskPub

        pub = ImageMaskPub()
        widget = self.image_tool._mask_ctrl_widget
        proc = self.pulse_worker._image_proc

        data, _ = self.data_with_assembled(1001, (4, 10, 10))

        # trigger the lazily evaluated subscriber
        proc.process(data)

        mask_gt = np.zeros(data['assembled']['data'].shape[-2:], dtype=np.bool)

        # test default
        np.testing.assert_array_equal(proc._image_mask, mask_gt)

        # test changing mask
        pub.draw((0, 0, 2, 3))
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
        np.testing.assert_array_equal(mask_gt, proc._image_mask)

        # add one more mask region
        pub.draw((1, 1, 2, 3))
        proc.process(data)
        mask_gt[1:4, 1:3] = True
        np.testing.assert_array_equal(mask_gt, proc._image_mask)

        # test erasing mask
        pub.erase((2, 2, 3, 3))
        proc.process(data)
        mask_gt[2:5, 2:5] = False
        np.testing.assert_array_equal(mask_gt, proc._image_mask)

        # test removing mask
        QTest.mouseClick(widget.remove_btn, Qt.LeftButton)
        proc.process(data)
        np.testing.assert_array_equal(np.zeros_like(mask_gt, dtype=np.bool), proc._image_mask)

        # test set mask
        pub.set(mask_gt)
        proc.process(data)
        np.testing.assert_array_equal(mask_gt, proc._image_mask)

        # test set a mask which has a different shape from the image
        mask_gt = np.ones((2, 2), dtype=np.bool)
        pub.set(mask_gt)
        with self.assertRaises(ImageProcessingError):
            proc.process(data)
        # an empty image mask with a different shape will be automatically reset
        mask_gt = np.zeros((2, 2), dtype=np.bool)
        pub.set(mask_gt)
        proc.process(data)
        np.testing.assert_array_equal(np.zeros((10, 10), dtype=np.bool), proc._image_mask)

    def testReferenceCtrlWidget(self):
        view = self.image_tool._reference_view
        widget = view._ctrl_widget
        corrected = view._corrected
        proc = self.pulse_worker._image_proc

        data, _ = self.data_with_assembled(1001, (4, 10, 10))

        # test setting reference (no image)
        QTest.mouseClick(widget.set_current_btn, Qt.LeftButton)
        updated, ref = proc._ref_sub.update()
        self.assertFalse(updated)
        self.assertIsNone(ref)

        # test setting reference
        corrected._image = 2 * np.ones((10, 10), np.float32)
        QTest.mouseClick(widget.set_current_btn, Qt.LeftButton)
        updated, ref = proc._ref_sub.update()
        self.assertTrue(updated)
        np.testing.assert_array_equal(corrected.image, ref)

        # test setting reference multiple times
        for i in range(5):
            corrected._image = np.random.rand(10, 10).astype(np.float32)
            QTest.mouseClick(widget.set_current_btn, Qt.LeftButton)
        updated, ref = proc._ref_sub.update()
        self.assertTrue(updated)
        np.testing.assert_array_equal(corrected.image, ref)

        # test removing reference
        QTest.mouseClick(widget.remove_btn, Qt.LeftButton)
        updated, ref = proc._ref_sub.update()
        self.assertTrue(updated)
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

        with patch('extra_foam.ipc.read_image', side_effect=_read_image_side_effect):
            with patch('extra_foam.gui.image_tool.reference_view.QFileDialog.getOpenFileName',
                       return_value=["reference/file/path"]):
                QTest.mouseClick(widget.load_btn, Qt.LeftButton)
                self.assertEqual("reference/file/path", widget.filepath_le.text())
                updated, ref = proc._ref_sub.update()
                self.assertTrue(updated)
                np.testing.assert_array_equal(ref, ref_gt)

                QTest.mouseClick(widget.remove_btn, Qt.LeftButton)
                self.assertEqual("", widget.filepath_le.text())
                updated, ref = proc._ref_sub.update()
                self.assertTrue(updated)
                self.assertIsNone(ref)

    def testBulletinView(self):
        processed = ProcessedData(1357)

        processed.image = ImageData.from_array(np.ones((10, 4, 4), np.float32))
        processed.image.dark_count = 99
        processed.image.n_dark_pulses = 10
        processed.pidx.mask_by_index([1, 3, 5, 6])
        self.gui._queue.append(processed)
        self.image_tool.updateWidgetsF()

        view = self.image_tool._bulletin_view
        self.assertEqual(1357, int(view._displayed_tid.intValue()))
        self.assertEqual(10, int(view._n_total_pulses.intValue()))
        self.assertEqual(6, int(view._n_kept_pulses.intValue()))
        self.assertEqual(99, int(view._dark_train_counter.intValue()))
        self.assertEqual(10, int(view._n_dark_pulses.intValue()))

        with patch.object(view._mon, "reset_process_count") as reset:
            view._reset_process_count_btn.clicked.emit()
            reset.assert_called_once()

    def testCalibrationCtrlWidget(self):
        widget = self.image_tool._calibration_view._ctrl_widget

        proc = self.pulse_worker._image_proc

        proc.update()
        self.assertFalse(proc._correct_gain)
        self.assertFalse(proc._correct_offset)
        self.assertEqual(slice(None), proc._gain_cells)
        self.assertEqual(slice(None), proc._offset_cells)
        self.assertTrue(proc._gain_cells_updated)
        self.assertTrue(proc._offset_cells_updated)
        self.assertTrue(proc._dark_as_offset)
        self.assertFalse(proc._recording_dark)

        widget._correct_gain_cb.setChecked(True)
        widget._correct_offset_cb.setChecked(True)
        widget._gain_cells_le.setText(":70")
        widget._offset_cells_le.setText("2:120:4")
        widget._dark_as_offset_cb.setChecked(False)
        QTest.mouseClick(widget.record_dark_btn, Qt.LeftButton)
        proc.update()
        self.assertTrue(proc._correct_gain)
        self.assertTrue(proc._correct_offset)
        self.assertEqual(slice(None, 70), proc._gain_cells)
        self.assertEqual(slice(2, 120, 4), proc._offset_cells)
        self.assertTrue(proc._gain_cells_updated)
        self.assertTrue(proc._offset_cells_updated)
        self.assertFalse(proc._dark_as_offset)
        self.assertTrue(proc._recording_dark)

        # test stop dark recording
        QTest.mouseClick(widget.record_dark_btn, Qt.LeftButton)
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

        gain_gt = np.random.randn(2, 2)
        offset_gt = np.random.randn(2, 2)

        def _read_constants_side_effect(fn, **kwargs):
            if fn == "gain/file/path":
                return gain_gt
            if fn == "offset/file/path":
                return offset_gt

        # caveat: first establish the connection
        proc._cal_sub.update()
        with patch('extra_foam.ipc.read_numpy_array', side_effect=_read_constants_side_effect):
            with patch('extra_foam.gui.image_tool.calibration_view.QFileDialog.getOpenFileName',
                       return_value=["gain/file/path"]):
                QTest.mouseClick(widget.load_gain_btn, Qt.LeftButton)
                time.sleep(0.1)  # wait to write into redis
                self.assertEqual("gain/file/path", widget.gain_fp_le.text())

                n_attempts = 0
                # repeat to prevent random failure at Travis
                while n_attempts < 10:
                    n_attempts += 1
                    gain_updated, gain, offset_updated, offset = proc._cal_sub.update()
                    if gain_updated:
                        break
                    time.sleep(0.001)

                np.testing.assert_array_equal(gain, gain_gt)
                self.assertFalse(offset_updated)
                self.assertIsNone(offset)

                QTest.mouseClick(widget.remove_gain_btn, Qt.LeftButton)
                self.assertEqual("", widget.gain_fp_le.text())
                gain_updated, gain, offset_updated, offset = proc._cal_sub.update()
                self.assertTrue(gain_updated)
                self.assertIsNone(gain)

            with patch('extra_foam.gui.image_tool.calibration_view.QFileDialog.getOpenFileName',
                       return_value=["offset/file/path"]):
                QTest.mouseClick(widget.load_offset_btn, Qt.LeftButton)
                time.sleep(0.1)  # wait to write data into redis
                self.assertEqual("offset/file/path", widget.offset_fp_le.text())
                gain_updated, gain, offset_updated, offset = proc._cal_sub.update()
                self.assertFalse(gain_updated)
                self.assertIsNone(gain)
                self.assertTrue(offset_updated)
                np.testing.assert_array_equal(offset, offset_gt)

                QTest.mouseClick(widget.remove_offset_btn, Qt.LeftButton)
                self.assertEqual("", widget.offset_fp_le.text())
                gain_updated, gain, offset_updated, offset = proc._cal_sub.update()
                self.assertTrue(offset_updated)
                self.assertIsNone(offset)

        # test loading meta data
        mediator = widget._mediator
        mediator.onCalDarkAsOffset(True)
        mediator.onCalGainCorrection(False)
        mediator.onCalOffsetCorrection(False)
        mediator.onCalGainMemoCellsChange([0, None, 2])
        mediator.onCalOffsetMemoCellsChange([0, None, 4])
        widget.loadMetaData()
        self.assertEqual(True, widget._dark_as_offset_cb.isChecked())
        self.assertEqual(False, widget._correct_gain_cb.isChecked())
        self.assertEqual(False, widget._correct_offset_cb.isChecked())
        self.assertEqual("0::2", widget._gain_cells_le.text())
        self.assertEqual("0::4", widget._offset_cells_le.text())

    def testAzimuthalInteg1dCtrlWidget(self):
        from extra_foam.pipeline.processors.azimuthal_integration import energy2wavelength
        from extra_foam.gui.ctrl_widgets.azimuthal_integ_ctrl_widget import \
            _DEFAULT_AZIMUTHAL_INTEG_POINTS

        widget = self.image_tool._azimuthal_integ_1d_view._ctrl_widget
        avail_norms = {value: key for key, value in widget._available_norms.items()}
        train_worker = self.train_worker
        proc = train_worker._ai_proc

        proc.update()

        # test default
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

        # test setting new values
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

        # test loading meta data
        mediator = widget._mediator
        mediator.onPhotonEnergyChange("2.0")
        mediator.onSampleDistanceChange("0.2")
        mediator.onAiIntegMethodChange("BBox")
        mediator.onAiNormChange(Normalizer.XGM)
        mediator.onAiIntegPointsChange(512)
        mediator.onAiIntegRangeChange((1, 2))
        mediator.onAiAucRangeChange((2, 3))
        mediator.onAiFomIntegRangeChange((3, 4))
        mediator.onAiPixelSizeXChange(0.001)
        mediator.onAiPixelSizeYChange(0.002)
        mediator.onAiIntegCenterXChange(1)
        mediator.onAiIntegCenterYChange(2)
        widget.loadMetaData()
        self.assertEqual("2.0", widget._photon_energy_le.text())
        self.assertEqual("0.2", widget._sample_dist_le.text())
        self.assertEqual("BBox", widget._integ_method_cb.currentText())
        self.assertEqual("XGM", widget._norm_cb.currentText())
        self.assertEqual("512", widget._integ_pts_le.text())
        self.assertEqual("1, 2", widget._integ_range_le.text())
        self.assertEqual("2, 3", widget._auc_range_le.text())
        self.assertEqual("3, 4", widget._fom_integ_range_le.text())
        self.assertEqual("0.001", widget._px_le.text())
        self.assertEqual("0.002", widget._py_le.text())
        self.assertEqual("1", widget._cx_le.text())
        self.assertEqual("2", widget._cy_le.text())

    def testRoiFomCtrlWidget(self):
        widget = self.image_tool._corrected_view._roi_fom_ctrl_widget
        avail_norms = {value: key for key, value in widget._available_norms.items()}
        avail_combos = {value: key for key, value in widget._available_combos.items()}
        avail_types = {value: key for key, value in widget._available_types.items()}

        proc = self.train_worker._image_roi
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

        # test activate/deactivate master-slave mode
        self.assertFalse(widget._master_slave_cb.isChecked())
        widget._master_slave_cb.setChecked(True)
        proc.update()
        self.assertTrue(proc._roi_fom_master_slave)
        self.assertFalse(widget._combo_cb.isEnabled())
        self.assertEqual("ROI1", widget._combo_cb.currentText())
        widget._master_slave_cb.setChecked(False)
        self.assertTrue(widget._combo_cb.isEnabled())

        # test loading meta data
        mediator = widget._mediator
        mediator.onRoiFomComboChange(RoiCombo.ROI2)
        mediator.onRoiFomTypeChange(RoiFom.MEAN)
        mediator.onRoiFomNormChange(Normalizer.XGM)
        mediator.onRoiFomMasterSlaveModeChange(False)
        widget.loadMetaData()
        self.assertEqual("ROI2", widget._combo_cb.currentText())
        self.assertEqual("MEAN", widget._type_cb.currentText())
        self.assertEqual("XGM", widget._norm_cb.currentText())
        self.assertFalse(widget._master_slave_cb.isChecked())

    def testRoiHistCtrl(self):
        widget = self.image_tool._corrected_view._roi_hist_ctrl_widget
        avail_combos = {value: key for key, value in widget._available_combos.items()}

        proc = self.pulse_worker._image_roi
        proc.update()

        # test default reconfigurable values
        self.assertEqual(RoiCombo.UNDEFINED, proc._hist_combo)
        self.assertEqual(10, proc._hist_n_bins)
        self.assertTupleEqual((-math.inf, math.inf), proc._hist_bin_range)

        # test setting new values
        widget._combo_cb.setCurrentText(avail_combos[RoiCombo.ROI1_SUB_ROI2])
        widget._n_bins_le.setText("100")
        widget._bin_range_le.setText("-1.0, 10.0")
        proc.update()
        self.assertEqual(RoiCombo.ROI1_SUB_ROI2, proc._hist_combo)
        self.assertEqual(100, proc._hist_n_bins)
        self.assertEqual((-1.0, 10.0), proc._hist_bin_range)

        mediator = widget._mediator
        mediator.onRoiHistComboChange(RoiCombo.ROI2)
        mediator.onRoiHistNumBinsChange(10)
        mediator.onRoiHistBinRangeChange((-3, 3))
        widget.loadMetaData()
        self.assertEqual("ROI2", widget._combo_cb.currentText())
        self.assertEqual("10", widget._n_bins_le.text())
        self.assertEqual("-3, 3", widget._bin_range_le.text())

    def testRoiNormCtrlWidget(self):
        widget = self.image_tool._corrected_view._roi_norm_ctrl_widget
        avail_combos = {value: key for key, value in widget._available_combos.items()}
        avail_types = {value: key for key, value in widget._available_types.items()}

        proc = self.train_worker._image_roi
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

        # test loading meta data
        mediator = widget._mediator
        mediator.onRoiNormComboChange(RoiCombo.ROI3_SUB_ROI4)
        mediator.onRoiNormTypeChange(RoiProjType.SUM)
        widget.loadMetaData()
        self.assertEqual("ROI3 - ROI4", widget._combo_cb.currentText())
        self.assertEqual("SUM", widget._type_cb.currentText())

    def testRoiProjCtrlWidget(self):
        widget = self.image_tool._corrected_view._roi_proj_ctrl_widget
        avail_norms_inv = widget._available_norms_inv
        avail_combos_inv = widget._available_combos_inv
        avail_types_inv = widget._available_types_inv

        proc = self.train_worker._image_roi
        proc.update()

        # test default reconfigurable values
        self.assertEqual(RoiCombo.ROI1, proc._proj_combo)
        self.assertEqual(RoiProjType.SUM, proc._proj_type)
        self.assertEqual('x', proc._proj_direct)
        self.assertEqual(Normalizer.UNDEFINED, proc._proj_norm)
        self.assertEqual((0, math.inf), proc._proj_fom_integ_range)
        self.assertEqual((0, math.inf), proc._proj_auc_range)

        # test setting new values
        widget._combo_cb.setCurrentText(avail_combos_inv[RoiCombo.ROI1_SUB_ROI2])
        widget._type_cb.setCurrentText(avail_types_inv[RoiProjType.MEAN])
        widget._direct_cb.setCurrentText('y')
        widget._norm_cb.setCurrentText(avail_norms_inv[Normalizer.ROI])
        widget._fom_integ_range_le.setText("10, 20")
        widget._auc_range_le.setText("30, 40")
        proc.update()
        self.assertEqual(RoiCombo.ROI1_SUB_ROI2, proc._proj_combo)
        self.assertEqual(RoiProjType.MEAN, proc._proj_type)
        self.assertEqual('y', proc._proj_direct)
        self.assertEqual(Normalizer.ROI, proc._proj_norm)
        self.assertEqual((10, 20), proc._proj_fom_integ_range)
        self.assertEqual((30, 40), proc._proj_auc_range)

        # test loading meta data
        mediator = widget._mediator
        mediator.onRoiProjComboChange(RoiCombo.ROI1_ADD_ROI2)
        mediator.onRoiProjTypeChange(RoiProjType.SUM)
        mediator.onRoiProjDirectChange('x')
        mediator.onRoiProjNormChange(Normalizer.XGM)
        mediator.onRoiProjAucRangeChange((1, 2))
        mediator.onRoiProjFomIntegRangeChange((-5, 5))
        widget.loadMetaData()
        self.assertEqual("ROI1 + ROI2", widget._combo_cb.currentText())
        self.assertEqual("SUM", widget._type_cb.currentText())
        self.assertEqual("x", widget._direct_cb.currentText())
        self.assertEqual("XGM", widget._norm_cb.currentText())
        self.assertEqual("1, 2", widget._auc_range_le.text())
        self.assertEqual("-5, 5", widget._fom_integ_range_le.text())

    def testGeometryCtrlWidget(self):
        from extra_foam.config import GeomAssembler
        from extra_foam.gui.ctrl_widgets.geometry_ctrl_widget import _parse_table_widget

        cw = self.image_tool._views_tab
        view = self.image_tool._geometry_view
        self.assertTrue(cw.isTabEnabled(cw.indexOf(view)))
        widget = view._ctrl_widget
        mask_ctrl_widget = self.image_tool._mask_ctrl_widget

        image_proc = self.pulse_worker._image_proc
        assembler = image_proc._assembler

        # test default
        image_proc.update()
        self.assertFalse(assembler._stack_only)
        self.assertEqual(GeomAssembler.OWN, assembler._assembler_type)
        self.assertListEqual([list(v) for v in config["QUAD_POSITIONS"]], assembler._coordinates)

        # prepare for the following test
        widget._stack_only_cb.setChecked(True)
        mask_ctrl_widget.mask_tile_cb.setChecked(True)
        mask_ctrl_widget.mask_asic_cb.setChecked(True)
        mask_ctrl_widget.mask_save_in_modules_cb.setChecked(True)
        image_proc.update()
        self.assertTrue(assembler._stack_only)
        self.assertTrue(assembler._mask_tile)
        self.assertTrue(assembler._mask_asic)

        # test setting new values
        assemblers_inv = widget._assemblers_inv
        widget._assembler_cb.setCurrentText(assemblers_inv[GeomAssembler.EXTRA_GEOM])
        self.assertFalse(widget._stack_only_cb.isEnabled())
        self.assertFalse(widget._stack_only_cb.isChecked())
        self.assertFalse(mask_ctrl_widget.mask_tile_cb.isEnabled())
        self.assertFalse(mask_ctrl_widget.mask_tile_cb.isChecked())
        self.assertFalse(mask_ctrl_widget.mask_asic_cb.isEnabled())
        self.assertFalse(mask_ctrl_widget.mask_asic_cb.isChecked())
        self.assertFalse(mask_ctrl_widget.mask_save_in_modules_cb.isEnabled())
        self.assertFalse(mask_ctrl_widget.mask_save_in_modules_cb.isChecked())
        widget._geom_file_le.setText("/geometry/file/")
        for i in range(4):
            for j in range(2):
                widget._coordinates_tb.cellWidget(j, i).setText("0.0")

        with patch.object(assembler, "_load_geometry") as mocked_load_geometry:
            image_proc.update()
            mocked_load_geometry.assert_called_once_with(
                False, "/geometry/file/", [[0., 0.] for i in range(4)], GeomAssembler.EXTRA_GEOM)
        self.assertFalse(assembler._mask_tile)

        widget._assembler_cb.setCurrentText(assemblers_inv[GeomAssembler.OWN])
        self.assertTrue(widget._stack_only_cb.isEnabled())
        self.assertTrue(mask_ctrl_widget.mask_tile_cb.isEnabled())
        self.assertTrue(mask_ctrl_widget.mask_save_in_modules_cb.isEnabled())

        # test loading meta data
        mediator = widget._mediator
        mediator.onGeomAssemblerChange(GeomAssembler.EXTRA_GEOM)
        mediator.onGeomFileChange('geometry/new_file')
        mediator.onGeomStackOnlyChange(False)
        quad_positions = [[1., 2.], [3., 4.], [5., 6.], [7., 8.]]
        mediator.onGeomCoordinatesChange(quad_positions)
        widget.loadMetaData()
        self.assertEqual("EXtra-geom", widget._assembler_cb.currentText())
        self.assertEqual('geometry/new_file', widget._geom_file_le.text())
        self.assertFalse(widget._stack_only_cb.isChecked())
        self.assertListEqual(quad_positions, _parse_table_widget((widget._coordinates_tb)))

    def testImageTransformCtrlWidget(self):
        tab = self.image_tool._views_tab
        TabIndex = self.image_tool.TabIndex
        tab.tabBarClicked.emit(TabIndex.IMAGE_TRANSFORM)
        tab.setCurrentIndex(TabIndex.IMAGE_TRANSFORM)

        ctrl_widget = self.image_tool._transform_view._ctrl_widget
        fft_widget = ctrl_widget._fourier_transform
        ed_widget = ctrl_widget._edge_detection

        proc = self.pulse_worker._image_transform_proc
        fft = proc._fft
        ed = proc._ed

        # test default
        # also test only parameters of the activated transform type are updated
        proc.update()
        self.assertEqual(1, proc._ma_window)
        self.assertEqual(ImageTransformType.FOURIER_TRANSFORM, proc._transform_type)
        self.assertIsNone(ed.kernel_size)
        # fourier transform
        self.assertTrue(fft.logrithmic)
        # edge detection
        ctrl_widget._opt_tab.setCurrentIndex(1)
        proc.update()
        self.assertEqual(ImageTransformType.EDGE_DETECTION, proc._transform_type)
        self.assertEqual(5, ed.kernel_size)
        self.assertEqual(1.1, ed.sigma)
        self.assertEqual((50, 100), ed.threshold)

        # test setting new values
        ctrl_widget._ma_window_le.setText("10")
        fft_widget.logrithmic_cb.setChecked(False)
        ed_widget.kernel_size_sp.setValue(3)
        ed_widget.sigma_sp.setValue(0.5)
        ed_widget.threshold_le.setText("-1, 1")
        proc.update()
        self.assertEqual(10, proc._ma_window)
        self.assertTrue(fft.logrithmic)
        self.assertEqual(3, ed.kernel_size)
        self.assertEqual(0.5, ed.sigma)
        self.assertEqual((-1, 1), ed.threshold)
        ctrl_widget._opt_tab.setCurrentIndex(0)
        proc.update()
        self.assertFalse(fft.logrithmic)

        # switch back to "overview"
        tab.tabBarClicked.emit(TabIndex.OVERVIEW)
        proc.update()
        # test unregistration
        self.assertEqual(ImageTransformType.UNDEFINED, proc._transform_type)
        tab.setCurrentIndex(TabIndex.OVERVIEW)

        # test loading meta data
        mediator = ctrl_widget._mediator
        mediator.onItTransformTypeChange(ImageTransformType.FOURIER_TRANSFORM)
        mediator.onItMaWindowChange("100")
        mediator.onItFftLogrithmicScaleChange(True)
        mediator.onItEdKernelSizeChange("3")
        mediator.onItEdSigmaChange("2.1")
        mediator.onItEdThresholdChange((20, 40))
        ctrl_widget.loadMetaData()
        self.assertEqual(ImageTransformType.UNDEFINED, proc._transform_type)  # unchanged
        self.assertEqual("100", ctrl_widget._ma_window_le.text())
        self.assertTrue(fft_widget.logrithmic_cb.isChecked())
        self.assertEqual("3", ed_widget.kernel_size_sp.text())
        self.assertEqual("2.10", ed_widget.sigma_sp.text())
        self.assertEqual("20, 40", ed_widget.threshold_le.text())

        with patch.object(mediator, "onItTransformTypeChange") as mocked:
            ctrl_widget.updateMetaData()
            mocked.assert_not_called()

    def testViewTabSwitching(self):
        tab = self.image_tool._views_tab
        self.assertEqual(0, tab.currentIndex())
        mask_ctrl_widget = self.image_tool._mask_ctrl_widget

        TabIndex = self.image_tool.TabIndex

        # switch to "gain / offset"
        record_btn = self.image_tool._calibration_view._ctrl_widget.record_dark_btn
        tab.tabBarClicked.emit(TabIndex.GAIN_OFFSET)
        tab.setCurrentIndex(TabIndex.GAIN_OFFSET)
        QTest.mouseClick(record_btn, Qt.LeftButton)  # start recording
        self.assertTrue(record_btn.isChecked())
        self.assertFalse(mask_ctrl_widget.draw_mask_btn.isEnabled())
        self.assertFalse(mask_ctrl_widget.erase_mask_btn.isEnabled())

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

        # switch to "image transform"
        tab.tabBarClicked.emit(TabIndex.IMAGE_TRANSFORM)
        tab.setCurrentIndex(TabIndex.IMAGE_TRANSFORM)

        # switch back to "overview"
        tab.tabBarClicked.emit(TabIndex.OVERVIEW)
        tab.setCurrentIndex(TabIndex.OVERVIEW)
        self.assertTrue(mask_ctrl_widget.draw_mask_btn.isEnabled())
        self.assertTrue(mask_ctrl_widget.erase_mask_btn.isEnabled())


class TestImageToolTs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('ePix100', 'MID')

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
                                               require_geometry=self.gui._require_geometry,
                                               parent=self.gui)
        self.image_tool = self.gui._image_tool

    def testGeneral(self):
        self.assertFalse(self.image_tool._pulse_resolved)
        self.assertFalse(self.image_tool._require_geometry)
        self.assertFalse(self.image_tool._image_ctrl_widget._pulse_resolved)
        self.assertFalse(self.image_tool._geometry_view._ctrl_widget._require_geometry)

    def testMovingAverageQLineEdit(self):
        # TODO: remove it in the future
        widget = self.image_tool._image_ctrl_widget
        # moving average is disabled
        self.assertFalse(widget.moving_avg_le.isEnabled())

    def testGeometryCtrlWidget(self):
        cw = self.image_tool._views_tab
        view = self.image_tool._geometry_view
        self.assertFalse(cw.isTabEnabled(cw.indexOf(view)))

    def testMaskCtrlWidget(self):
        widget = self.image_tool._mask_ctrl_widget

        self.assertFalse(config["MASK_TILE_EDGE"])
        mask_asic_state = config["MASK_ASIC_EDGE"]
        # test default
        self.assertFalse(widget.mask_tile_cb.isChecked())
        self.assertEqual(mask_asic_state, widget.mask_asic_cb.isChecked())
        self.assertFalse(widget.mask_save_in_modules_cb.isChecked())

        # test loading meta data
        # test if the meta data is invalid
        mediator = widget._mediator
        mediator.onImageMaskTileEdgeChange(True)
        mediator.onImageMaskAsicEdgeChange(not mask_asic_state)
        mediator.onImageMaskSaveInModulesToggled(True)
        widget.loadMetaData()
        self.assertFalse(widget.mask_tile_cb.isChecked())
        self.assertFalse(widget.mask_asic_cb.isChecked())
        self.assertFalse(widget.mask_save_in_modules_cb.isChecked())

    def testCalibrationCtrlWidget(self):
        widget = self.image_tool._calibration_view._ctrl_widget
        self.assertFalse(widget._gain_cells_le.isEnabled())
        self.assertFalse(widget._offset_cells_le.isEnabled())

        # test loading meta data
        # test if the meta data is invalid
        mediator = widget._mediator
        mediator.onCalGainMemoCellsChange([0, None, 2])
        mediator.onCalOffsetMemoCellsChange([0, None, 2])
        widget.loadMetaData()
        self.assertEqual(":", widget._gain_cells_le.text())
        self.assertEqual(":", widget._offset_cells_le.text())


if __name__ == '__main__':
    unittest.main()
