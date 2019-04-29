import unittest

import numpy as np

from karaboFAI.config import config, PumpProbeMode, PumpProbeType
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.pipeline.data_model import (
    PumpProbeData, ProcessedData
)
from karaboFAI.pipeline.processors import PumpProbeProcessor


class TestPumpProbeProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config["PIXEL_SIZE"] = 1e-6
        config["MASK_RANGE"] = (None, None)

    def setUp(self):
        self._proc = PumpProbeProcessor()
        self._proc.analysis_type = PumpProbeType.AZIMUTHAL_INTEGRATION
        PumpProbeData.clear()

        self._proc.fom_itgt_range = (1, 5)

        self._data = []
        intensity = np.array([[0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1],
                              [0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1]])
        # a train with 4 pulses
        imgs = np.arange(16, dtype=np.float).reshape(4, 2, 2)
        for i in range(10):
            self._data.append(ProcessedData(i, imgs))
            self._data[i].ai.momentum = np.linspace(1, 5, 5)
            self._data[i].ai.intensities = (i+1)*intensity

    def testExceptions(self):
        # test raises when pulse IDs are out of range
        self._proc.mode = PumpProbeMode.SAME_TRAIN
        data = self._data[0]

        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3, 5]
        with self.assertRaisesRegex(ProcessingError, "Out of range: off"):
            self._proc.run_once(data)

        self._proc.on_pulse_ids = [0, 2, 4]
        self._proc.off_pulse_ids = [1, 3]
        with self.assertRaisesRegex(ProcessingError, "Out of range: on"):
            self._proc.run_once(data)

    def testPreDefinedOff(self):
        pass

    def testSameTrain(self):
        self._proc.mode = PumpProbeMode.SAME_TRAIN
        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3]

        fom_hist_gt = []
        train_ids_gt = []

        # 1st train
        data = self._data[0]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 1, 0, 1, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([1, 0, 1, 0, 1])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(5)
        train_ids_gt.append(0)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 2, 0, 2, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([2, 0, 2, 0, 2])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(10)
        train_ids_gt.append(1)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 3, 0, 3, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([3, 0, 3, 0, 3])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(15)
        train_ids_gt.append(2)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

    def testSameTrainMovingAverage(self):
        self._run_same_train_moving_average_test()
        ProcessedData.clear_pp_hist()
        self._proc.reset()  # test reset
        self._run_same_train_moving_average_test()

    def _run_same_train_moving_average_test(self):
        self._proc.mode = PumpProbeMode.SAME_TRAIN
        self._proc.ma_window = 3
        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3]

        fom_hist_gt = []
        train_ids_gt = []

        # 1st train
        data = self._data[0]
        self._proc.run_once(data)

        self.assertEqual(1, self._proc._ma_count)
        on_data_gt = np.array([0, 1, 0, 1, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([1, 0, 1, 0, 1])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(5)
        train_ids_gt.append(0)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.run_once(data)

        self.assertEqual(2, self._proc._ma_count)
        on_data_gt = np.array([0, 1.5, 0, 1.5, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([1.5, 0, 1.5, 0, 1.5])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(7.5)
        train_ids_gt.append(1)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.run_once(data)

        self.assertEqual(3, self._proc._ma_count)
        on_data_gt = np.array([0, 2, 0, 2, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([2, 0, 2, 0, 2])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(10)
        train_ids_gt.append(2)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 4th train
        data = self._data[3]
        self._proc.run_once(data)

        # Since the moving average is an approximation, it gives 2.666667
        # instead of (2 + 3 + 4) / 3 = 3.
        self.assertEqual(3, self._proc._ma_count)
        on_data_gt = np.array([0, 2.666667, 0, 2.666667, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([2.666667, 0, 2.666667, 0, 2.666667])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(13.333333)  # = 2.666667 * 5
        train_ids_gt.append(3)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_almost_equal(train_ids_gt, tids)
        np.testing.assert_array_almost_equal(fom_hist_gt, foms)

    def testEvenTrainOn(self):
        """On-pulse has even id."""
        self._proc.mode = PumpProbeMode.EVEN_TRAIN_ON
        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3]

        fom_hist_gt = []
        train_ids_gt = []

        # 1st train
        data = self._data[0]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 1, 0, 1, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.run_once(data)

        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([2, 0, 2, 0, 2])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(8)
        train_ids_gt.append(1)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 3, 0, 3, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 4th train
        data = self._data[3]
        self._proc.run_once(data)

        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([4, 0, 4, 0, 4])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(18)
        train_ids_gt.append(3)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 5th train was lost

        # 6th train (off pulse is followed by an off pulse)
        data = self._data[5]
        self._proc.run_once(data)

        self.assertTrue(data.pp.on_data is None)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 7th train
        data = self._data[6]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 7, 0, 7, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 7th train was sent twice (on train is followed by an on train)
        data = self._data[6]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 7, 0, 7, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 8th train was lost

        # 9th train (on train is followed by an on train)
        data = self._data[8]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 9, 0, 9, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 10th train
        data = self._data[9]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 9, 0, 9, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([10, 0, 10, 0, 10])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)

        fom_hist_gt.append(48)
        train_ids_gt.append(9)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

    def testOddTrainOn(self):
        """On-pulse has odd id."""
        self._proc.mode = PumpProbeMode.ODD_TRAIN_ON
        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3]

        fom_hist_gt = []
        train_ids_gt = []

        # 1st train
        data = self._data[0]
        self._proc.run_once(data)

        self.assertTrue(data.pp.on_data is None)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 2, 0, 2, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.run_once(data)

        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([3, 0, 3, 0, 3])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(13)
        train_ids_gt.append(2)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 4th train
        data = self._data[3]
        self._proc.run_once(data)

        on_data_gt = np.array([0, 4, 0, 4, 0])
        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        self.assertTrue(data.pp.off_data is None)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 5th train
        data = self._data[4]
        self._proc.run_once(data)

        np.testing.assert_array_almost_equal(on_data_gt, data.pp.on_data)
        off_data_gt = np.array([5, 0, 5, 0, 5])
        np.testing.assert_array_almost_equal(off_data_gt, data.pp.off_data)
        fom_hist_gt.append(23)
        train_ids_gt.append(4)
        tids, foms, _ = data.pp.fom
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)
