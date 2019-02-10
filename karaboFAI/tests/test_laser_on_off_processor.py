import unittest

import numpy as np

from karaboFAI.data_processing import ProcessedData, OpLaserMode
from karaboFAI.data_processing.data_model import LaserOnOffData
from karaboFAI.data_processing.data_processor import LaserOnOffProcessor


class TestLaserOnOffWindow(unittest.TestCase):
    def setUp(self):
        self._proc = LaserOnOffProcessor()
        LaserOnOffData.clear()

        self._proc.normalization_range = (1, 5)
        self._proc.integration_range = (1, 5)
        self._proc.moving_average_window = 9999
        
        self._data = []
        intensity = np.array([[0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1],
                              [0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1]])
        for i in range(10):
            self._data.append(ProcessedData(i,
                                            momentum=np.linspace(1, 5, 5),
                                            intensities=(i+1)*intensity))

    def testNormalMode(self):
        self._proc.laser_mode = OpLaserMode.NORMAL
        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3]

        fom_hist_gt = []
        train_ids_gt = []

        # 1st train
        data = self._data[0]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(data.on_off.off_pulse, off_pulse_gt)
        on_ma_gt = [0, 1, 0, 1, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [1, 0, 1, 0, 1]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(2.5)
        train_ids_gt.append(0)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(data.on_off.off_pulse, off_pulse_gt)
        on_ma_gt = [0, 1.5, 0, 1.5, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [1.5, 0, 1.5, 0, 1.5]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(2.5)
        train_ids_gt.append(1)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(data.on_off.off_pulse, off_pulse_gt)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [2, 0, 2, 0, 2]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(2.5)
        train_ids_gt.append(2)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

    def testEvenOddMode(self):
        """On-pulse has even id."""
        self._proc.laser_mode = OpLaserMode.EVEN_ON
        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3]

        fom_hist_gt = []
        train_ids_gt = []

        # 1st train
        data = self._data[0]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        self.assertTrue(data.on_off.off_pulse is None)
        on_ma_gt = [0, 1, 0, 1, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        self.assertTrue(self._proc._off_pulses_ma is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(0)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(data.on_off.off_pulse, off_pulse_gt)
        on_ma_gt = [0, 1, 0, 1, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [2, 0, 2, 0, 2]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(2.5)
        train_ids_gt.append(1)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        self.assertTrue(data.on_off.off_pulse is None)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [2, 0, 2, 0, 2]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(None)
        train_ids_gt.append(2)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 4th train
        data = self._data[3]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(data.on_off.off_pulse, off_pulse_gt)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(2.5)
        train_ids_gt.append(3)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 5th train was lost

        # 6th train (off pulse is followed by an off pulse)
        data = self._data[5]
        self._proc.process(data)

        self.assertTrue(data.on_off.on_pulse is None)
        self.assertTrue(data.on_off.off_pulse is None)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(None)
        train_ids_gt.append(5)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 7th train
        data = self._data[6]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        self.assertTrue(data.on_off.off_pulse is None)
        on_ma_gt = [0, 3.666667, 0, 3.666667, 0]  # (1 + 3 + 7) / 3
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(None)
        train_ids_gt.append(6)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 7th train was sent twice (on train is followed by an on train)
        data = self._data[6]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        self.assertTrue(data.on_off.off_pulse is None)
        on_ma_gt = [0, 3.666667, 0, 3.666667, 0]  # should be unchanged
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(None)
        train_ids_gt.append(6)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 8th train was lost

        # 9th train (on train is followed by an on train)
        data = self._data[8]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        self.assertTrue(data.on_off.off_pulse is None)
        on_ma_gt = [0, 4.333333, 0, 4.333333, 0]  # (1 + 3 + 9)/3
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(None)
        train_ids_gt.append(8)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 10th train
        data = self._data[9]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(data.on_off.off_pulse, off_pulse_gt)
        on_ma_gt = [0, 4.333333, 0, 4.333333, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [5.333333, 0, 5.333333, 0, 5.333333]  # (2 + 4 + 10)/3
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(2.5)
        train_ids_gt.append(9)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

    def testOddEvenMode(self):
        """On-pulse has odd id."""
        self._proc.laser_mode = OpLaserMode.ODD_ON
        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3]

        fom_hist_gt = []
        train_ids_gt = []

        # 1st train
        data = self._data[0]
        self._proc.process(data)

        self.assertTrue(data.on_off.on_pulse is None)
        self.assertTrue(data.on_off.off_pulse is None)
        self.assertTrue(self._proc._on_pulses_ma is None)
        self.assertTrue(self._proc._off_pulses_ma is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(0)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        self.assertTrue(data.on_off.off_pulse is None)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        self.assertTrue(self._proc._off_pulses_ma is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(1)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(data.on_off.off_pulse, off_pulse_gt)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(2.5)
        train_ids_gt.append(2)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 4th train
        data = self._data[3]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        self.assertTrue(data.on_off.off_pulse is None)
        on_ma_gt = [0, 3, 0, 3, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(None)
        train_ids_gt.append(3)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 5th train
        data = self._data[4]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(data.on_off.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(data.on_off.off_pulse, off_pulse_gt)
        on_ma_gt = [0, 3, 0, 3, 0]
        np.testing.assert_array_almost_equal(self._proc._on_pulses_ma, on_ma_gt)
        off_ma_gt = [4, 0, 4, 0, 4]
        np.testing.assert_array_almost_equal(self._proc._off_pulses_ma, off_ma_gt)
        fom_hist_gt.append(2.5)
        train_ids_gt.append(4)
        tids, foms, _ = data.on_off.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)
