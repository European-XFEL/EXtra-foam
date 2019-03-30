import unittest

import numpy as np

from karaboFAI.config import PumpProbeMode
from karaboFAI.pipeline.data_model import PumpProbeData, ProcessedData
from karaboFAI.pipeline.data_processor import PumpProbeProcessor


class TestPumpProbeProcessor(unittest.TestCase):
    def setUp(self):
        self._proc = PumpProbeProcessor()
        PumpProbeData.clear()

        self._proc.fom_itgt_range = (1, 5)

        self._data = []
        intensity = np.array([[0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1],
                              [0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1]])
        for i in range(10):
            self._data.append(ProcessedData(i))
            self._data[i].momentum = np.linspace(1, 5, 5)
            self._data[i].intensities = (i+1)*intensity

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
        self._proc.process(data)

        on_pulse_gt = np.array([0, 1, 0, 1, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        off_pulse_gt = np.array([1, 0, 1, 0, 1])
        np.testing.assert_array_almost_equal(off_pulse_gt, data.pp.off_pulse)
        fom_hist_gt.append(5)
        train_ids_gt.append(0)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 2, 0, 2, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        off_pulse_gt = np.array([2, 0, 2, 0, 2])
        np.testing.assert_array_almost_equal(off_pulse_gt, data.pp.off_pulse)
        fom_hist_gt.append(10)
        train_ids_gt.append(1)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 3, 0, 3, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        off_pulse_gt = np.array([3, 0, 3, 0, 3])
        np.testing.assert_array_almost_equal(off_pulse_gt, data.pp.off_pulse)
        fom_hist_gt.append(15)
        train_ids_gt.append(2)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

    def testEvenTrainOn(self):
        """On-pulse has even id."""
        self._proc.mode = PumpProbeMode.EVEN_TRAIN_ON
        self._proc.on_pulse_ids = [0, 2]
        self._proc.off_pulse_ids = [1, 3]

        fom_hist_gt = []
        train_ids_gt = []

        # 1st train
        data = self._data[0]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 1, 0, 1, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(0)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 1, 0, 1, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        off_pulse_gt = np.array([2, 0, 2, 0, 2])
        np.testing.assert_array_almost_equal(off_pulse_gt, data.pp.off_pulse)
        fom_hist_gt.append(8)
        train_ids_gt.append(1)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 3, 0, 3, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(2)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 4th train
        data = self._data[3]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 3, 0, 3, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        off_pulse_gt = np.array([4, 0, 4, 0, 4])
        np.testing.assert_array_almost_equal(off_pulse_gt, data.pp.off_pulse)
        fom_hist_gt.append(18)
        train_ids_gt.append(3)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 5th train was lost

        # 6th train (off pulse is followed by an off pulse)
        data = self._data[5]
        self._proc.process(data)

        self.assertTrue(data.pp.on_pulse is None)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(5)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 7th train
        data = self._data[6]
        self._proc.process(data)

        on_pulse_gt = [0, 7, 0, 7, 0]
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(6)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 7th train was sent twice (on train is followed by an on train)
        data = self._data[6]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 7, 0, 7, 0])  # unchanged
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(6)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 8th train was lost

        # 9th train (on train is followed by an on train)
        data = self._data[8]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 9, 0, 9, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(8)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 10th train
        data = self._data[9]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 9, 0, 9, 0])
        np.testing.assert_array_almost_equal(data.pp.on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([10, 0, 10, 0, 10])
        np.testing.assert_array_almost_equal(data.pp.off_pulse, off_pulse_gt)
        fom_hist_gt.append(48)
        train_ids_gt.append(9)
        tids, foms, _ = data.pp.foms
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
        self._proc.process(data)

        self.assertTrue(data.pp.on_pulse is None)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(0)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 2nd train
        data = self._data[1]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 2, 0, 2, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(1)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 3rd train
        data = self._data[2]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 2, 0, 2, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        off_pulse_gt = np.array([3, 0, 3, 0, 3])
        np.testing.assert_array_almost_equal(off_pulse_gt, data.pp.off_pulse)
        fom_hist_gt.append(13)
        train_ids_gt.append(2)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 4th train
        data = self._data[3]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 4, 0, 4, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        self.assertTrue(data.pp.off_pulse is None)
        fom_hist_gt.append(None)
        train_ids_gt.append(3)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)

        # 5th train
        data = self._data[4]
        self._proc.process(data)

        on_pulse_gt = np.array([0, 4, 0, 4, 0])
        np.testing.assert_array_almost_equal(on_pulse_gt, data.pp.on_pulse)
        off_pulse_gt = np.array([5, 0, 5, 0, 5])
        np.testing.assert_array_almost_equal(off_pulse_gt, data.pp.off_pulse)
        fom_hist_gt.append(23)
        train_ids_gt.append(4)
        tids, foms, _ = data.pp.foms
        np.testing.assert_array_equal(train_ids_gt, tids)
        np.testing.assert_array_equal(fom_hist_gt, foms)
