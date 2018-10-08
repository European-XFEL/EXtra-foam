import unittest

import numpy as np

from karaboFAI.widgets.pyqtgraph import mkQApp
from karaboFAI.widgets import LaserOnOffWindow
from karaboFAI.data_processing import ProcessedData
from karaboFAI.main_gui import MainGUI


mkQApp()


class TestLaserOnOffWindow(unittest.TestCase):
    def setUp(self):
        self._data = []
        intensity = np.array([[0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1],
                              [0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1]])
        for i in range(10):
            self._data.append(ProcessedData(i,
                                            momentum=np.linspace(1, 5, 5),
                                            intensity=(i+1)*intensity))

        self._on_pulses_ids = [0, 2]
        self._off_pulses_ids = [1, 3]
        self._normalization_range = (1, 5)
        self._fom_range = (1, 5)

        self._available_modes = list(LaserOnOffWindow.modes.keys())

    def testNormalMode(self):
        win = LaserOnOffWindow(MainGUI.Data4Visualization(),
                               self._on_pulses_ids,
                               self._off_pulses_ids,
                               self._normalization_range,
                               self._fom_range,
                               self._available_modes[0],
                               ma_window_size=4)

        # 1st train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[0])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 1, 0, 1, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [1, 0, 1, 0, 1]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, range(1))

        # 2nd train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[1])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 1.5, 0, 1.5, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [1.5, 0, 1.5, 0, 1.5]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5]*2)
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, range(2))

        # 3rd train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[2])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [2, 0, 2, 0, 2]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5]*3)
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, range(3))

        # 4th train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[3])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 2.5, 0, 2.5, 0]  # (1+2+3+4)/4
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [2.5, 0, 2.5, 0, 2.5]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5]*4)
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, range(4))

        # 5th train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[4])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 3.5, 0, 3.5, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3.5, 0, 3.5, 0, 3.5]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5]*5)
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, range(5))

    def testEvenOddMode(self):
        """On-pulse has even id."""
        win = LaserOnOffWindow(MainGUI.Data4Visualization(),
                               self._on_pulses_ids,
                               self._off_pulses_ids,
                               self._normalization_range,
                               self._fom_range,
                               self._available_modes[1],
                               ma_window_size=9999)

        # 1st train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[0])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        self.assertTrue(normalized_off_pulse is None)
        on_ma_gt = [0, 1, 0, 1, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        self.assertTrue(win._off_pulses_ma is None)

        np.testing.assert_array_almost_equal(win._fom_hist, [])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [])

        # 2nd train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[1])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 1, 0, 1, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [2, 0, 2, 0, 2]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [1])

        # 3rd train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[2])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        self.assertTrue(normalized_off_pulse is None)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [2, 0, 2, 0, 2]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [1])

        # 4th train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[3])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5, 2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [1, 3])

        # 5th train was lost

        # 6th train (off pulse is followed by an off pulse)
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[5])
        self.assertTrue(normalized_on_pulse is None)
        self.assertTrue(normalized_off_pulse is None)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5, 2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [1, 3])

        # 7th train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[6])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        self.assertTrue(normalized_off_pulse is None)
        on_ma_gt = [0, 3.666667, 0, 3.666667, 0]  # (1 + 3 + 7) / 3
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5, 2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [1, 3])

        # 7th train was sent twice (on train is followed by an on train)
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[6])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        self.assertTrue(normalized_off_pulse is None)
        on_ma_gt = [0, 3.666667, 0, 3.666667, 0]  # should be unchanged
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5, 2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [1, 3])

        # 8th train was lost

        # 9th train (on train is followed by an on train)
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[8])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        self.assertTrue(normalized_off_pulse is None)
        on_ma_gt = [0, 4.333333, 0, 4.333333, 0]  # (1 + 3 + 9)/3
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5, 2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [1, 3])

        # 10th train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[9])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 4.333333, 0, 4.333333, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [5.333333, 0, 5.333333, 0, 5.333333]  # (2 + 4 + 10)/3
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5, 2.5, 2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [1, 3, 9])

    def testOddEvenMode(self):
        """On-pulse has odd id."""
        win = LaserOnOffWindow(MainGUI.Data4Visualization(),
                               self._on_pulses_ids,
                               self._off_pulses_ids,
                               self._normalization_range,
                               self._fom_range,
                               self._available_modes[2],
                               ma_window_size=9999)

        # 1st train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[0])
        self.assertTrue(normalized_on_pulse is None)
        self.assertTrue(normalized_off_pulse is None)
        self.assertTrue(win._on_pulses_ma is None)
        self.assertTrue(win._off_pulses_ma is None)

        np.testing.assert_array_almost_equal(win._fom_hist, [])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [])

        # 2nd train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[1])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        self.assertTrue(normalized_off_pulse is None)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        self.assertTrue(win._off_pulses_ma is None)

        np.testing.assert_array_almost_equal(win._fom_hist, [])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [])

        # 3rd train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[2])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 2, 0, 2, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [2])

        # 4th train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[3])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        self.assertTrue(normalized_off_pulse is None)
        on_ma_gt = [0, 3, 0, 3, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3, 0, 3, 0, 3]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [2])

        # 5th train
        normalized_on_pulse, normalized_off_pulse = win._update(self._data[4])
        on_pulse_gt = np.array([0, 0.5, 0, 0.5, 0])
        np.testing.assert_array_almost_equal(normalized_on_pulse, on_pulse_gt)
        off_pulse_gt = np.array([0.5, 0, 0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(normalized_off_pulse, off_pulse_gt)
        on_ma_gt = [0, 3, 0, 3, 0]
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [4, 0, 4, 0, 4]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5, 2.5])
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, [2, 4])