import unittest

import numpy as np

from karaboFAI.windows import LaserOnOffWindow
from karaboFAI.data_processing import (
    ProcessedData, DataSource, Data4Visualization
)
from karaboFAI.main_fai_gui import MainFaiGUI


class TestLaserOnOffWindow(unittest.TestCase):
    gui = MainFaiGUI("LPD")
    win = LaserOnOffWindow(Data4Visualization(), parent=gui)

    gui.data_ctrl_widget.data_source_sgn.emit(
        DataSource.CALIBRATED_FILE)
    gui.analysis_ctrl_widget.normalization_range_sgn.emit(1, 5)
    gui.analysis_ctrl_widget.diff_integration_range_sgn.emit(1, 5)
    gui.analysis_ctrl_widget.ma_window_size_sgn.emit(9999)

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

    def testNormalMode(self):
        self.gui.analysis_ctrl_widget.on_off_pulse_ids_sgn.emit(
            "normal", [0, 2], [1, 3])

        win = self.win
        win._reset()

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
        on_ma_gt = [0, 3., 0, 3., 0]  # (1+2+3+4+5)/5
        np.testing.assert_array_almost_equal(win._on_pulses_ma, on_ma_gt)
        off_ma_gt = [3., 0, 3., 0, 3.]
        np.testing.assert_array_almost_equal(win._off_pulses_ma, off_ma_gt)

        np.testing.assert_array_almost_equal(win._fom_hist, [2.5]*5)
        np.testing.assert_array_almost_equal(win._fom_hist_train_id, range(5))

    def testEvenOddMode(self):
        """On-pulse has even id."""
        self.gui.analysis_ctrl_widget.on_off_pulse_ids_sgn.emit(
            "even/odd", [0, 2], [1, 3])

        win = self.win
        win._reset()

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
        self.gui.analysis_ctrl_widget.on_off_pulse_ids_sgn.emit(
            "odd/even", [0, 2], [1, 3])

        win = self.win
        win._reset()

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
