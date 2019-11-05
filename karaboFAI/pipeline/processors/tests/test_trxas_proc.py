"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.pipeline.processors.tests import _BaseProcessorTest
from karaboFAI.pipeline.processors.tr_xas import TrXasProcessor


class TestTrXasProcessor(_BaseProcessorTest):
    def setUp(self):
        self._proc = TrXasProcessor()

    def tearDown(self):
        self._proc._clear_history()

    def test1dBinning(self):
        proc = self._proc

        n = 10
        proc._a13 = np.random.randn(n).tolist()
        proc._a23 = np.random.randn(n).tolist()
        proc._a21 = np.random.randn(n).tolist()

        proc._delays = np.arange(10).tolist()
        proc._n_delay_bins = 4
        proc._delay_range = [1, 3]

        proc._new_1d_binning()
        self.assertFalse(proc._bin1d)
        self.assertTrue(proc._bin2d)
        self.assertEqual(4, len(proc._a13_stats))
        self.assertEqual(4, len(proc._a23_stats))
        self.assertEqual(4, len(proc._a21_stats))
        self.assertListEqual([1, 0, 1, 1], proc._delay_bin_counts.tolist())
        self.assertListEqual([1., 1.5, 2., 2.5, 3.], proc._delay_bin_edges.tolist())

        # new outsider data point
        proc._update_1d_binning(0.1, 0.2, 0.3, 3)  # delay = 3 has index 3 instead of 2
        self.assertListEqual([1, 0, 1, 1], proc._delay_bin_counts.tolist())

        # new valid data point
        proc._update_1d_binning(0.1, 0.2, 0.3, 2)  # delay = 2 has index 2 instead of 1
        self.assertListEqual([1, 0, 2, 1], proc._delay_bin_counts.tolist())

        # TODO: test moving average calculation

    def test2dBinning(self):
        proc = self._proc

        n = 10
        proc._a21 = np.random.randn(n).tolist()

        proc._delays = np.arange(10).tolist()
        proc._energies = (np.arange(10) + 1).tolist()
        proc._n_delay_bins = 4
        proc._delay_range = [1, 8]
        proc._n_energy_bins = 2
        proc._energy_range = [2, 6]

        proc._new_2d_binning()
        self.assertTrue(proc._bin1d)
        self.assertFalse(proc._bin2d)

        self.assertTupleEqual((2, 4), proc._a21_heat.shape)
        self.assertTupleEqual((2, 4), proc._a21_heatcount.shape)
        self.assertListEqual([[2, 0, 0, 0], [0, 2, 1, 0]], proc._a21_heatcount.tolist())
        self.assertIsNone(proc._delay_bin_edges)  # calculated in _new_1d_binning
        self.assertListEqual([2, 4, 6], proc._energy_bin_edges.tolist())

        # we need the delay bin edges
        proc._a13 = np.random.randn(n).tolist()
        proc._a23 = np.random.randn(n).tolist()
        proc._new_1d_binning()

        # new outsider data point
        proc._update_2d_binning(0.1, 7, 6)  # energy = 6 has index 2 instead of 1
        self.assertListEqual([[2, 0, 0, 0], [0, 2, 1, 0]], proc._a21_heatcount.tolist())

        # new valid data point
        proc._update_2d_binning(0.1, 1, 2)  # energy = 2 has index 0 instead of -1
        self.assertListEqual([[3, 0, 0, 0], [0, 2, 1, 0]], proc._a21_heatcount.tolist())

        # TODO: test moving average calculation
