"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from extra_foam.pipeline.processors.tests import _BaseProcessorTest
from extra_foam.pipeline.processors.tr_xas import TrXasProcessor


class TestTrXasProcessor(_BaseProcessorTest):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = TrXasProcessor()
        yield
        self._proc._clear_history()

    def testNotTriggered(self):
        proc = self._proc
        data, processed = self.simple_data(1001, (4, 2, 2))

        # nothing should happen
        proc._meta.has_analysis = MagicMock(return_value=False)
        proc._get_data_point = MagicMock()
        proc.process(data)
        proc._get_data_point.assert_not_called()

    def test1dBinning(self):
        proc = self._proc

        n = 10
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))
        proc._a21.extend(np.random.randn(n))

        proc._delays.extend(np.arange(10))
        proc._n_delay_bins = 4
        proc._delay_range = [1, 3]

        proc._new_1d_binning()
        assert not proc._bin1d
        assert proc._bin2d
        assert 4 == len(proc._a13_stats)
        assert 4 == len(proc._a23_stats)
        assert 4 == len(proc._a21_stats)
        assert [1, 0, 1, 1] == proc._delay_bin_counts.tolist()
        assert [1., 1.5, 2., 2.5, 3.] == proc._delay_bin_edges.tolist()

        # new outsider data point
        proc._update_1d_binning(0.1, 0.2, 0.3, 3.5)  # index 3
        assert [1, 0, 1, 1] == proc._delay_bin_counts.tolist()

        # new valid data point
        proc._update_1d_binning(0.1, 0.2, 0.3, 2)  # index 2
        assert [1, 0, 2, 1] == proc._delay_bin_counts.tolist()

        # TODO: test moving average calculation

    def test2dBinning(self):
        proc = self._proc

        n = 10
        proc._a21.extend(np.random.randn(n))
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))

        proc._delays.extend(np.arange(10))
        proc._energies.extend(np.arange(10) + 1)
        proc._n_delay_bins = 4
        proc._delay_range = [1, 8]
        proc._n_energy_bins = 2
        proc._energy_range = [2, 6]

        proc._new_2d_binning()
        assert proc._bin1d
        assert not proc._bin2d

        assert (4, 2) == proc._a21_heat.shape
        assert (4, 2) == proc._a21_heatcount.shape
        assert [[2, 0], [0, 2], [0, 1], [0, 0]], proc._a21_heatcount.tolist()
        assert [2, 4, 6] == proc._energy_bin_edges.tolist()
        assert proc._delay_bin_edges is None  # calculated in _new_1d_binning
        proc._new_1d_binning()
        assert [1.0, 2.75, 4.5, 6.25, 8.0] == proc._delay_bin_edges.tolist()

        # new outsider data point
        proc._update_2d_binning(0.1, 6.5, 7)  # index (2, 3)
        assert [[2, 0], [0, 2], [0, 1], [0, 0]] == proc._a21_heatcount.tolist()

        # new valid data point
        proc._update_2d_binning(0.1, 2, 1)  # index (0, 0)
        assert [[3, 0], [0, 2], [0, 1], [0, 0]] == proc._a21_heatcount.tolist()

        # TODO: test moving average calculation

    @patch('extra_foam.ipc.ProcessLogger.error')
    def testProcess(self, error):
        proc = self._proc
        data, processed = self.simple_data(1001, (4, 2, 2))

        proc._meta.has_analysis = MagicMock(return_value=True)
        proc._n_delay_bins = 4
        proc._delay_range = [-1, 1]
        proc._n_energy_bins = 2
        proc._energy_range = [-1, 1]
        n = 10
        proc._delays.extend(np.random.randn(n))
        proc._energies.extend(np.random.randn(n))
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))
        proc._a21.extend(np.random.randn(n))
        proc.process(data)

        xas = processed.trxas
        np.testing.assert_array_almost_equal([-.75, -0.25, 0.25, .75], xas.delay_bin_centers)
        np.testing.assert_array_almost_equal([-.5, .5], xas.energy_bin_centers)
