import unittest

import numpy as np

from karaboFAI.algorithms import nanmean_axis0_para


class TestPynumpy(unittest.TestCase):

    def test_nanmeanparaimp(self):
        data = np.ones([2, 4, 2])
        data[0, 0, 1] = np.nan
        data[1, 0, 1] = np.nan
        data[1, 2, 0] = np.nan

        ret = nanmean_axis0_para(data, chunk_size=2, max_workers=2)

        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., 1.]])
        np.testing.assert_array_almost_equal(expected, ret)

        ret = nanmean_axis0_para(data, chunk_size=1, max_workers=1)

        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., 1.]])
        np.testing.assert_array_almost_equal(expected, ret)
