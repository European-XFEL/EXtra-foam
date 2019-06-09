import unittest
import time

import numpy as np

from karaboFAI.xtnumpy import xt_nanmean_image

from karaboFAI.algorithms import nanmean_image


class TestXtnumpy(unittest.TestCase):
    def testXtNanmeanImage(self):
        # test invalid shapes
        data = np.ones([2, 2])
        with self.assertRaises(ValueError):
            nanmean_image(data)

        data = np.ones([2, 2, 2, 2])
        with self.assertRaises(ValueError):
            nanmean_image(data)

        # test nanmean

        data = np.ones([2, 4, 2], dtype=np.float64)
        data[0, 0, 1] = np.nan
        data[1, 0, 1] = np.nan
        data[1, 2, 0] = np.nan
        data[0, 3, 1] = np.inf

        expected = np.array([[1., np.nan], [1., 1.], [1., 1.], [1., np.inf]])
        np.testing.assert_array_almost_equal(expected, xt_nanmean_image(data))

        # test performance

        data = np.ones((64, 1024, 1024), dtype=np.float64)
        data[::2, ::2, ::2] = np.nan

        t0 = time.perf_counter()
        ret_cpp = xt_nanmean_image(data)
        dt_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        ret_py = nanmean_image(data, max_workers=4)
        dt_py = time.perf_counter() - t0

        print(f"dt (cpp): {dt_cpp:.4f}, dt (numpy_para): {dt_py:.4f}")
