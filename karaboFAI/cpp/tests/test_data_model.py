import unittest
import time

import numpy as np

from karaboFAI.pipeline.data_model import RawImageData as RawImageDataPy

from karaboFAI.cpp import (
    RawImageDataFloat, RawImageDataDouble,
    MovingAverageArrayFloat, MovingAverageArrayDouble,
    MovingAverageFloat, MovingAverageDouble
)


class TestRawImageDataCpp(unittest.TestCase):
    def testTrainResolved(self):
        arr = np.ones((3, 3), dtype=np.float32)
        data = RawImageDataFloat(arr)
        self.assertEqual(1, data.nImages())

        data.setWindow(5)
        # window will not change until the next data arrive
        self.assertEqual(5, data.window())
        self.assertEqual(1, data.count())
        data.set(3*arr)
        self.assertEqual(5, data.window())
        self.assertEqual(2, data.count())
        np.testing.assert_array_equal(2 * arr, data.get())

        # set a ma window which is smaller than the current window
        data.setWindow(3)
        self.assertEqual(3, data.window())
        # count and value have not changed yet
        self.assertEqual(2, data.count())
        np.testing.assert_array_equal(2 * arr, data.get())

        # set an image with a different shape
        new_arr = 2 * np.ones((3, 1), dtype=np.float32)
        data.set(new_arr)
        self.assertEqual(3, data.window())
        self.assertEqual(1, data.count())
        np.testing.assert_array_equal(new_arr, data.get())

    def testPulseResolved(self):
        arr = np.ones((3, 4, 4), dtype=np.float32)

        data = RawImageDataFloat(arr)

        self.assertEqual(3, data.nImages())

        data.setWindow(10)
        self.assertEqual(10, data.window())
        self.assertEqual(1, data.count())
        new_data = 5 * arr
        data.set(new_data)
        self.assertEqual(10, data.window())
        self.assertEqual(2, data.count())
        np.testing.assert_array_equal(3 * arr, data.get())

        # set a ma window which is smaller than the current window
        data.setWindow(2)
        self.assertEqual(2, data.window())
        # count and value have not changed yet
        self.assertEqual(2, data.count())
        np.testing.assert_array_equal(3 * arr, data.get())

        # set a data with a different number of images
        new_arr = 5 * np.ones((5, 4, 4), dtype=np.float32)
        data.set(new_arr)
        self.assertEqual(2, data.window())
        self.assertEqual(1, data.count())
        np.testing.assert_array_equal(new_arr, data.get())

    def testPerformance(self):
        self._run_performance_with_type(np.float32)
        self._run_performance_with_type(np.float64)

    def _run_performance_with_type(self, dtype):
        imgs_cpp = np.ones((64, 1024, 512), dtype=dtype)

        t0 = time.perf_counter()
        data_cpp = RawImageDataFloat(imgs_cpp)
        dt_ctor_cpp = time.perf_counter() - t0

        data_cpp.setWindow(10)
        t0 = time.perf_counter()
        data_cpp.set(imgs_cpp)
        dt_ma_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        imgs_cpp_ = data_cpp.get()
        dt_get_cpp = time.perf_counter() - t0

        # compare with Python
        class Dummy:
            data = RawImageDataPy()

        imgs_py = np.ones((64, 1024, 512), dtype=dtype)

        t0 = time.perf_counter()
        dummy =  Dummy()
        dummy.data = imgs_py
        dt_ctor_py = time.perf_counter() - t0

        Dummy.data.window = 10
        t0 = time.perf_counter()
        dummy.data = imgs_py
        dt_ma_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        imgs_py_ = dummy.data
        dt_get_py = time.perf_counter() - t0

        print(f"Performance {dtype} in constructor: "
              f"dt (cpp): {dt_ctor_cpp:.10f}, dt (numpy): {dt_ctor_py:.10f}")

        print(f"Performance {dtype}  in calculating the moving average: "
              f"dt (cpp): {dt_ma_cpp:.10f}, dt (numpy+cpp): {dt_ma_py:.10f}")

        print(f"Performance {dtype}  in retrieving the data: "
              f"dt (cpp): {dt_get_cpp:.10f}, dt (numpy): {dt_get_py:.10f}")


class TestMovingAverageArrayCpp(unittest.TestCase):
    def test_general(self):
        self._run_with_type(np.float32)
        self._run_with_type(np.float64)

    def _run_with_type(self, dtype):
        vec = np.ones(10, dtype=dtype)
        if dtype == np.float32:
            data = MovingAverageArrayFloat(vec)
        else:  # dtype == np.float64:
            data = MovingAverageArrayDouble(vec)

        self.assertEqual(1, data.window())
        self.assertEqual(1, data.count())
        np.testing.assert_array_equal(vec, data.get())

        data.setWindow(3)
        data.set(2 * vec)
        self.assertEqual(3, data.window())
        self.assertEqual(2, data.count())
        np.testing.assert_array_equal(1.5 * vec, data.get())

        data.set(3 * vec)
        self.assertEqual(3, data.window())
        self.assertEqual(3, data.count())
        np.testing.assert_array_equal(2.0 * vec, data.get())

        data.set(4 * vec)
        self.assertEqual(3, data.window())
        self.assertEqual(3, data.count())
        np.testing.assert_array_almost_equal(2.666667 * vec, data.get())

        # new MA window is smaller than the current one
        data.setWindow(2)
        self.assertEqual(2, data.window())
        # count and value have not changed yet
        self.assertEqual(3, data.count())
        np.testing.assert_array_almost_equal(2.666667 * vec, data.get())

        # set an array with a different size
        vec = np.ones(5, dtype=dtype)
        data.set(vec)
        self.assertEqual(2, data.window())
        self.assertEqual(1, data.count())
        np.testing.assert_array_equal(vec, data.get())

        # set an array with a different size again
        vec = np.ones(2, dtype=dtype)
        data.set(vec)
        self.assertEqual(2, data.window())
        self.assertEqual(1, data.count())
        np.testing.assert_array_equal(vec, data.get())


class TestMovingAverageCpp(unittest.TestCase):
    def test_general(self):
        self._run_with_type(np.float32)
        self._run_with_type(np.float64)

    def _run_with_type(self, dtype):
        if dtype == np.float32:
            data = MovingAverageFloat(1.0)
        else:  # dtype == np.float64:
            data = MovingAverageDouble(1.0)

        self.assertEqual(1, data.window())
        self.assertEqual(1, data.count())
        self.assertEqual(1.0, data.get())

        data.setWindow(3)
        data.set(2.0)
        self.assertEqual(3, data.window())
        self.assertEqual(2, data.count())
        self.assertEqual(1.5, data.get())

        data.set(3.0)
        self.assertEqual(3, data.window())
        self.assertEqual(3, data.count())
        self.assertEqual(2.0, data.get())

        data.set(4.0)
        self.assertEqual(3, data.window())
        self.assertEqual(3, data.count())
        self.assertAlmostEqual(2.666667, data.get(), places=6)

        # new MA window is smaller than the current one
        data.setWindow(2)
        self.assertEqual(2, data.window())
        # count and value have not changed yet
        self.assertEqual(3, data.count())
        self.assertAlmostEqual(2.666667, data.get(), places=6)
