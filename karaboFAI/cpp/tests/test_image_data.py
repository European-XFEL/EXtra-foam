import unittest
import time

import numpy as np

from karaboFAI.pipeline.processors.image_processor import \
    RawImageData as RawImageDataPy

from karaboFAI.cpp import RawImageData


class TestRawImageDataCpp(unittest.TestCase):
    def testTrainResolved(self):
        data = RawImageData(np.ones((3, 3), dtype=np.float32))
        self.assertEqual(1, data.nImages())

        data.setMovingAverageWindow(5)
        # ma_window will not change until the next data arrive
        self.assertEqual(1, data.getMovingAverageWindow())
        self.assertEqual(1, data.getMovingAverageCount())
        data.setImages(3*np.ones((3, 3), dtype=np.float32))
        self.assertEqual(5, data.getMovingAverageWindow())
        self.assertEqual(2, data.getMovingAverageCount())
        np.testing.assert_array_equal(2*np.ones((3, 3), dtype=np.float32),
                                      data.getImages())

        # set a ma window which is smaller than the current window
        data.setMovingAverageWindow(3)
        self.assertEqual(5, data.getMovingAverageWindow())
        self.assertEqual(2, data.getMovingAverageCount())
        new_data = 2 * np.ones((3, 3), dtype=np.float32)
        data.setImages(new_data)
        self.assertEqual(3, data.getMovingAverageWindow())
        self.assertEqual(1, data.getMovingAverageCount())
        np.testing.assert_array_equal(new_data, data.getImages())

        # set an image with a different shape
        new_data = 2 * np.ones((3, 1), dtype=np.float32)
        data.setImages(new_data)
        self.assertEqual(3, data.getMovingAverageWindow())
        self.assertEqual(1, data.getMovingAverageCount())
        np.testing.assert_array_equal(new_data, data.getImages())

    def testPulseResolved(self):
        data = RawImageData(np.ones((3, 4, 4), dtype=np.float32))

        self.assertEqual(3, data.nImages())

        data.setMovingAverageWindow(10)
        # ma_window will not change until the next data arrive
        self.assertEqual(1, data.getMovingAverageWindow())
        self.assertEqual(1, data.getMovingAverageCount())
        new_data = 5 * np.ones((3, 4, 4), dtype=np.float32)
        data.setImages(new_data)
        self.assertEqual(10, data.getMovingAverageWindow())
        self.assertEqual(2, data.getMovingAverageCount())
        np.testing.assert_array_equal(3*np.ones((3, 4, 4), dtype=np.float32),
                                      data.getImages())

        # set a ma window which is smaller than the current window
        data.setMovingAverageWindow(2)
        self.assertEqual(10, data.getMovingAverageWindow())
        self.assertEqual(2, data.getMovingAverageCount())
        new_data = 0.1 * np.ones((3, 4, 4), dtype=np.float32)
        data.setImages(new_data)
        self.assertEqual(2, data.getMovingAverageWindow())
        self.assertEqual(1, data.getMovingAverageCount())
        np.testing.assert_array_equal(new_data, data.getImages())

        # set a data with a different number of images
        new_data = 5 * np.ones((5, 4, 4))
        data.setImages(new_data)
        self.assertEqual(2, data.getMovingAverageWindow())
        self.assertEqual(1, data.getMovingAverageCount())
        np.testing.assert_array_equal(new_data, data.getImages())

        new_data = 1 * np.ones((5, 3, 4))
        data.setImages(new_data)
        self.assertEqual(2, data.getMovingAverageWindow())
        self.assertEqual(1, data.getMovingAverageCount())
        np.testing.assert_array_equal(new_data, data.getImages())

    def testPerformance(self):

        imgs_cpp = np.ones((60, 1024, 1024), dtype=np.float32)

        t0 = time.perf_counter()
        data_cpp = RawImageData(imgs_cpp)
        dt_ctor_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        data_cpp.setMovingAverageWindow(10)
        data_cpp.setImages(imgs_cpp)
        dt_ma_cpp = time.perf_counter() - t0

        t0 = time.perf_counter()
        imgs_cpp_ = data_cpp.getImages()
        dt_get_cpp = time.perf_counter() - t0

        # compare with Python
        imgs_py = np.ones((60, 1024, 1024), dtype=np.float32)

        t0 = time.perf_counter()
        data_py = RawImageDataPy(imgs_py)
        dt_ctor_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        data_py.ma_window = 10
        data_py.images = imgs_py
        dt_ma_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        imgs_py_ = data_py.images
        dt_get_py = time.perf_counter() - t0

        self.assertLess(dt_ma_cpp, dt_ma_py)

        print(f"Performance in constructor: "
              f"dt (cpp): {dt_ctor_cpp:.10f}, dt (numpy): {dt_ctor_py:.10f}")

        print(f"Performance in calculating the moving average: "
              f"dt (cpp): {dt_ma_cpp:.10f}, dt (numpy): {dt_ma_py:.10f}")

        print(f"Performance in retrieving the data: "
              f"dt (cpp): {dt_get_cpp:.10f}, dt (numpy): {dt_get_py:.10f}")