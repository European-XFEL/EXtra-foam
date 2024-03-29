import unittest
from unittest.mock import patch

import numpy as np

from extra_foam.pipeline.data_model import (
    PulseIndexMask, MovingAverageArray, MovingAverageScalar,
    ImageData, ProcessedData, RawImageData
)
from extra_foam.config import config


class TestMovingAverageScalar(unittest.TestCase):
    def testGeneral(self):
        class Dummy:
            data = MovingAverageScalar()

        dm = Dummy()

        dm.data = 1.0
        self.assertEqual(1, Dummy.data.window)
        self.assertEqual(1.0, dm.data)

        Dummy.data.window = 5
        self.assertEqual(5, Dummy.data.window)
        self.assertEqual(1, Dummy.data.count)
        dm.data = 2.0
        self.assertEqual(5, Dummy.data.window)
        self.assertEqual(2, Dummy.data.count)
        self.assertEqual(1.5, dm.data)
        dm.data = 3.0
        self.assertEqual(5, Dummy.data.window)
        self.assertEqual(3, Dummy.data.count)
        self.assertEqual(2.0, dm.data)

        # set a ma window which is smaller than the current window
        Dummy.data.window = 3
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(3, Dummy.data.count)
        self.assertEqual(2.0, dm.data)

        del dm.data
        self.assertIsNone(dm.data)
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(0, Dummy.data.count)

        dm.data = 1.0
        self.assertEqual(1.0, dm.data)
        dm.data = None
        self.assertIsNone(dm.data)
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(0, Dummy.data.count)


class TestMovingAverageArray(unittest.TestCase):
    def test1DArray(self):
        class Dummy:
            data = MovingAverageArray()

        dm = Dummy()

        arr = np.array([1, np.nan, 3], dtype=np.float32)
        dm.data = arr.copy()

        self.assertEqual(1, Dummy.data.window)

        Dummy.data.window = 5
        self.assertEqual(5, Dummy.data.window)
        self.assertEqual(1, Dummy.data.count)
        dm.data = np.array([3, 2, np.nan], dtype=np.float32)
        self.assertEqual(5, Dummy.data.window)
        self.assertEqual(2, Dummy.data.count)
        np.testing.assert_array_equal(
            np.array([2, np.nan, np.nan], dtype=np.float32), dm.data)

        # set a ma window which is smaller than the current window
        Dummy.data.window = 3
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(2, Dummy.data.count)
        np.testing.assert_array_equal(
            np.array([2, np.nan, np.nan], dtype=np.float32), dm.data)

        # set a data with a different shape
        new_arr = np.array([2, np.nan, 1, 3], dtype=np.float32)
        dm.data = new_arr
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(1, Dummy.data.count)
        np.testing.assert_array_equal(new_arr, dm.data)

        del dm.data
        self.assertIsNone(dm.data)
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(0, Dummy.data.count)

        dm.data = new_arr.copy()
        np.testing.assert_array_equal(new_arr, dm.data)
        dm.data = None
        self.assertIsNone(dm.data)
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(0, Dummy.data.count)


class TestRawImageData(unittest.TestCase):
    # This tests 2d and 3d MovingAverageArray
    def testTrainResolved(self):
        class Dummy:
            data = RawImageData()

        dm = Dummy()

        arr = np.ones((3, 3), dtype=np.float32)
        arr[0][2] = np.nan
        dm.data = arr

        self.assertEqual(1, Dummy.data.n_images)

        Dummy.data.window = 5
        self.assertEqual(5, Dummy.data.window)
        self.assertEqual(1, Dummy.data.count)
        arr = 3 * np.ones((3, 3), dtype=np.float32)
        arr[1][2] = np.nan
        dm.data = arr
        self.assertEqual(5, Dummy.data.window)
        self.assertEqual(2, Dummy.data.count)
        expected = 2 * np.ones((3, 3), dtype=np.float32)
        expected[1][2] = np.nan
        expected[0][2] = np.nan
        np.testing.assert_array_equal(expected, dm.data)

        # set a ma window which is smaller than the current window
        Dummy.data.window = 3
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(2, Dummy.data.count)
        np.testing.assert_array_equal(expected, dm.data)

        # set an image with a different shape
        new_arr = 2*np.ones((3, 1), dtype=np.float32)
        dm.data = new_arr
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(1, Dummy.data.count)
        np.testing.assert_array_equal(new_arr, dm.data)

        del dm.data
        self.assertIsNone(dm.data)
        self.assertEqual(3, Dummy.data.window)
        self.assertEqual(0, Dummy.data.count)

    def testPulseResolved(self):
        class Dummy:
            data = RawImageData()

        dm = Dummy()

        arr = np.ones((3, 4, 4), dtype=np.float32)
        arr[1][2][1] = np.nan
        self.assertEqual(0, Dummy.data.n_images)

        dm.data = arr
        self.assertEqual(3, Dummy.data.n_images)
        self.assertIs(dm.data, arr)

        Dummy.data.window = 10
        self.assertEqual(10, Dummy.data.window)
        self.assertEqual(1, Dummy.data.count)
        dm.data = 5 * np.ones((3, 4, 4), dtype=np.float32)
        dm.data[2][3][3] = np.nan
        self.assertEqual(10, Dummy.data.window)
        self.assertEqual(2, Dummy.data.count)
        expected = 3 * np.ones((3, 4, 4), dtype=np.float32)
        expected[1][2][1] = np.nan
        expected[2][3][3] = np.nan
        np.testing.assert_array_equal(expected, dm.data)

        # set a ma window which is smaller than the current window
        Dummy.data.window = 2
        self.assertEqual(2, Dummy.data.window)
        self.assertEqual(2, Dummy.data.count)
        np.testing.assert_array_equal(expected, dm.data)

        # set a data with a different number of images
        new_arr = 5 * np.ones((5, 4, 4))
        dm.data = new_arr
        self.assertEqual(2, Dummy.data.window)
        self.assertEqual(1, Dummy.data.count)
        np.testing.assert_array_equal(new_arr, dm.data)

        del dm.data
        self.assertIsNone(dm.data)
        self.assertEqual(2, Dummy.data.window)
        self.assertEqual(0, Dummy.data.count)

    def testCopyFirst(self):
        class Dummy:
            data = RawImageData(copy_first=True)

        dm = Dummy()

        arr = np.ones((3, 4, 4), dtype=np.float32)
        dm.data = arr
        self.assertIsNot(dm.data, arr)


class TestProcessedData(unittest.TestCase):
    def testGeneral(self):
        # ---------------------
        # pulse-resolved data
        # ---------------------

        data = ProcessedData(1234)

        self.assertEqual(1234, data.tid)
        self.assertEqual(0, data.n_pulses)

        data.image = ImageData.from_array(np.zeros((1, 2, 2)))
        self.assertEqual(1, data.n_pulses)

        data = ProcessedData(1235)
        data.image = ImageData.from_array(np.zeros((3, 2, 2)))
        self.assertEqual(3, data.n_pulses)

        # ---------------------
        # train-resolved data
        # ---------------------

        data = ProcessedData(1236)
        data.image = ImageData.from_array(np.zeros((2, 2)))

        self.assertEqual(1236, data.tid)
        self.assertEqual(1, data.n_pulses)


class TestImageData(unittest.TestCase):

    def testFromArray(self):
        with self.assertRaises(TypeError):
            ImageData.from_array()

        with self.assertRaises(ValueError):
            ImageData.from_array(np.ones(2))

        with self.assertRaises(ValueError):
            ImageData.from_array(np.ones((2, 2, 2, 2)))

        image_data = ImageData.from_array(np.ones((2, 2, 3)))
        self.assertEqual((2, 3), image_data.mask.shape)

        image_data = ImageData.from_array(np.ones((3, 2)))
        self.assertEqual((3, 2), image_data.mask.shape)

    @patch.dict(config._data, {'PIXEL_SIZE': 2e-3})
    def testInitWithSpecifiedParametersPS(self):
        with self.assertRaises(ValueError):
            ImageData.from_array(np.ones((2, 2, 2)), sliced_indices=[0, 1, 2])

        with self.assertRaises(ValueError):
            ImageData.from_array(np.ones((2, 2, 2)), sliced_indices=[1, 1, 1])

        imgs = np.ones((3, 2, 3))
        imgs[:, 0, :] = 2
        image_mask = np.zeros((2, 3), dtype=bool)
        image_mask[::2, ::2] = True
        image_data = ImageData.from_array(imgs,
                                          image_mask=image_mask,
                                          threshold_mask=(0, 1),
                                          poi_indices=[0, 1])

        self.assertEqual(2e-3, image_data.pixel_size)
        self.assertIsInstance(image_data.images, list)
        self.assertEqual(3, image_data.n_images)

        self.assertListEqual([0, 1, 2], image_data.sliced_indices)
        np.testing.assert_array_equal(np.array([[np.nan, np.nan, np.nan], [1., 1., 1.]]),
                                      image_data.images[0])
        np.testing.assert_array_equal(np.array([[np.nan, np.nan, np.nan], [1., 1., 1.]]),
                                      image_data.images[1])
        self.assertIsNone(image_data.images[2])

        np.testing.assert_array_equal(np.array([[2., 2., 2.], [1., 1., 1.]]),
                                      image_data.mean)
        np.testing.assert_array_equal(np.array([[np.nan, np.nan, np.nan], [1., 1., 1.]]),
                                      image_data.masked_mean)

        self.assertIsNone(image_data.gain_mean)
        self.assertIsNone(image_data.offset_mean)
        self.assertEqual((0, 1), image_data.threshold_mask)

    @patch.dict(config._data, {'PIXEL_SIZE': 2e-3})
    def testInitWithSpecifiedParametersTS(self):
        with self.assertRaises(ValueError):
            ImageData.from_array(np.ones((2, 2)), sliced_indices=[0])

        img = np.array([[2, 1], [1, 1]])
        image_data = ImageData.from_array(img, threshold_mask=(0, 1))

        self.assertEqual([0], image_data.sliced_indices)
        self.assertEqual([None], image_data.images)
        self.assertEqual(1, image_data.n_images)

        np.testing.assert_array_equal(np.array([[2., 1.], [1., 1.]]),
                                      image_data.mean)
        np.testing.assert_array_equal(np.array([[np.nan, 1.], [1., 1.]]),
                                      image_data.masked_mean)

        self.assertEqual((0, 1), image_data.threshold_mask)


class TestIndexMask(unittest.TestCase):
    def testGeneral(self):
        mask = PulseIndexMask()

        mask.mask_by_index([0, 5])
        mask.mask_by_index(7)
        self.assertEqual(3, mask.n_dropped(10))
        self.assertEqual(1, mask.n_dropped(4))
        self.assertEqual(7, mask.n_kept(10))
        self.assertEqual(3, mask.n_kept(4))
        self.assertListEqual([0, 5, 7], mask.dropped_indices(100).tolist())
        self.assertListEqual([0, 5], mask.dropped_indices(6).tolist())
        self.assertEqual(97, len(mask.kept_indices(100)))
        self.assertEqual(4, len(mask.kept_indices(6)))
        for i in [0, 5, 7]:
            self.assertNotIn(i, mask.kept_indices(100))
            self.assertNotIn(i, mask.kept_indices(6))

        mask.reset()
        self.assertEqual(10, mask.n_kept(10))

    def testMaskByArray(self):
        mask = PulseIndexMask()

        mask.mask_by_array(np.array([True, False]))
        self.assertListEqual([0], mask.dropped_indices(100).tolist())
        mask.mask_by_array(np.array([1, 1, 0, 1, 0], dtype=bool))
        self.assertListEqual([0, 1, 3], mask.dropped_indices(100).tolist())


class TestRoiGeom(unittest.TestCase):
    def setUp(self):
        self._img = np.arange(100).reshape((10, 10))
        self._img_array = np.arange(400).reshape((4, 10, 10))

    def testRect(self):
        from extra_foam.pipeline.data_model import RectRoiGeom

        for img in [self._img, self._img_array]:
            # roi.geometry == [0, 0, -1, -1]
            roi = RectRoiGeom()
            self.assertIsNone(roi.rect(img))

            # no intersection
            roi.geometry = [0, 0, 0, 2]
            self.assertIsNone(roi.rect(img))

            # has intersection
            roi.geometry = [1, 2, 3, 2]
            np.testing.assert_array_equal(img[..., 2:2+2, 1:1+3], roi.rect(img))


class TestXgmData(unittest.TestCase):

    from extra_foam.pipeline.data_model import XgmData

    def testGeneral(self):
        data = self.XgmData()

        data.intensity, data.x, data.y = 100., 0.1, -0.1
        with self.assertRaises(AttributeError):
            data.xx = 0.2


class TestDigitizerData(unittest.TestCase):

    from extra_foam.pipeline.data_model import (
        _DigitizerDataItem, _DigitizerChannelData, DigitizerData)

    def testGeneral(self):
        data = self.DigitizerData()

        self.assertIn('A', data)

        for cn, item in data.items():
            self.assertIsInstance(item, self._DigitizerDataItem)

        data['D'].pulse_integral = [1, 2, 3]
        with self.assertRaises(AttributeError):
            data['D'].sample


class TestBinData(unittest.TestCase):

    from extra_foam.pipeline.data_model import BinData

    def testGeneral(self):
        data = self.BinData()

        # test mapping
        self.assertEqual(2, len(data))
        self.assertIn(0, data)
        self.assertIn(1, data)
        self.assertNotIn(2, data)

        for b in data:
            self.assertIsInstance(b, self.BinData.BinDataItem)
        with self.assertRaises(IndexError):
            data[2]
        self.assertIsInstance(data[1], self.BinData.BinDataItem)

        # test slots
        with self.assertRaises(AttributeError):
            data.b = self.BinData.BinDataItem()


class TestCorrelationData(unittest.TestCase):

    from extra_foam.pipeline.data_model import CorrelationData

    def testGeneral(self):
        data = self.CorrelationData()

        # test mapping
        self.assertEqual(2, len(data))
        self.assertIn(0, data)
        self.assertIn(1, data)
        self.assertNotIn(2, data)

        for c in data:
            self.assertIsInstance(c, data.CorrelationDataItem)
        with self.assertRaises(IndexError):
            data[2]
        self.assertIsInstance(data[1], data.CorrelationDataItem)

        # test slots
        with self.assertRaises(AttributeError):
            data.c = data.CorrelationDataItem()


class TestHistogramData(unittest.TestCase):

    def testGeneral(self):
        from extra_foam.pipeline.data_model import _HistogramDataItem, HistogramDataPulse

        data = HistogramDataPulse()
        hist_gt, bin_centers_gt = np.arange(0, 10, 1), np.arange(0, 20, 2)

        self.assertEqual(0, len(data))

        # __getitem__ and __setitem__

        with self.assertRaises(KeyError):
            data['abc'] = (hist_gt, bin_centers_gt)

        with self.assertRaises(KeyError):
            data['1'] = (hist_gt, bin_centers_gt)

        with self.assertRaises(KeyError):
            data[2700] = (hist_gt, bin_centers_gt)

        data[1] = (hist_gt, bin_centers_gt, 1, 2, 3)
        data[100] = (hist_gt, bin_centers_gt, 1, 2, 3)
        np.testing.assert_array_equal(hist_gt, data[100].hist)
        np.testing.assert_array_equal(bin_centers_gt, data[100].bin_centers)
        self.assertEqual(1, data[100].mean)
        self.assertEqual(2, data[100].median)
        self.assertEqual(3, data[100].std)

        # __iter__
        for _, item in data.items():
            self.assertIsInstance(item, _HistogramDataItem)

        # __delitem__ and __len__
        self.assertEqual(2, len(data))
        del data[100]
        self.assertEqual(1, len(data))
        del data[1]
        self.assertEqual(0, len(data))
