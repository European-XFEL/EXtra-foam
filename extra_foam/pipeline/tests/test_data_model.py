import unittest
from unittest.mock import patch

import numpy as np

from extra_foam.pipeline.data_model import (
    AccumulatedPairData, CorrelationData, PulseIndexMask, MovingAverageArray,
    MovingAverageScalar, ImageData, PairData, ProcessedData, PumpProbeData,
    RawImageData
)
from extra_foam.config import config


class TestPairData(unittest.TestCase):
    def testPairData(self):
        class Dummy:
            hist = PairData(device_id="device1", property="property1")

        dm = Dummy()

        dm.hist = (3, 200)
        dm.hist = (4, 220)
        corr_hist, fom_hist, info = dm.hist
        np.testing.assert_array_almost_equal([3, 4], corr_hist)
        np.testing.assert_array_almost_equal([200, 220], fom_hist)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])

        # test clear history

        del dm.hist
        corr_hist, fom_hist, info = dm.hist
        np.testing.assert_array_almost_equal([], corr_hist)
        np.testing.assert_array_almost_equal([], fom_hist)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])

        # ----------------------------
        # test when max length reached
        # ----------------------------

        Dummy.hist.MAX_LENGTH = 1000
        overflow = 10
        for i in range(Dummy.hist.MAX_LENGTH + overflow):
            dm.hist = (i, i)
        corr, fom, _ = dm.hist
        self.assertEqual(Dummy.hist.MAX_LENGTH, len(corr))
        self.assertEqual(Dummy.hist.MAX_LENGTH, len(fom))
        self.assertEqual(overflow, corr[0])
        self.assertEqual(overflow, fom[0])
        self.assertEqual(Dummy.hist.MAX_LENGTH + overflow - 1, corr[-1])
        self.assertEqual(Dummy.hist.MAX_LENGTH + overflow - 1, fom[-1])

    def testAccumulatedPairData(self):
        class Dummy:
            hist = AccumulatedPairData(
                device_id="device1", property="property1", resolution=0.1)

        dm = Dummy()

        self.assertEqual(2, AccumulatedPairData._min_count)

        # distance between two adjacent data > resolution
        dm.hist = (1, 0.3)
        dm.hist = (2, 0.4)
        corr_hist, fom_hist, info = dm.hist
        np.testing.assert_array_equal([], corr_hist)
        np.testing.assert_array_equal([], fom_hist.count)
        np.testing.assert_array_equal([], fom_hist.avg)
        np.testing.assert_array_equal([], fom_hist.min)
        np.testing.assert_array_equal([], fom_hist.max)

        dm.hist = (2.02, 0.5)
        corr_hist, fom_hist, info = dm.hist
        np.testing.assert_array_equal([2.01], corr_hist)
        np.testing.assert_array_equal([2], fom_hist.count)
        np.testing.assert_array_almost_equal([0.425], fom_hist.min)
        np.testing.assert_array_almost_equal([0.475], fom_hist.max)
        np.testing.assert_array_equal([0.45], fom_hist.avg)

        dm.hist = (2.11, 0.6)
        corr_hist, fom_hist, info = dm.hist
        np.testing.assert_array_equal([3], fom_hist.count)
        np.testing.assert_array_almost_equal([0.4591751709536137], fom_hist.min)
        np.testing.assert_array_almost_equal([0.5408248290463863], fom_hist.max)
        np.testing.assert_array_equal([0.5], fom_hist.avg)

        # new point
        dm.hist = (2.31, 1)
        dm.hist = (2.41, 2)
        corr_hist, fom_hist, info = dm.hist
        np.testing.assert_array_equal([3, 2], fom_hist.count)
        np.testing.assert_array_almost_equal([0.4591751709536137, 1.25], fom_hist.min)
        np.testing.assert_array_almost_equal([0.5408248290463863, 1.75], fom_hist.max)
        np.testing.assert_array_equal([0.5, 1.5], fom_hist.avg)

        # test clear history

        del dm.hist
        corr_hist, fom_hist, info = dm.hist
        np.testing.assert_array_equal([], corr_hist)
        np.testing.assert_array_equal([], fom_hist.count)
        np.testing.assert_array_equal([], fom_hist.avg)
        np.testing.assert_array_equal([], fom_hist.min)
        np.testing.assert_array_equal([], fom_hist.max)

        # ----------------------------
        # test when max length reached
        # ----------------------------

        Dummy.hist.MAX_LENGTH = 1000
        Dummy.hist._resolution = 1.0
        overflow = 10
        for i in range(2000 + 2 * overflow):
            # two adjacent data point will be grouped together since
            # resolution is 1.0
            dm.hist = (i, i)
        corr_hist, fom_hist, _ = dm.hist
        self.assertEqual(Dummy.hist.MAX_LENGTH, len(corr_hist))
        self.assertEqual(Dummy.hist.MAX_LENGTH, len(fom_hist.avg))
        self.assertEqual(2*overflow + 0.5, corr_hist[0])
        self.assertEqual(2*overflow + 0.5, fom_hist.avg[0])
        self.assertEqual(2*(Dummy.hist.MAX_LENGTH + overflow - 1) + 0.5, corr_hist[-1])
        self.assertEqual(2*(Dummy.hist.MAX_LENGTH + overflow - 1) + 0.5, fom_hist.avg[-1])


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


class TestMovingAverageArray1D(unittest.TestCase):
    def test1darray(self):
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


class TestPumpProbeData(unittest.TestCase):
    def testGeneral(self):
        data = PumpProbeData()

        data.fom = 1
        data.update_hist(11)
        data.fom =2
        data.update_hist(22)

        tids, foms, _ = data.fom_hist
        np.testing.assert_array_equal([11, 22], tids)
        np.testing.assert_array_equal([1, 2], foms)

        data.reset = True
        data.fom = 3
        data.update_hist(33)
        tids, foms, _ = data.fom_hist
        np.testing.assert_array_equal([33], tids)
        np.testing.assert_array_equal([3], foms)


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

    @patch.dict(config._data, {'PIXEL_SIZE': 2e-3})
    def testInitWithSpecifiedParameters(self):

        # ---------------------
        # pulse-resolved data
        # ---------------------
        with self.assertRaises(ValueError):
            ImageData.from_array(np.ones((2, 2, 2)), sliced_indices=[0, 1, 2])

        with self.assertRaises(ValueError):
            ImageData.from_array(np.ones((2, 2, 2)), sliced_indices=[1, 1, 1])

        imgs = np.ones((3, 2, 2))
        imgs[:, 0, :] = 2
        image_data = ImageData.from_array(imgs,
                                          threshold_mask=(0, 1),
                                          background=-100,
                                          poi_indices=[0, 1])

        self.assertEqual(2e-3, image_data.pixel_size)
        self.assertIsInstance(image_data.images, list)
        self.assertEqual(3, image_data.n_images)

        self.assertListEqual([0, 1, 2], image_data.sliced_indices)
        np.testing.assert_array_equal(np.array([[2., 2.], [1., 1.]]),
                                      image_data.images[0])
        np.testing.assert_array_equal(np.array([[2., 2.], [1., 1.]]),
                                      image_data.images[1])
        self.assertIsNone(image_data.images[2])

        np.testing.assert_array_equal(np.array([[2., 2.], [1., 1.]]),
                                      image_data.mean)
        np.testing.assert_array_equal(np.array([[0., 0.], [1., 1.]]),
                                      image_data.masked_mean)

        self.assertEqual(-100, image_data.background)
        self.assertEqual((0, 1), image_data.threshold_mask)

        # ---------------------
        # train-resolved data
        # ---------------------
        with self.assertRaises(ValueError):
            ImageData.from_array(np.ones((2, 2)), sliced_indices=[0])

        img = np.array([[2, 1], [1, 1]])
        image_data = ImageData.from_array(img, threshold_mask=(0, 1))

        self.assertEqual([0], image_data.sliced_indices)
        self.assertEqual([None], image_data.images)
        self.assertEqual(1, image_data.n_images)

        np.testing.assert_array_equal(np.array([[2., 1.], [1., 1.]]),
                                      image_data.mean)
        np.testing.assert_array_equal(np.array([[0., 1.], [1., 1.]]),
                                      image_data.masked_mean)

        self.assertEqual(0, image_data.background)
        self.assertEqual((0, 1), image_data.threshold_mask)


class TestCorrelationData(unittest.TestCase):
    def testGeneral(self):
        data = CorrelationData()

        self.assertIsInstance(data.correlation1.__class__.hist, PairData)
        self.assertIsNot(data.correlation1.__class__.hist,
                         data.correlation2.__class__.hist)

        # test update_params and clear

        data.correlation1.update_params(1, 11, 'dev1', 'ppt1', 0.0)
        data.correlation2.update_params(2, 22, 'dev2', 'ppt2', 0.0)
        data.update_hist()

        x, y, info = data.correlation1.hist
        np.testing.assert_array_equal([1], x)
        np.testing.assert_array_equal([11], y)
        self.assertEqual('dev1', info['device_id'])
        self.assertEqual('ppt1', info['property'])
        self.assertEqual(0.0, info['resolution'])

        data.correlation1.x = None
        data.correlation1.reset = True
        data.correlation2.reset = True
        data.update_hist()

        x, y, info = data.correlation1.hist
        np.testing.assert_array_equal([], x)
        np.testing.assert_array_equal([], y)
        self.assertEqual('dev1', info['device_id'])
        self.assertEqual('ppt1', info['property'])
        self.assertEqual(0.0, info['resolution'])

        # test resolution switching

        data.correlation1.update_params(1, 11, 'dev1', 'ppt1', 1.0)
        data.update_hist()
        self.assertIsInstance(data.correlation1.__class__.hist, AccumulatedPairData)
        self.assertIsInstance(data.correlation2.__class__.hist, PairData)

        data.correlation1.update_params(1, 11, 'dev1', 'ppt1', 0.0)
        data.update_hist()
        self.assertIsInstance(data.correlation1.__class__.hist, PairData)


class TestIndexMask(unittest.TestCase):
    def testGeneral(self):
        mask = PulseIndexMask()

        mask.mask([0, 5])
        mask.mask(7)
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
            roi.geometry = [-3, -4, 2, 2]
            self.assertIsNone(roi.rect(img))

            # has intersection
            roi.geometry = [1, 2, 3, 2]
            np.testing.assert_array_equal(img[..., 2:2+2, 1:1+3], roi.rect(img))
