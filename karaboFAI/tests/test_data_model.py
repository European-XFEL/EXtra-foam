import unittest

import numpy as np

from karaboFAI.data_processing.data_model import (
    AbstractData, ImageData, ProcessedData, TrainData
)


class TestImageData(unittest.TestCase):
    def test_invalidInput(self):
        with self.assertRaises(TypeError):
            ImageData([1, 2, 3])

        with self.assertRaises(ValueError):
            ImageData(np.arange(2))

        with self.assertRaises(ValueError):
            ImageData(np.arange(16).reshape(2, 2, 2, 2))

    def test_cropping(self):
        imgs = np.arange(100, dtype=np.float).reshape(10, 10)
        img_data = ImageData(imgs)

        img_data.crop_area = (6, 7, 1, 2)
        self.assertTupleEqual(img_data.mean.shape, (7, 6))
        self.assertTupleEqual((1, 2), img_data.pos(0, 0))

        img_data.crop_area = (3, 4, 0, 1)
        self.assertTupleEqual(img_data.mean.shape, (4, 3))
        self.assertTupleEqual((0, 1), img_data.pos(0, 0))

    def test_poni(self):
        imgs = np.arange(20, dtype=np.float).reshape(5, 4)

        img_data = ImageData(np.copy(imgs))

        self.assertTupleEqual((0, 0), img_data.poni)
        img_data.crop_area = (3, 2, 0, 1)
        self.assertTupleEqual((-1, 0), img_data.poni)

        img_data.crop_area = (3, 2, 1, 2)
        img_data.poni = (-2, 12)
        self.assertTupleEqual((-4, 11), img_data.poni)

    def test_trainresolved(self):
        imgs_orig = np.arange(16, dtype=np.float).reshape(4, 4)

        img_data = ImageData(np.copy(imgs_orig))
        mask = (1, 4)
        img_data.threshold_mask = mask
        bkg = 1.0
        crop_area = (3, 2, 0, 1)
        img_data.background = bkg
        img_data.crop_area = crop_area

        # calculate the ground truth
        w, h, x, y = crop_area
        imgs = np.copy(imgs_orig)[y:y+h, x:x+w]
        imgs -= bkg

        self.assertEqual(1, img_data.n_images)

        np.testing.assert_array_equal(imgs, img_data.images)

        np.testing.assert_array_equal(imgs, img_data.mean)

        # test threshold mask
        masked_imgs = np.copy(imgs)
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change threshold mask
        img_data.threshold_mask = None

        imgs = np.copy(imgs_orig)[y:y+h, x:x+w]  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = imgs
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change crop
        img_data.crop_area = None

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = np.copy(imgs)
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change background
        bkg = 0
        img_data.background = bkg

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = np.copy(imgs)
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

    def test_trainresolved_ma(self):
        """Test the case with moving average of image."""
        ImageData.reset()  # clear the data from test_trainresolved

        imgs_orig = np.arange(16, dtype=np.float).reshape(4, 4)

        img_data = ImageData(np.copy(imgs_orig))
        ImageData.set_moving_average_window(3)
        img_data = ImageData(imgs_orig - 2)
        self.assertEqual(2, img_data.moving_average_count)
        np.testing.assert_array_equal(imgs_orig - 1, img_data.masked_mean)
        img_data = ImageData(imgs_orig + 2)
        self.assertEqual(3, img_data.moving_average_count)
        np.testing.assert_array_equal(imgs_orig, img_data.masked_mean)
        img_data = ImageData(np.copy(imgs_orig))
        self.assertEqual(3, img_data.moving_average_count)
        np.testing.assert_array_equal(imgs_orig, img_data.masked_mean)

        # with background
        img_data = ImageData(np.copy(imgs_orig), background=1)
        np.testing.assert_array_equal(imgs_orig - 1, img_data.masked_mean)

        img_data.background = 2
        np.testing.assert_array_equal(imgs_orig - 2, img_data.masked_mean)

        ImageData.set_moving_average_window(4)
        img_data = ImageData(imgs_orig - 4, background=1)
        self.assertEqual(4, img_data.moving_average_count)
        np.testing.assert_array_equal(imgs_orig - 2, img_data.masked_mean)

        # the moving average implementation does not affect the cropping
        # and masking implementation which was first done without moving
        # average

    def test_pulseresolved(self):
        imgs_orig = np.arange(32, dtype=np.float).reshape((2, 4, 4))
        img_data = ImageData(np.copy(imgs_orig))
        img_data.threshold_mask = (1, 4)
        bkg = 1.0
        crop_area = (3, 2, 0, 1)
        img_data.background = bkg
        img_data.crop_area = crop_area

        # calculate the ground truth
        w, h, x, y = crop_area
        imgs = np.copy(imgs_orig)[:, y:y + h, x:x + w]
        imgs -= bkg

        self.assertEqual(2, img_data.n_images)

        np.testing.assert_array_equal(imgs, img_data.images)

        np.testing.assert_array_equal(imgs.mean(axis=0), img_data.mean)

        # test threshold mask
        masked_imgs = imgs.mean(axis=0)
        masked_imgs[(masked_imgs < 1)] = 1.0
        masked_imgs[(masked_imgs > 4)] = 4.0
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change threshold mask
        mask = (2, 12)
        img_data.threshold_mask = mask

        imgs = np.copy(imgs_orig)[:, y:y+h, x:x+w]  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = np.copy(imgs.mean(axis=0))
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change crop
        img_data.crop_area = None

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = imgs.mean(axis=0)
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change background
        bkg = 0
        img_data.background = bkg

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = imgs.mean(axis=0)
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

    def test_pulseresolved_ma(self):
        ImageData.reset()

        imgs_orig = np.arange(32, dtype=np.float).reshape((2, 4, 4))
        mean_orig = np.mean(imgs_orig, axis=0)

        img_data = ImageData(np.copy(imgs_orig[0, ...]))
        img_data = ImageData(np.copy(imgs_orig))
        # test automatic reset if the new images have different shape
        self.assertEqual(1, img_data.moving_average_count)
        self.assertEqual(1, img_data.moving_average_count)

        ImageData.set_moving_average_window(3)
        img_data = ImageData(imgs_orig - 2)
        self.assertEqual(2, img_data.moving_average_count)
        np.testing.assert_array_equal(mean_orig - 1, img_data.masked_mean)
        img_data = ImageData(imgs_orig + 2)
        self.assertEqual(3, img_data.moving_average_count)
        np.testing.assert_array_equal(mean_orig, img_data.masked_mean)
        img_data = ImageData(np.copy(imgs_orig))
        self.assertEqual(3, img_data.moving_average_count)
        np.testing.assert_array_equal(mean_orig, img_data.masked_mean)

        # with background
        img_data = ImageData(np.copy(imgs_orig), background=1)
        np.testing.assert_array_equal(mean_orig - 1, img_data.masked_mean)

        img_data.background = 2
        np.testing.assert_array_equal(mean_orig - 2, img_data.masked_mean)

        ImageData.set_moving_average_window(4)
        img_data = ImageData(imgs_orig - 4, background=1)
        self.assertEqual(4, img_data.moving_average_count)
        np.testing.assert_array_equal(mean_orig - 2, img_data.masked_mean)


class TestTrainData(unittest.TestCase):
    def test_general(self):
        class Dummy(AbstractData):
            values = TrainData()

        dm = Dummy()

        dm.values = (1, 'a')
        dm.values = (2, 'b')
        tids, values, _ = dm.values
        self.assertListEqual([1, 2], tids)
        self.assertListEqual(['a', 'b'], values)

        dm.values = (3, 'c')
        tids, values, _ = dm.values
        self.assertListEqual([1, 2, 3], tids)
        self.assertListEqual(['a', 'b', 'c'], values)

        del dm.values
        tids, values, _ = dm.values
        self.assertListEqual([2, 3], tids)
        self.assertListEqual(['b', 'c'], values)

        Dummy.clear()
        tids, values, _ = dm.values
        self.assertListEqual([], tids)
        self.assertListEqual([], values)


class TestProcessedData(unittest.TestCase):
    def test_general(self):
        data = ProcessedData(1234)
        self.assertEqual(1234, data.tid)

        data.roi.values1 = (1234, None)
        tids, values, _ = data.roi.values1
        self.assertListEqual([1234], tids)
        self.assertListEqual([None], values)

        data.roi.values1 = (1235, 2.0)
        tids, values, _ = data.roi.values1
        self.assertListEqual([1234, 1235], tids)
        self.assertListEqual([None, 2.0], values)

    def test_CorrelationData(self):
        data = ProcessedData(-1)

        data.add_correlator(0, "device1", "property1")
        data.correlation.param0 = (10, 20)
        data.correlation.param0 = (11, 22)
        fom, corr, info = data.correlation.param0
        self.assertListEqual([10, 11], fom)
        self.assertListEqual([20, 22], corr)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])

        data.add_correlator(1, "device2", "property2")
        data.correlation.param1 = (100, 200)
        data.correlation.param1 = (110, 220)
        fom, corr, info = data.correlation.param1
        self.assertListEqual([100, 110], fom)
        self.assertListEqual([200, 220], corr)
        self.assertEqual("device2", info["device_id"])
        self.assertEqual("property2", info["property"])
        # check that param0 remains unchanged
        fom, corr, info = data.correlation.param0
        self.assertListEqual([10, 11], fom)
        self.assertListEqual([20, 22], corr)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])

        # test clear history
        ProcessedData.clear_correlation_hist()
        fom, corr, info = data.correlation.param0
        self.assertListEqual([], fom)
        self.assertListEqual([], corr)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])
        fom, corr, info = data.correlation.param1
        self.assertListEqual([], fom)
        self.assertListEqual([], corr)
        self.assertEqual("device2", info["device_id"])
        self.assertEqual("property2", info["property"])

        # when device_id or property is empty, the corresponding 'param'
        # will be removed
        data.add_correlator(0, "", "property2")
        with self.assertRaises(AttributeError):
            data.correlation.param0

        data.add_correlator(1, "device2", "")
        with self.assertRaises(AttributeError):
            data.correlation.param1
