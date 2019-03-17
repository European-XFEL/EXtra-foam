import unittest

import numpy as np

from karaboFAI.pipeline.data_model import (
    AbstractData, ImageData, ProcessedData, TrainData
)
from karaboFAI.logger import logger
from karaboFAI.config import config, ImageMaskChange


class TestImageData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config["PIXEL_SIZE"] = 1e-6
        config["MASK_RANGE"] = (None, None)

    def setUp(self):
        ImageData.reset()

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

        img_data.set_crop_area(True, 1, 2, 6, 7)
        img_data.update()
        self.assertTupleEqual((7, 6), img_data.mean.shape)
        self.assertTupleEqual((1, 2), img_data.pos(0, 0))
        # test the change can be seen by the new instance
        self.assertTupleEqual((7, 6), ImageData(imgs).mean.shape)
        self.assertTupleEqual((1, 2), ImageData(imgs).pos(0, 0))

        img_data.set_crop_area(True, 0, 1, 3, 4)
        img_data.update()
        self.assertTupleEqual((4, 3), img_data.mean.shape)
        self.assertTupleEqual((0, 1), img_data.pos(0, 0))

    def test_poni(self):
        imgs = np.arange(20, dtype=np.float).reshape(5, 4)

        img_data = ImageData(np.copy(imgs))

        self.assertTupleEqual((0, 0), img_data.poni)
        img_data.set_crop_area(True, 0, 1, 3, 2)
        img_data.update()
        self.assertTupleEqual((-1, 0), img_data.poni)
        # test the change can be seen by the new instance
        self.assertTupleEqual((-1, 0), ImageData(imgs).poni)

        img_data.set_crop_area(True, 1, 2, 3, 2)
        img_data.update()
        img_data.poni = (-2, 12)
        self.assertTupleEqual((-4, 11), img_data.poni)

    def test_imagemask(self):
        imgs_orig = np.arange(25, dtype=np.float).reshape(5, 5)
        mask = np.zeros_like(imgs_orig, dtype=bool)

        img_data = ImageData(np.copy(imgs_orig))
        img_data.set_image_mask(ImageMaskChange.MASK, 1, 1, 2, 2)
        img_data.update()
        mask[1:3, 1:3] = True
        np.testing.assert_array_equal(mask, img_data.image_mask)
        # test the change can be seen by the new instance
        np.testing.assert_array_equal(mask, ImageData(imgs_orig).image_mask)

        img_data.set_image_mask(ImageMaskChange.UNMASK, 2, 2, 4, 4)
        img_data.update()
        mask[2:4, 2:4] = False
        np.testing.assert_array_equal(mask, img_data.image_mask)

        img_data.set_image_mask(ImageMaskChange.CLEAR, 0, 0, 0, 0)
        img_data.update()
        mask[:] = False
        np.testing.assert_array_equal(mask, img_data.image_mask)

        mask[3:, 3:] = True
        img_data.set_image_mask(ImageMaskChange.REPLACE, mask, 0, 0, 0)
        img_data.update()
        np.testing.assert_array_equal(mask, img_data.image_mask)

        # image mask changes as crop area changes
        img_data.set_crop_area(True, 1, 1, 3, 3)
        img_data.update()
        np.testing.assert_array_equal(mask[1:4, 1:4], img_data.image_mask)

    def test_thresholdmask(self):
        imgs_orig = np.arange(25, dtype=np.float).reshape(5, 5)
        img_data = ImageData(np.copy(imgs_orig))
        self.assertTupleEqual((-np.inf, np.inf), img_data.threshold_mask)
        np.testing.assert_array_equal(imgs_orig, img_data.masked_mean)

        img_data.set_threshold_mask(None, 5)
        img_data.update()
        self.assertTupleEqual((-np.inf, 5), img_data.threshold_mask)
        # test the change can be seen by the new instance
        self.assertTupleEqual((-np.inf, 5),
                              ImageData(imgs_orig).threshold_mask)
        imgs = np.copy(imgs_orig)
        imgs[imgs > 5] = 5
        np.testing.assert_array_equal(imgs, img_data.masked_mean)

        img_data.set_threshold_mask(1, None)
        img_data.update()
        self.assertTupleEqual((1, np.inf), img_data.threshold_mask)
        imgs = np.copy(imgs_orig)
        imgs[imgs < 1] = 1
        np.testing.assert_array_equal(imgs, img_data.masked_mean)

        img_data.set_threshold_mask(3, 8)
        img_data.update()
        self.assertTupleEqual((3, 8), img_data.threshold_mask)
        imgs = np.copy(imgs_orig)
        imgs[imgs < 3] = 3
        imgs[imgs > 8] = 8
        np.testing.assert_array_equal(imgs, img_data.masked_mean)

    def test_trainresolved(self):
        imgs_orig = np.arange(16, dtype=np.float).reshape(4, 4)

        img_data = ImageData(np.copy(imgs_orig))
        mask = (1, 4)
        img_data.set_threshold_mask(*mask)
        bkg = 1.0
        crop_area = (0, 1, 3, 2)
        img_data.set_background(bkg)
        img_data.set_crop_area(True, *crop_area)
        img_data.update()

        self.assertEqual(imgs_orig.shape, img_data.shape)
        self.assertEqual(bkg, img_data.background)
        # test the change can be seen by the new instance
        self.assertEqual(bkg, ImageData(imgs_orig).background)
        self.assertEqual(1, img_data.n_images)

        # calculate the ground truth
        x, y, w, h = crop_area
        imgs = np.copy(imgs_orig)[y:y+h, x:x+w]
        imgs -= bkg

        np.testing.assert_array_equal(imgs, img_data.images)
        np.testing.assert_array_equal(imgs, img_data.mean)

        # test threshold mask
        masked_imgs = np.copy(imgs)
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # clear threshold mask
        img_data.set_threshold_mask(None, None)
        img_data.update()

        imgs = np.copy(imgs_orig)[y:y+h, x:x+w]  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = imgs
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # clear crop
        img_data.set_crop_area(False, 0, 0, 0, 0)
        img_data.update()

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = np.copy(imgs)
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change background
        bkg = -1.0
        img_data.set_background(bkg)
        img_data.update()

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = np.copy(imgs)
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

    def test_trainresolved_ma(self):
        """Test the case with moving average of image."""
        imgs_orig = np.arange(16, dtype=np.float).reshape(4, 4)

        img_data = ImageData(np.copy(imgs_orig))
        img_data.set_ma_window(3)
        img_data = ImageData(imgs_orig - 2)
        self.assertEqual(2, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig - 1, img_data.masked_mean)
        img_data = ImageData(imgs_orig + 2)
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig, img_data.masked_mean)
        img_data = ImageData(np.copy(imgs_orig))
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig, img_data.masked_mean)

        # with background
        img_data = ImageData(np.copy(imgs_orig))
        img_data.set_background(1.0)
        img_data.update()
        np.testing.assert_array_equal(imgs_orig - 1, img_data.masked_mean)

        img_data.set_background(2.0)
        img_data.update()
        np.testing.assert_array_equal(imgs_orig - 2, img_data.masked_mean)

        # test moving average window size change
        img_data.set_ma_window(4)
        img_data = ImageData(imgs_orig - 4)
        img_data.set_background(1.0)
        img_data.update()
        self.assertEqual(4, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig - 2, img_data.masked_mean)

        img_data.set_ma_window(2)
        self.assertEqual(4, img_data.ma_window)
        self.assertEqual(4, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig - 2, img_data.masked_mean)
        img_data.update()
        # ma_window and ma_count should not be updated even if "update"
        # method is called
        self.assertEqual(4, img_data.ma_window)
        self.assertEqual(4, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig - 2, img_data.masked_mean)
        # we will see the change when new data is received
        img_data = ImageData(np.copy(imgs_orig))
        self.assertEqual(2, img_data.ma_window)
        self.assertEqual(1, img_data.ma_count)
        img_data.set_background(0)
        img_data.update()
        np.testing.assert_array_equal(imgs_orig, img_data.images)

        # the moving average implementation does not affect the cropping
        # and masking implementation which was first done without moving
        # average

    def test_pulseresolved(self):
        imgs_orig = np.arange(32, dtype=np.float).reshape((2, 4, 4))
        img_data = ImageData(np.copy(imgs_orig))

        img_data.set_threshold_mask(1, 4)
        bkg = 1.0
        crop_area = (0, 1, 3, 2)
        img_data.set_background(bkg)
        img_data.set_crop_area(True, *crop_area)
        img_data.update()

        self.assertEqual(imgs_orig.shape[1:], img_data.shape)
        self.assertEqual(imgs_orig.shape[0], img_data.n_images)

        # calculate the ground truth
        x, y, w, h = crop_area
        imgs = np.copy(imgs_orig)[:, y:y + h, x:x + w]
        imgs -= bkg

        np.testing.assert_array_equal(imgs, img_data.images)
        np.testing.assert_array_equal(imgs.mean(axis=0), img_data.mean)
        # test the change can be seen by the new instance
        np.testing.assert_array_equal(imgs, ImageData(imgs_orig).images)

        # test threshold mask
        masked_imgs = imgs.mean(axis=0)
        masked_imgs[(masked_imgs < 1)] = 1.0
        masked_imgs[(masked_imgs > 4)] = 4.0
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change threshold mask
        mask = (2, 12)
        img_data.set_threshold_mask(*mask)
        img_data.update()

        imgs = np.copy(imgs_orig)[:, y:y+h, x:x+w]  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = np.copy(imgs.mean(axis=0))
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # clear crop
        img_data.set_crop_area(False, 0, 0, 0, 0)
        img_data.update()

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = imgs.mean(axis=0)
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

        # change background
        bkg = 0
        img_data.set_background(bkg)
        img_data.update()

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = imgs.mean(axis=0)
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

    def test_pulseresolved_ma(self):
        imgs_orig = np.arange(32, dtype=np.float).reshape((2, 4, 4))
        mean_orig = np.mean(imgs_orig, axis=0)

        img_data = ImageData(np.copy(imgs_orig[0, ...]))
        img_data.set_ma_window(10)
        with self.assertLogs(logger, "ERROR"):
            ImageData(np.copy(imgs_orig))

        ImageData.reset()
        img_data = ImageData(np.copy(imgs_orig))
        img_data.set_ma_window(3)
        img_data = ImageData(imgs_orig - 2)
        self.assertEqual(2, img_data.ma_count)
        np.testing.assert_array_equal(mean_orig - 1, img_data.masked_mean)
        img_data = ImageData(imgs_orig + 2)
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(mean_orig, img_data.masked_mean)
        img_data = ImageData(np.copy(imgs_orig))
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(mean_orig, img_data.masked_mean)

        # with background
        img_data = ImageData(np.copy(imgs_orig))
        img_data.set_background(1)
        img_data.update()
        np.testing.assert_array_equal(mean_orig - 1, img_data.masked_mean)

        img_data.set_background(2)
        img_data.update()
        np.testing.assert_array_equal(mean_orig - 2, img_data.masked_mean)

        img_data.set_ma_window(4)
        img_data = ImageData(imgs_orig - 4)
        img_data.set_background(1)
        img_data.update()
        self.assertEqual(4, img_data.ma_count)
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
