import unittest

import numpy as np

from karaboFAI.pipeline.data_model import (
    AbstractData, AccumulatedPairData, ImageData,
    DataManager, ProcessedData, PumpProbeData, RoiData, PairData
)
from karaboFAI.logger import logger
from karaboFAI.config import config, ImageMaskChange


class TestRoiData(unittest.TestCase):
    def test_general(self):
        data = RoiData()

        n_rois = len(config["ROI_COLORS"])
        for i in range(1, n_rois+1):
            self.assertTrue(hasattr(RoiData, f"roi{i}_hist"))
            self.assertIsInstance(getattr(RoiData, f"roi{i}_hist"), PairData)
            self.assertTrue(hasattr(data, f"roi{i}"))

        with self.assertRaises(AttributeError):
            getattr(RoiData, f"roi{n_rois+1}_hist")

        with self.assertRaises(AttributeError):
            getattr(data, f"roi{n_rois+1}")


class TestImageData(unittest.TestCase):
    def setUp(self):
        ImageData.clear()

    def test_memoryTrainResolved(self):
        imgs_orig = np.arange(100, dtype=np.float).reshape(10, 10)
        img_data = ImageData(imgs_orig)
        img_data.set_reference()
        img_data.update()

        imgs = img_data.images
        img_mean = img_data.mean
        masked_mean = img_data.masked_mean
        ref = img_data.ref
        np.testing.assert_array_equal(imgs, img_mean)
        imgs[0, 0] = 123456
        self.assertEqual(img_mean[0, 0], imgs[0, 0])
        # masked_mean and mean do not share memory space
        self.assertNotEqual(masked_mean[0, 0], img_mean[0, 0])
        self.assertNotEqual(ref[0, 0], img_mean[0, 0])

    def test_memoryPulseResolved(self):
        imgs_orig = np.arange(100, dtype=np.float).reshape(4, 5, 5)
        img_data = ImageData(imgs_orig)
        img_data.set_reference()
        img_data.update()

        img_mean = img_data.mean
        masked_mean = img_data.masked_mean
        ref = img_data.ref
        np.testing.assert_array_equal(img_mean, ref)
        img_mean[0, 0] = 123456
        # Note: this is different from train-resolved data!!!
        # masked_mean and mean do not share memory space
        self.assertNotEqual(masked_mean[0, 0], img_mean[0, 0])
        self.assertNotEqual(ref[0, 0], img_mean[0, 0])

    def test_invalidInput(self):
        with self.assertRaises(TypeError):
            ImageData([1, 2, 3])

        with self.assertRaises(ValueError):
            ImageData(np.arange(2))

        with self.assertRaises(ValueError):
            ImageData(np.arange(16).reshape(2, 2, 2, 2))

    def test_referenceTrainResolved(self):
        imgs_orig = np.arange(25, dtype=np.float).reshape(5, 5)

        img_data = ImageData(np.copy(imgs_orig))
        img_data.set_threshold_mask(5, 20)
        img_data.set_reference()
        img_data.update()

        masked_gt = np.copy(imgs_orig)
        masked_gt[masked_gt < 5] = 5
        masked_gt[masked_gt > 20] = 20
        np.testing.assert_array_equal(imgs_orig, img_data.ref)
        np.testing.assert_array_equal(masked_gt, img_data.masked_ref)

        # test new instance
        img_data.set_threshold_mask(20, None)
        img_data = ImageData(np.copy(imgs_orig))
        masked_gt = np.copy(imgs_orig)
        masked_gt[masked_gt < 20] = 20
        np.testing.assert_array_equal(imgs_orig, img_data.ref)
        np.testing.assert_array_equal(masked_gt, img_data.masked_ref)

        # test remove reference
        img_data.remove_reference()
        img_data.update()
        self.assertEqual(None, img_data.ref)
        self.assertEqual(None, img_data.masked_ref)

    def test_referencePulseResolved(self):
        imgs_orig = np.arange(32, dtype=np.float).reshape(2, 4, 4)
        imgs_mean = np.mean(imgs_orig, axis=0)

        img_data = ImageData(np.copy(imgs_orig))
        img_data.set_threshold_mask(5, 20)
        img_data.set_reference()
        img_data.update()

        masked_gt = np.copy(imgs_mean)
        masked_gt[masked_gt < 5] = 5
        masked_gt[masked_gt > 20] = 20
        np.testing.assert_array_equal(imgs_mean, img_data.ref)
        np.testing.assert_array_equal(masked_gt, img_data.masked_ref)

        # test new instance
        img_data.set_threshold_mask(20, None)
        img_data = ImageData(np.copy(imgs_orig))
        masked_gt = np.copy(imgs_mean)
        masked_gt[masked_gt < 20] = 20
        np.testing.assert_array_equal(imgs_mean, img_data.ref)
        np.testing.assert_array_equal(masked_gt, img_data.masked_ref)

        # test remove reference
        img_data.remove_reference()
        img_data.update()
        self.assertEqual(None, img_data.ref)
        self.assertEqual(None, img_data.masked_ref)

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

    def test_thresholdmask(self):
        imgs_orig = np.arange(25, dtype=np.float).reshape(5, 5)
        img_data = ImageData(np.copy(imgs_orig))
        img_data.set_reference()
        img_data.update()

        self.assertTupleEqual((-1e5, 1e5), img_data.threshold_mask)
        np.testing.assert_array_equal(imgs_orig, img_data.masked_mean)
        np.testing.assert_array_equal(imgs_orig, img_data.masked_ref)

        img_data.set_threshold_mask(None, 5)
        img_data.update()
        self.assertTupleEqual((-np.inf, 5), img_data.threshold_mask)
        # test the change can be seen by the new instance
        self.assertTupleEqual((-np.inf, 5),
                              ImageData(imgs_orig).threshold_mask)
        imgs = np.copy(imgs_orig)
        imgs[imgs > 5] = 5
        np.testing.assert_array_equal(imgs, img_data.masked_mean)
        np.testing.assert_array_equal(imgs, img_data.masked_ref)

        img_data.set_threshold_mask(1, None)
        img_data.update()
        self.assertTupleEqual((1, np.inf), img_data.threshold_mask)
        imgs = np.copy(imgs_orig)
        imgs[imgs < 1] = 1
        np.testing.assert_array_equal(imgs, img_data.masked_mean)
        np.testing.assert_array_equal(imgs, img_data.masked_ref)

        img_data.set_threshold_mask(3, 8)
        img_data.update()
        self.assertTupleEqual((3, 8), img_data.threshold_mask)
        imgs = np.copy(imgs_orig)
        imgs[imgs < 3] = 3
        imgs[imgs > 8] = 8
        np.testing.assert_array_equal(imgs, img_data.masked_mean)
        np.testing.assert_array_equal(imgs, img_data.masked_ref)

    def test_trainresolved(self):
        imgs_orig = np.arange(16, dtype=np.float).reshape(4, 4)

        img_data = ImageData(np.copy(imgs_orig))
        self.assertFalse(img_data.pulse_resolved())
        mask = (1, 4)
        img_data.set_threshold_mask(*mask)
        bkg = 1.0
        img_data.set_background(bkg)
        img_data.update()

        self.assertEqual(imgs_orig.shape, img_data.shape)
        self.assertEqual(bkg, img_data.background)
        # test the change can be seen by the new instance
        self.assertEqual(bkg, ImageData(imgs_orig).background)
        self.assertEqual(1, img_data.n_images)

        # calculate the ground truth
        imgs = np.copy(imgs_orig)
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

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = imgs
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

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

        # the moving average implementation does not affect the masking
        # implementation which was first done without moving average

    def test_pulseresolved(self):
        imgs_orig = np.arange(32, dtype=np.float).reshape((2, 4, 4))
        img_data = ImageData(np.copy(imgs_orig))
        self.assertTrue(img_data.pulse_resolved())

        img_data.set_threshold_mask(1, 4)
        bkg = 1.0
        img_data.set_background(bkg)
        img_data.update()

        self.assertEqual(imgs_orig.shape[1:], img_data.shape)
        self.assertEqual(imgs_orig.shape[0], img_data.n_images)

        # calculate the ground truth
        imgs = np.copy(imgs_orig)
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

        imgs = np.copy(imgs_orig)  # recalculate the ground truth
        imgs -= bkg
        masked_imgs = np.copy(imgs.mean(axis=0))
        masked_imgs[(masked_imgs < mask[0])] = mask[0]
        masked_imgs[(masked_imgs > mask[1])] = mask[1]
        np.testing.assert_array_equal(masked_imgs, img_data.masked_mean)

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

        ImageData.clear()
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

    def test_pulseResolvedSliceMaskedMean(self):
        imgs_orig = np.arange(32, dtype=np.float).reshape((2, 4, 4))
        mean_orig = np.mean(imgs_orig, axis=0)
        img_data = ImageData(np.copy(imgs_orig))
        mask_range = (1, 4)
        img_data.set_threshold_mask(*mask_range)
        img_data.update()

        with self.assertRaises(IndexError):
            img_data.sliced_masked_mean([1, 3])

        np.testing.assert_array_equal(np.clip(mean_orig, *mask_range),
                                      img_data.sliced_masked_mean([0, 1]))
        np.testing.assert_array_equal(np.clip(imgs_orig[0], *mask_range),
                                      img_data.sliced_masked_mean([0]))

    def test_trainResolvedSliceMaskedMean(self):
        imgs_orig = np.arange(16, dtype=np.float).reshape((4, 4))
        img_data = ImageData(np.copy(imgs_orig))
        mask_range = (1, 4)
        img_data.set_threshold_mask(*mask_range)
        img_data.update()

        with self.assertRaises(IndexError):
            img_data.sliced_masked_mean([1, 2])
        with self.assertRaises(IndexError):
            img_data.sliced_masked_mean([1])

        np.testing.assert_array_equal(np.clip(imgs_orig, *mask_range),
                                      img_data.sliced_masked_mean([0]))


class TestPairData(unittest.TestCase):
    def test_general(self):
        class Dummy(AbstractData):
            values = PairData()

        dm = Dummy()

        dm.values = (1, 10)
        dm.values = (2, 20)
        tids, values, _ = dm.values
        np.testing.assert_array_equal([1, 2], tids)
        np.testing.assert_array_equal([10, 20], values)

        dm.values = (3, 30)
        tids, values, _ = dm.values
        np.testing.assert_array_equal([1, 2, 3], tids)
        np.testing.assert_array_equal([10, 20, 30], values)

        del dm.values
        tids, values, _ = dm.values
        np.testing.assert_array_equal([2, 3], tids)
        np.testing.assert_array_equal([20, 30], values)

        Dummy.clear()
        tids, values, _ = dm.values
        np.testing.assert_array_equal([], tids)
        np.testing.assert_array_equal([], values)


class TestCorrelationData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._manager = DataManager()

    def setUp(self):
        self._manager.reset_correlation()
        self._manager.remove_correlations()

    def testPairData(self):
        data = ProcessedData(1)
        with self.assertRaises(ValueError):
            self._manager.add_correlation(0, "device1", "property1")

        self._manager.add_correlation(1, "device1", "property1")
        data.correlation.correlation1 = (1, 0.5)
        data.correlation.correlation1 = (2, 0.6)
        corr_hist, fom_hist, info = data.correlation.correlation1
        np.testing.assert_array_almost_equal([1, 2], corr_hist)
        np.testing.assert_array_almost_equal([0.5, 0.6], fom_hist)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])

        self._manager.add_correlation(2, "device2", "property2")
        data.correlation.correlation2 = (3, 200)
        data.correlation.correlation2 = (4, 220)
        corr_hist, fom_hist, info = data.correlation.correlation2
        np.testing.assert_array_almost_equal([3, 4], corr_hist)
        np.testing.assert_array_almost_equal([200, 220], fom_hist)
        self.assertEqual("device2", info["device_id"])
        self.assertEqual("property2", info["property"])
        # check that correlation1 remains unchanged
        corr_hist, fom_hist, info = data.correlation.correlation1
        np.testing.assert_array_almost_equal([1, 2], corr_hist)
        np.testing.assert_array_almost_equal([0.5, 0.6], fom_hist)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])

        # test clear history
        self._manager.reset_correlation()
        corr_hist, fom_hist, info = data.correlation.correlation1
        np.testing.assert_array_almost_equal([], corr_hist)
        np.testing.assert_array_almost_equal([], fom_hist)
        self.assertEqual("device1", info["device_id"])
        self.assertEqual("property1", info["property"])
        corr_hist, fom_hist, info = data.correlation.correlation2
        np.testing.assert_array_almost_equal([], corr_hist)
        np.testing.assert_array_almost_equal([], fom_hist)
        self.assertEqual("device2", info["device_id"])
        self.assertEqual("property2", info["property"])

        # when device_id or property is empty, the corresponding 'param'
        # will be removed
        self._manager.add_correlation(1, "", "property2")
        with self.assertRaises(AttributeError):
            data.correlation.correlation1

        self._manager.add_correlation(2, "device2", "")
        with self.assertRaises(AttributeError):
            data.correlation.correlation2

        # test CorrelationData.remove_correlations()
        self._manager.add_correlation(1, "device1", "property1")
        self._manager.add_correlation(2, "device1", "property1")
        self.assertListEqual(['correlation1', 'correlation2'],
                             self._manager.get_correlations())
        self._manager.remove_correlations()
        self.assertListEqual([], self._manager.get_correlations())

        # test when resolution becomes non-zero
        self._manager.add_correlation(1, "device1", "property1", 0.2)
        self.assertIsInstance(data.correlation.__class__.__dict__['correlation1'],
                              AccumulatedPairData)

        # ----------------------------
        # test when max length reached
        # ----------------------------

        self._manager.add_correlation(1, "device1", "property1")
        # override the class attribute
        max_len = 1000
        data.correlation.__class__.__dict__['correlation1'].MAX_LENGTH = max_len
        overflow = 10
        for i in range(max_len + overflow):
            data.correlation.correlation1 = (i, i)
        corr, fom, _ = data.correlation.correlation1
        self.assertEqual(max_len, len(corr))
        self.assertEqual(max_len, len(fom))
        self.assertEqual(overflow, corr[0])
        self.assertEqual(overflow, fom[0])
        self.assertEqual(max_len + overflow - 1, corr[-1])
        self.assertEqual(max_len + overflow - 1, fom[-1])

    def testAccumulatedPairData(self):
        data = ProcessedData(1)
        self.assertEqual(2, AccumulatedPairData._min_count)

        self._manager.add_correlation(1, "device1", "property1", 0.1)
        data.correlation.correlation1 = (1, 0.3)
        data.correlation.correlation1 = (2, 0.4)
        corr_hist, fom_hist, info = data.correlation.correlation1
        np.testing.assert_array_equal([], corr_hist)
        np.testing.assert_array_equal([], fom_hist.count)
        np.testing.assert_array_equal([], fom_hist.avg)
        np.testing.assert_array_equal([], fom_hist.min)
        np.testing.assert_array_equal([], fom_hist.max)

        data.correlation.correlation1 = (2.02, 0.5)
        corr_hist, fom_hist, info = data.correlation.correlation1
        np.testing.assert_array_equal([2.01], corr_hist)
        np.testing.assert_array_equal([2], fom_hist.count)
        np.testing.assert_array_almost_equal([0.425], fom_hist.min)
        np.testing.assert_array_almost_equal([0.475], fom_hist.max)
        np.testing.assert_array_equal([0.45], fom_hist.avg)

        data.correlation.correlation1 = (2.11, 0.6)
        corr_hist, fom_hist, info = data.correlation.correlation1
        np.testing.assert_array_equal([3], fom_hist.count)
        np.testing.assert_array_almost_equal([0.4591751709536137], fom_hist.min)
        np.testing.assert_array_almost_equal([0.5408248290463863], fom_hist.max)
        np.testing.assert_array_equal([0.5], fom_hist.avg)

        # new point
        data.correlation.correlation1 = (2.31, 1)
        data.correlation.correlation1 = (2.41, 2)
        corr_hist, fom_hist, info = data.correlation.correlation1
        np.testing.assert_array_equal([3, 2], fom_hist.count)
        np.testing.assert_array_almost_equal([0.4591751709536137, 1.25], fom_hist.min)
        np.testing.assert_array_almost_equal([0.5408248290463863, 1.75], fom_hist.max)
        np.testing.assert_array_equal([0.5, 1.5], fom_hist.avg)

        # test when resolution changes
        self._manager.add_correlation(1, "device1", "property1", 0.2)
        corr_hist, fom_hist, info = data.correlation.correlation1
        np.testing.assert_array_equal([], corr_hist)
        np.testing.assert_array_equal([], fom_hist.count)
        np.testing.assert_array_equal([], fom_hist.min)
        np.testing.assert_array_equal([], fom_hist.max)
        np.testing.assert_array_equal([], fom_hist.avg)

        # test when resolution becomes 0
        self._manager.add_correlation(1, "device1", "property1")
        self.assertIsInstance(data.correlation.__class__.__dict__['correlation1'],
                              PairData)

        # ----------------------------
        # test when max length reached
        # ----------------------------

        self._manager.add_correlation(1, "device1", "property1", 1.0)
        # override the class attribute
        max_len = 1000
        data.correlation.__class__.__dict__['correlation1'].MAX_LENGTH = max_len
        overflow = 10
        for i in range(2*max_len + 2*overflow):
            # two adjacent data point will be grouped together since
            # resolution is 1.0
            data.correlation.correlation1 = (i, i)
        corr_hist, fom_hist, _ = data.correlation.correlation1
        self.assertEqual(max_len, len(corr_hist))
        self.assertEqual(max_len, len(fom_hist.avg))
        self.assertEqual(2*overflow + 0.5, corr_hist[0])
        self.assertEqual(2*overflow + 0.5, fom_hist.avg[0])
        self.assertEqual(2*(max_len + overflow - 1) + 0.5, corr_hist[-1])
        self.assertEqual(2*(max_len + overflow - 1) + 0.5, fom_hist.avg[-1])


class TestProcessedData(unittest.TestCase):
    def setUp(self):
        DataManager().reset_roi()

    def testGeneral(self):
        data = ProcessedData(1234)
        self.assertEqual(1234, data.tid)

        data.roi.roi1_hist = (1234, None)
        tids, roi1_hist, _ = data.roi.roi1_hist
        np.testing.assert_array_equal([1234], tids)
        np.testing.assert_array_equal([None], roi1_hist)

        data.roi.roi1_hist = (1235, 2.0)
        tids, roi1_hist, _ = data.roi.roi1_hist
        np.testing.assert_array_equal([1234, 1235], tids)
        np.testing.assert_array_equal([None, 2.0], roi1_hist)


class TestPumpProbeData(unittest.TestCase):
    def setUp(self):
        DataManager().reset_pp()

    def testGeneral(self):
        data = PumpProbeData()

        self.assertEqual(1, data.ma_window)
        self.assertEqual(0, data.ma_count)

        window = 6

        data.ma_window = window
        self.assertEqual(window, data.ma_window)

        x_gt = np.ones(10)  # x should not change
        this_on = 15 * np.ones(10)
        this_off = 10 * np.ones(10)
        on_gt = np.copy(this_on)
        off_gt = np.copy(this_off)

        # test clear
        data.data = (x_gt, this_on, this_off)
        PumpProbeData.clear()
        self.assertEqual(1, data.ma_window)
        self.assertEqual(0, data.ma_count)

        # set data again
        data.ma_window = window
        data.data = (x_gt, this_on, this_off)
        # count < window size
        for i in range(data.ma_window - 1):
            data.data = (x_gt, this_on - 2 - i, this_off - 2 - i)
            x, on, off = data.data
            on_gt -= 1
            off_gt -= 1
            np.testing.assert_array_equal(x_gt, x)
            np.testing.assert_array_equal(on_gt, on)
            np.testing.assert_array_equal(off_gt, off)
        self.assertEqual(window, data.ma_count)

        # on = 10 * np.ones(1), off = 5 * np.ones(1)
        x, on, off = data.data

        this_on = 4 * np.ones(10)
        this_off = 2 * np.ones(10)
        data.data = (x_gt, this_on, this_off)
        self.assertEqual(window, data.ma_count)
        x, on, off = data.data
        np.testing.assert_array_equal(9 * np.ones(10), on)  # 10 + (4 - 10)/6
        np.testing.assert_array_equal(4.5 * np.ones(10), off)  # 5 + (2 - 5)/6
