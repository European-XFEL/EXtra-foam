import unittest
import numpy as np

from karaboFAI.config import config, RoiFom
from karaboFAI.pipeline.data_model import ProcessedData, ImageData, RoiData
from karaboFAI.pipeline.data_processor import RoiProcessor


class TestCorrelationProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._count = 0
        config["PIXEL_SIZE"] = 1e-6
        config["MASK_RANGE"] = (None, None)

    def setUp(self):
        RoiData.clear()
        ImageData.clear()

        self._proc = RoiProcessor()
        self._rois = self._proc._rois
        img = np.ones((100, 100))
        self.__class__._count += 1
        self._proc_data = ProcessedData(self._count, images=img)

    def testProcessRData(self):
        count = self.__class__._count

        # ROIs are all None
        self.assertEqual(len(config["ROI_COLORS"]), len(self._proc._rois))
        self.assertEqual(None, self._rois[0])
        self._proc.process(self._proc_data)

        # Set ROI1 and ROI4
        self._proc.set(1, (2, 2, 0, 0))
        self._proc.set(4, (3, 3, 1, 1))
        with self.assertRaises(IndexError):
            self._proc.set(5, None)

        # FOM of roi is None
        self.assertEqual(None, self._proc.roi_fom)
        self._proc.process(self._proc_data)
        self.assertEqual(self._proc_data.roi.roi1, self._rois[0])
        self.assertEqual(self._proc_data.roi.roi2, self._rois[1])
        self.assertEqual(self._proc_data.roi.roi3, self._rois[2])
        self.assertEqual(self._proc_data.roi.roi4, self._rois[3])
        tid, value, _ = self._proc_data.roi.roi1_hist
        self.assertListEqual([count] * 2, tid)
        self.assertListEqual([0] * 2, value)

        # FOM of roi is SUM
        self._proc.roi_fom = RoiFom.SUM
        self._proc.process(self._proc_data)
        tid, value, _ = self._proc_data.roi.roi1_hist
        self.assertListEqual([0, 0, 4], value)
        tid, value, _ = self._proc_data.roi.roi2_hist
        self.assertListEqual([0, 0, 0], value)
        tid, value, _ = self._proc_data.roi.roi3_hist
        self.assertListEqual([0, 0, 0], value)
        tid, value, _ = self._proc_data.roi.roi4_hist
        self.assertListEqual([0, 0, 9], value)

        # reference image
        tid, value, _ = self._proc_data.roi.roi4_hist_ref
        self.assertListEqual([0, 0, 0], value)

        # set a reference image
        self._proc_data._image_data.set_reference()
        self._proc_data._image_data.update()
        self._proc.process(self._proc_data)

        tid, value, _ = self._proc_data.roi.roi1_hist_ref
        self.assertListEqual([0, 0, 0, 4], value)
        tid, value, _ = self._proc_data.roi.roi2_hist_ref
        self.assertListEqual([0, 0, 0, 0], value)
        tid, value, _ = self._proc_data.roi.roi3_hist_ref
        self.assertListEqual([0, 0, 0, 0], value)
        tid, value, _ = self._proc_data.roi.roi4_hist_ref
        self.assertListEqual([0, 0, 0, 9], value)
