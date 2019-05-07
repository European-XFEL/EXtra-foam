import unittest
import numpy as np

from karaboFAI.config import config, RoiFom
from karaboFAI.pipeline.data_model import ProcessedData, ImageData, RoiData
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.pipeline.processors.roi import RoiProcessor


class TestRoiProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._count = 0
        config["PIXEL_SIZE"] = 1e-6
        config["MASK_RANGE"] = (None, None)

    def setUp(self):
        RoiData.clear()
        ImageData.clear()

        self._proc = RoiProcessor()
        self._rois = self._proc._raw_rois
        img = np.ones((100, 100))
        self.__class__._count += 1
        self._proc_data = ProcessedData(self._count, images=img)

    def testProcessRoiData(self):
        count = self.__class__._count

        # ROIs are all None
        self.assertEqual(len(config["ROI_COLORS"]), len(self._rois))
        self.assertEqual(None, self._rois[0])
        self._proc.run_once(self._proc_data)

        # Set ROI1 and ROI4
        self._proc.set_roi(1, (2, 2, 0, 0))
        self._proc.set_roi(4, (3, 3, 1, 1))
        with self.assertRaises(IndexError):
            self._proc.set_roi(5, None)

        # FOM of ROI is None
        self.assertEqual(None, self._proc.fom_type)
        self._proc.fom_type = RoiFom.MEAN

        # set the first history data
        self._proc.run_once(self._proc_data)
        self.assertEqual(self._proc_data.roi.roi1, self._rois[0])
        self.assertEqual(self._proc_data.roi.roi2, self._rois[1])
        self.assertEqual(self._proc_data.roi.roi3, self._rois[2])
        self.assertEqual(self._proc_data.roi.roi4, self._rois[3])

        tid, value, _ = self._proc_data.roi.roi1_hist
        np.testing.assert_array_equal([count], tid)
        np.testing.assert_array_equal([1], value)

        # FOM of roi is SUM
        self._proc.fom_type = RoiFom.SUM
        # set the second history data
        self._proc.run_once(self._proc_data)
        tid, value, _ = self._proc_data.roi.roi1_hist
        np.testing.assert_array_equal([1, 4], value)
        tid, value, _ = self._proc_data.roi.roi2_hist
        np.testing.assert_array_equal([0, 0], value)
        tid, value, _ = self._proc_data.roi.roi3_hist
        np.testing.assert_array_equal([0, 0], value)
        tid, value, _ = self._proc_data.roi.roi4_hist
        np.testing.assert_array_equal([1, 9], value)

    def testRoiPumpProbeFomProcessor(self):
        pass