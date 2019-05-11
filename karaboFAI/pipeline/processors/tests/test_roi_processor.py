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

    def setUp(self):
        RoiData.clear()
        ImageData.clear()

        self._proc = RoiProcessor()
        self._proc.roi_fom_handler = np.sum
        self._rois = self._proc.regions
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
        self._proc.visibilities[0] = True
        self._proc.regions[0] = [0, 0, 2, 2]
        self._proc.visibilities[3] = True
        self._proc.regions[3] = [1, 1, 3, 3]

        # FOM of ROI is None
        self.assertEqual(None, self._proc.fom_type)
        self._proc.fom_type = RoiFom.MEAN

        # set the first history data
        self._proc.run_once(self._proc_data)
        self.assertEqual(self._proc_data.roi.roi1, self._rois[0])
        self.assertEqual(self._proc_data.roi.roi2, self._rois[1])
        self.assertEqual(self._proc_data.roi.roi3, self._rois[2])
        self.assertEqual(self._proc_data.roi.roi4, self._rois[3])

        # FOM of roi is SUM
        self._proc.fom_type = RoiFom.SUM
        # set the second history data
        self._proc.run_once(self._proc_data)

        self.assertEqual(4, self._proc_data.roi.roi1_fom)
        self.assertEqual(None, self._proc_data.roi.roi2_fom)
        self.assertEqual(None, self._proc_data.roi.roi3_fom)
        self.assertEqual(9, self._proc_data.roi.roi4_fom)
