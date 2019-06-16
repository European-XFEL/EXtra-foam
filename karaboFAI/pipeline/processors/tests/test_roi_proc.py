import unittest
import numpy as np

from karaboFAI.config import config, RoiFom
from karaboFAI.pipeline.data_model import ProcessedData, RoiData
from karaboFAI.pipeline.processors.roi import RoiProcessor


class TestRoiProcessor(unittest.TestCase):
    def setUp(self):
        RoiData.clear()

        self._proc = RoiProcessor()
        self._proc.roi_fom_handler = np.sum
        self._rois = self._proc.regions

    def testProcessRoiData(self):
        processed = ProcessedData(1, np.ones((100, 100)))
        data = {'processed': processed}

        # ROIs are all None
        self.assertEqual(len(config["ROI_COLORS"]), len(self._rois))
        self.assertEqual(None, self._rois[0])
        self._proc.run_once(data)

        # Set ROI1 and ROI4
        self._proc.visibilities[0] = True
        self._proc.regions[0] = [0, 0, 2, 2]
        self._proc.visibilities[3] = True
        self._proc.regions[3] = [1, 1, 3, 3]

        # FOM of ROI is None
        self.assertEqual(None, self._proc.fom_type)
        self._proc.fom_type = RoiFom.MEAN

        # set the first history data
        self._proc.run_once(data)
        self.assertEqual(processed.roi.roi1, self._rois[0])
        self.assertEqual(processed.roi.roi2, self._rois[1])
        self.assertEqual(processed.roi.roi3, self._rois[2])
        self.assertEqual(processed.roi.roi4, self._rois[3])

        # FOM of roi is SUM
        self._proc.fom_type = RoiFom.SUM
        # set the second history data
        self._proc.run_once(data)

        self.assertEqual(4, processed.roi.roi1_fom)
        self.assertEqual(None, processed.roi.roi2_fom)
        self.assertEqual(None, processed.roi.roi3_fom)
        self.assertEqual(9, processed.roi.roi4_fom)
