import unittest

from karaboFAI.pipeline.processors import AzimuthalIntegrationProcessor


class TestAzimuthalIntegrationProcessor(unittest.TestCase):
    def setUp(self):
        self._proc = AzimuthalIntegrationProcessor()

        self._proc.sample_distance = 0.2
        self._proc.wavelength = 1.0e-10
        self._proc.integration_center = (100, 100)
        self._proc.integration_method = None
        self._proc.integration_range = None
        self._proc.integration_points = None
        self._proc.normalizer = None
        self._proc.auc_x_range = (1, 5)
        self._proc.fom_itgt_range = (1, 5)
        self._proc.moving_avg_window = 100
