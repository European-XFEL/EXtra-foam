import unittest

from karaboFAI.config import CorrelationFom
from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.processors import CorrelationProcessor
from karaboFAI.pipeline.exceptions import ProcessingError


class TestCorrelationProcessor(unittest.TestCase):
    def setUp(self):
        self._proc = CorrelationProcessor()

    def testRaise(self):
        for fom in CorrelationFom:
            self._proc.fom_type = fom
            if fom == CorrelationFom.PUMP_PROBE_FOM:
                with self.assertRaisesRegex(ProcessingError, "Pump-probe"):
                    self._proc.run_once(ProcessedData(1))
            elif fom == CorrelationFom.AZIMUTHAL_INTEG_MEAN:
                with self.assertRaisesRegex(ProcessingError, "Azimuthal integration"):
                    self._proc.run_once(ProcessedData(1))
            elif fom == CorrelationFom.UNDEFINED:
                pass
            else:
                with self.assertRaisesRegex(ProcessingError, "ROI"):
                    self._proc.run_once(ProcessedData(1))

        self._proc.fom_type = "unknown"
        with self.assertRaisesRegex(ProcessingError, "Unknown"):
            self._proc.run_once(ProcessedData(1))
