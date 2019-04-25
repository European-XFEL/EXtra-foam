import unittest

from karaboFAI.config import FomName
from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.data_processor import CorrelationProcessor
from karaboFAI.pipeline.exceptions import ProcessingError


class TestCorrelationProcessor(unittest.TestCase):
    def setUp(self):
        self._proc = CorrelationProcessor()

    def testRaise(self):
        for fom in FomName:
            self._proc.fom_name = fom
            if fom == FomName.PUMP_PROBE_FOM:
                with self.assertRaisesRegex(ProcessingError, "Pump-probe"):
                    self._proc.process(ProcessedData(1))
            elif fom == FomName.AI_MEAN:
                with self.assertRaisesRegex(ProcessingError, "result is not"):
                    self._proc.process(ProcessedData(1))
            else:
                self._proc.process(ProcessedData(1))

        self._proc.fom_name = "unknown"
        with self.assertRaisesRegex(ProcessingError, "Unknown"):
            self._proc.process(ProcessedData(1))
