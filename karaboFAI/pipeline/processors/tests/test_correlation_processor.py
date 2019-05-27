import unittest

from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.processors import CorrelationProcessor
from karaboFAI.pipeline.exceptions import ProcessingError


class TestCorrelationProcessor(unittest.TestCase):
    def setUp(self):
        self._proc = CorrelationProcessor()

    def testRaise(self):
        self._proc.fom_type = "unknown"
        with self.assertRaisesRegex(ProcessingError, "Unknown"):
            self._proc.run_once(ProcessedData(1))
