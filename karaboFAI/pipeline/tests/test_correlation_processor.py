import unittest

from karaboFAI.config import FomName
from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.data_processor import CorrelationProcessor


class TestCorrelationProcessor(unittest.TestCase):
    def setUp(self):
        self._proc = CorrelationProcessor()

    def testProcessingEmptyData(self):
        # test not raise if the ProcessedData history is empty
        for fom in FomName:
            self._proc.fom_name = fom
            self._proc.process(ProcessedData(-1))
