import unittest

import numpy as np

from karaboFAI.pipeline.data_model import ImageData, ProcessedData


class _BaseProcessorTest(unittest.TestCase):
    def simple_data(self, tid, shape):
        """Return a 'data' used in pipeline."""
        processed = ProcessedData(1001)
        processed.image = ImageData.from_array(np.random.randn(*shape))
        data = {'processed': processed,
                'raw': dict()}
        return data, processed
