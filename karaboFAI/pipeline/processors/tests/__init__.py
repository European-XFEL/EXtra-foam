import unittest

import numpy as np

from karaboFAI.pipeline.data_model import ImageData, ProcessedData


class _BaseProcessorTest(unittest.TestCase):
    def simple_data(self, tid, shape, *,
                    dtype=np.float32, fill=None, **kwargs):
        """Return a 'data' used in pipeline."""
        processed = ProcessedData(tid)
        if fill is None:
            imgs = np.random.randn(*shape).astype(dtype)
        else:
            imgs = np.ones(shape, dtype=dtype)
        processed.image = ImageData.from_array(imgs, **kwargs)

        data = {'processed': processed,
                'raw': dict()}
        return data, processed

    def data_with_assembled(self, tid, shape, *,
                            dtype=np.float32, fill=None, **kwargs):
        processed = ProcessedData(tid)
        if fill is None:
            imgs = np.random.randn(*shape).astype(dtype)
        else:
            imgs = np.ones(shape, dtype=dtype)
        processed.image = ImageData.from_array(imgs, **kwargs)

        data = {'processed': processed,
                'raw': dict(),
                'assembled': imgs}
        return data, processed
