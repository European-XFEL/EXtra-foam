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
                            dtype=np.float32,
                            fill=None,
                            with_image_mask=False, **kwargs):
        processed = ProcessedData(tid)
        if fill is None:
            imgs = np.random.randn(*shape).astype(dtype)
        else:
            imgs = np.ones(shape, dtype=dtype)
        processed.image = ImageData.from_array(imgs, **kwargs)

        if with_image_mask:
            image_mask = np.zeros(shape[-2:], dtype=np.bool)
            image_mask[::2, ::2] = True
            processed.image.image_mask = image_mask

        data = {'processed': processed,
                'raw': dict(),
                'detector': {
                    'assembled': imgs,
                    'pulse_slicer': slice(None, None),
                }}
        return data, processed
