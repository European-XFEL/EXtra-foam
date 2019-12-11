import numpy as np

from extra_foam.pipeline.data_model import ImageData, ProcessedData


class _BaseProcessorTest:
    def _gen_images(self, gen, shape, dtype):
        if gen == 'random':
            imgs = np.random.randn(*shape).astype(dtype)
        elif gen == 'ones':
            imgs = np.ones(shape, dtype=dtype)
        else:
            raise ValueError

        return imgs

    def simple_data(self, tid, shape, *,
                    dtype=np.float32, gen='random', **kwargs):
        """Return a 'data' used in pipeline."""
        processed = ProcessedData(tid)
        imgs = self._gen_images(gen, shape, dtype)

        processed.image = ImageData.from_array(imgs, **kwargs)

        data = {'processed': processed,
                'raw': dict()}
        return data, processed

    def data_with_assembled(self, tid, shape, *,
                            dtype=np.float32,
                            gen='random',
                            with_image_mask=False, **kwargs):
        processed = ProcessedData(tid)
        imgs = self._gen_images(gen, shape, dtype)

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
