import random

import numpy as np

from extra_foam.database import SourceCatalog, SourceItem
from extra_foam.pipeline.data_model import ImageData, ProcessedData
from extra_foam.pipeline.processors.base_processor import (
    SimplePairSequence, OneWayAccuPairSequence)
from extra_foam.config import config, DataSource


class _TestDataMixin:

    @classmethod
    def _gen_images(cls, gen, shape, dtype):
        if gen == 'random':
            imgs = np.random.randn(*shape).astype(dtype)
        elif gen == 'ones':
            imgs = np.ones(shape, dtype=dtype)
        else:
            raise ValueError

        return imgs

    @classmethod
    def simple_data(cls, tid, shape, *,
                    src_type=DataSource.BRIDGE,
                    dtype=config['SOURCE_PROC_IMAGE_DTYPE'],
                    gen='random', **kwargs):
        """Return a 'data' used in pipeline."""
        data, processed = cls.data_with_assembled(
            tid, shape, src_type=src_type, dtype=dtype, gen=gen, **kwargs
        )
        del data['assembled']
        return data, processed

    @classmethod
    def data_with_assembled(cls, tid, shape, *,
                            src_type=DataSource.BRIDGE,
                            dtype=config['SOURCE_PROC_IMAGE_DTYPE'],
                            gen='random',
                            with_image_mask=False,
                            slicer=None,
                            with_xgm=False,
                            with_digitizer=False,
                            **kwargs):
        imgs = cls._gen_images(gen, shape, dtype)

        processed = ProcessedData(tid)
        processed.image = ImageData.from_array(imgs, **kwargs)
        if with_image_mask:
            image_mask = np.zeros(shape[-2:], dtype=np.bool)
            image_mask[::2, ::2] = True
            processed.image.image_mask = image_mask

        slicer = slice(None, None) if slicer is None else slicer
        src_list = [('Foo', 'oof'), ('Bar', 'rab'), ('karaboFAI', 'extra_foam')]
        src_name, key_name = random.choice(src_list)

        catalog = SourceCatalog()
        ctg = 'ABCD'
        src = f'{src_name} {key_name}'
        catalog.add_item(SourceItem(ctg, src_name, [], key_name, slicer, None))
        catalog._main_detector = src

        n_pulses = processed.n_pulses

        if with_xgm:
            # generate XGM data
            processed.pulse.xgm.intensity = np.random.rand(n_pulses)
            processed.xgm.intensity = random.random()
            processed.xgm.x = random.random()
            processed.xgm.y = random.random()

        if with_digitizer:
            # generate digitizer data
            digitizer = processed.pulse.digitizer
            digitizer.ch_normalizer = 'B'
            for ch in digitizer:
                digitizer[ch].pulse_integral = np.random.rand(n_pulses)

        data = {
            'processed': processed,
            'catalog': catalog,
            'meta': {
                src: {
                    'timestamp.tid': tid,
                    'source_type': src_type,
                }
            },
            'raw': {
                src: dict()
            },
            'assembled': {
                'data': imgs,
                'sliced': imgs[slicer]
            }
        }

        return data, processed

    @classmethod
    def processed_data(cls, tid, shape, *,
                       gen='random',
                       dtype=config['SOURCE_PROC_IMAGE_DTYPE'],
                       roi_histogram=False,
                       histogram=False,
                       correlation=False,
                       binning=False,
                       **kwargs):

        processed = ProcessedData(tid)
        imgs = cls._gen_images(gen, shape, dtype)
        processed.image = ImageData.from_array(imgs, **kwargs)

        if roi_histogram:
            pass

        if histogram:
            hist = processed.hist
            hist.hist = np.arange(10)
            hist.bin_centers = np.arange(10) / 100.
            hist.mean, hist.median, hist.std = 1., 0, 0.1

        if correlation:
            corr_resolution = 2
            for i in range(2):
                corr = processed.corr[i]
                if i == 0:
                    data = SimplePairSequence()
                else:
                    data = OneWayAccuPairSequence(corr_resolution)

                for j in range(5):
                    data.append((j, 5 * j))

                corr.x, corr.y = data.data()
                corr.source = f"abc - {i}"
                corr.resolution = 0 if i == 0 else corr_resolution

        if binning:
            pass

        return processed
