import random
import time

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
        elif gen == 'range':
            imgs = np.arange(np.prod(shape), dtype=dtype).reshape(*shape)
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
                            slicer=None,
                            with_xgm=False,
                            with_digitizer=False,
                            **kwargs):
        imgs = cls._gen_images(gen, shape, dtype)

        processed = ProcessedData(tid)

        processed.image = ImageData.from_array(imgs, **kwargs)

        if imgs.ndim == 2:
            slicer = None
        else:
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
            }
        }
        if imgs.ndim == 2:
            data['assembled']['sliced'] = imgs
        else:
            data['assembled']['sliced'] = imgs[slicer]

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


class _RawDataMixin:
    """Generate raw data used in test."""
    @staticmethod
    def _update_metadata(meta, src, timestamp, tid):
        sec, frac = str(timestamp).split('.')
        meta[src] = {
            'source': src,
            'timestamp': timestamp,
            'timestamp.tid': tid,
            'timestamp.sec': sec,
            'timestamp.frac': frac.ljust(18, '0')  # attosecond resolution
        }
        return meta

    def _create_catalog(self, mapping):
        """Generate source catalog.

        :param dict mapping: a dictionary with keys being the device categories
            and values being a list of (device ID, property).
        """
        catalog = SourceCatalog()
        for ctg, srcs in mapping.items():
            for src, ppt in srcs:
                catalog.add_item(SourceItem(ctg, src, [], ppt, None, None))
        return catalog

    def _gen_kb_data(self, tid, mapping):
        """Generate empty data in European XFEL data format.

        :param int tid: train ID.
        :param dict mapping: a dictionary with keys being the device IDs /
            output channels and values being a list of (property, value).
        """
        meta, data = {}, {}

        for src, ppts in mapping.items():
            self._update_metadata(meta, src, time.time(), tid)

            data[src] = dict()
            for ppt, value in ppts:
                data[src][ppt] = value

        return data, meta

    def _gen_data(self, tid, mapping, *, source_type=None):
        """Generate empty data in EXtra-foam data format.

        :param int tid: train ID.
        :param dict mapping: a dictionary with keys being the device IDs /
            output channels and values being the list of (property, value).
        """
        meta, data = {}, {}

        for name, ppts in mapping.items():
            for ppt, value in ppts:
                if ".value" in ppt:
                    # slow data from files
                    ppt = ppt[:-6]
                src = f"{name} {ppt}"
                data[src] = value
                meta[src] = {"train_id": tid, "source_type": source_type}

        return {
            "raw": data,
            "processed": None,
            "meta": meta,
            "catalog": None,
        }
