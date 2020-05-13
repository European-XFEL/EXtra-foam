"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pipeline.data_model import ProcessedData
from ..config import config, DataSource


class DataTransformer:
    """DataTransformer class.

    Transform required data (data in SourceCatalog) and correlate them
    by train ID.
    """

    def __init__(self, catalog, *, cache_size=None):
        """Initialization.

        :param SourceCatalog catalog: data source catalog. The contained
            source items are all indispensable.
        :param int cache_size: maximum length of the cache used in data
            correlation by train ID
        """
        self._catalog = catalog

        self._cached = OrderedDict()
        self._cache_size = config["TRANSFORMER_CACHE_SIZE"] \
            if cache_size is None else cache_size

        # keep the latest correlated train ID
        self._correlated_tid = -1

    @staticmethod
    def transform_euxfel(data, *, catalog=None, source_type=DataSource.UNKNOWN):
        """Transform European XFEL data.

        :param tuple data: a tuple of (data, meta).
        :param SourceCatalog catalog: catalog for data source items.
        :param DataSource source_type: the format of the main detector
            source.

        :return: (raw, meta, tid)
        :rtype: (dict, dict, int)
        """
        raw, meta = data

        tids = set()
        for src in meta:
            tids.add(meta[src]['timestamp.tid'])

        new_raw, new_meta = dict(), dict()

        if not tids:
            return dict(), dict(), -1

        if len(tids) > 1:
            raise RuntimeError(
                f"Received data sources with different train IDs: {tids}")

        tid = tids.pop()
        for src, item in catalog.items():
            src_name, modules, src_ppt = item.name, item.modules, item.property

            if modules:
                prefix, suffix = src_name.split("*")
                new_raw[src] = dict()
                module_data = new_raw[src]
                n_found = 0
                for idx in modules:
                    module_name = f"{prefix}{idx}{suffix}"
                    if module_name in raw:
                        module_data[module_name] = raw[module_name]
                        n_found += 1

                if n_found == 0:
                    # there is no module data
                    continue
                else:
                    new_meta[src] = {
                        'train_id': tid, 'source_type': source_type,
                    }
            else:
                try:
                    # caveat: the sequence matters because of property
                    try:
                        new_raw[src] = raw[src_name][src_ppt]
                    except KeyError:
                        new_raw[src] = raw[src_name][f"{src_ppt}.value"]
                except KeyError:
                    # if the requested source or property is not in the data
                    continue

                new_meta[src] = {
                    'train_id': tid, 'source_type': source_type,
                }

        return new_raw, new_meta, tid

    def _found_all(self, meta):
        for k in self._catalog:
            if k not in meta:
                return False
        return True

    def correlate(self, data, *, source_type=DataSource.UNKNOWN):
        """Transform and correlate.

        :param tuple data: (data, meta).
        :param DataSource source_type: source type.

        :return: (correlated, dropped)
        :rtype: (dict, list)
        """
        catalog = self._catalog
        raw, meta, tid = self.transform_euxfel(
            data, catalog=catalog, source_type=source_type)

        correlated = None
        dropped = []
        if tid > 0:
            # update cached data
            cached = self._cached.setdefault(
                tid, {'meta': dict(), 'raw': dict()})

            cached['meta'].update(meta)
            cached['raw'].update(raw)

            if self._found_all(cached['meta']):
                correlated = {
                    'catalog': catalog.__copy__(),
                    'meta': cached['meta'],
                    'raw': cached['raw'],
                    'processed': ProcessedData(tid)
                }
                self._correlated_tid = tid
                while True:
                    # delete old data
                    key, item = self._cached.popitem(last=False)
                    if key == tid:
                        break

                    dropped.append((key, self._not_found_message(
                        key, item['meta'].keys())))

            if len(self._cached) > self._cache_size:
                key, item = self._cached.popitem(last=False)
                dropped.append((key, self._not_found_message(
                    key, item['meta'].keys())))

        return correlated, dropped

    def _not_found_message(self, tid, found):
        not_found = []
        for src in self._catalog.keys():
            if src not in found:
                not_found.append(src)

        msg = f"Train {tid} dropped! "
        # TODO: in case of reducing requested sources on the
        #       fly, the old uncorrelated data may become
        #       correlated data. Here, we do not declare them
        #       correlated data since it will produce a peak
        #       load on the pipeline. On the other hand, it
        #       is impossible to have a detailed error message
        #       since we do not know which data are requested
        #       in the old time.
        if not_found:
            msg += f"Found {len(found)} out of {len(self._catalog)} sources. " \
                   f"Not found: {not_found[0]} ..."
        return msg

    def reset(self):
        """Override."""
        self._cached.clear()
        self._correlated_tid = -1
