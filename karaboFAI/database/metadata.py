"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import pickle

from .base_proxy import _AbstractProxy
from .db_utils import redis_except_handler


class MetaMetadata(type):
    def __new__(mcs, name, bases, class_dict):
        proc_list = []
        for k, v in class_dict.items():
            if isinstance(v, str):
                s = v.split(":")
                if len(s) == 3 and s[1] == 'proc':
                    proc_list.append(s[2])

        class_dict['processors'] = proc_list
        cls = type.__new__(mcs, name, bases, class_dict)
        return cls


class Metadata(metaclass=MetaMetadata):

    CONNECTION = "meta:connection"
    ANALYSIS_TYPE = "meta:analysis_type"

    # The key of processors' metadata must end with '_PROC'
    GLOBAL_PROC = "meta:proc:global"
    IMAGE_PROC = "meta:proc:image"
    GEOMETRY_PROC = "meta:proc:geometry"
    AZIMUTHAL_INTEG_PROC = "meta:proc:azimuthal_integration"
    PUMP_PROBE_PROC = "meta:proc:pump_probe"
    ROI_PROC = "meta:proc:roi"
    CORRELATION_PROC = "meta:proc:correlation"
    BIN_PROC = "meta:proc:bin"
    STATISTICS_PROC = "meta:proc:statistics"
    PULSE_FILTER_PROC = "meta:proc:pulse_filter"
    DARK_RUN_PROC = "meta:proc:dark_run"
    TR_XAS_PROC = "meta:proc:tr_xas"


class MetaProxy(_AbstractProxy):
    """Proxy for retrieving metadata."""
    SESSION = "meta:session"

    # The real key depends on the category of the data source. For example,
    # 'XGM' has the key 'meta:sources:XGM' and 'DSSC' has the key
    # 'meta:sources:DSSC'.
    # The value is an unordered set for each source.
    DATA_SOURCE = "meta:data_source"

    def set_session(self, mapping):
        return self.hmset(self.SESSION, mapping)

    def get_session(self):
        return self.hget_all(self.SESSION)

    @redis_except_handler
    def add_data_source(self, src):
        """Add a data source.

        :return: the number of elements that were added to the set, not
                 including all the elements already present into the set.
        """
        return self._db.sadd(f'{self.DATA_SOURCE}:{src.category}',
                             pickle.dumps(src))

    @redis_except_handler
    def remove_data_source(self, src):
        """Remove a data source.

        :return: the number of members that were removed from the set,
                 not including non existing members.
        """
        return self._db.srem(f'{self.DATA_SOURCE}:{src.category}',
                             pickle.dumps(src))

    @redis_except_handler
    def get_all_data_sources(self, category):
        """Get all the data sources in a category.

        :return: a list of SourceItem.
        """
        return [pickle.loads(src) for src in
                self._db_nodecode.smembers(f'{self.DATA_SOURCE}:{category}')]
