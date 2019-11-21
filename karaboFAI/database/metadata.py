"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import pickle

from .base_proxy import _AbstractProxy
from .db_utils import redis_except_handler
from ..config import AnalysisType


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

    ANALYSIS_TYPE = "meta:analysis_type"

    def set_session(self, mapping):
        return self.hmset(self.SESSION, mapping)

    def get_session(self):
        return self.hget_all(self.SESSION)

    def has_analysis(self, analysis_type):
        """Check if the given analysis type has been registered.

        :param AnalysisType analysis_type: analysis type.
        """
        return int(self.hget(self.ANALYSIS_TYPE, analysis_type))

    def has_any_analysis(self, analysis_types):
        """Check if any of the listed analysis types has been registered.

        :param tuple/list analysis_types: a list of AnalysisType instances.
        """
        if not isinstance(analysis_types, (tuple, list)):
            raise TypeError("Input must be a tuple or list!")

        for analysis_type in analysis_types:
            if int(self.hget(self.ANALYSIS_TYPE, analysis_type)) > 0:
                return True
        return False

    def has_all_analysis(self, analysis_types):
        """Check if all of the listed analysis types have been registered.

        :param tuple/list analysis_types: a list of AnalysisType instances.
        """
        if not isinstance(analysis_types, (tuple, list)):
            raise TypeError("Input must be a tuple or list!")

        for analysis_type in analysis_types:
            if int(self.hget(self.ANALYSIS_TYPE, analysis_type)) <= 0:
                return False
        return True

    def get_all_analysis(self):
        """Query all the registered analysis types.

        :return: None if the connection failed;
                 otherwise, a dictionary of key-value pairs (
                 analysis type: number of registrations).
        """
        return self.hget_all(self.ANALYSIS_TYPE)

    def initialize_analysis_types(self):
        """Initialize all analysis types in Redis.

        Prevent 'has_analysis', 'has_any_analysis' and 'has_all_analysis'
        from getting None when querying.
        """
        return self.hmset(self.ANALYSIS_TYPE, {t: 0 for t in AnalysisType})

    def register_analysis(self, analysis_type):
        """Register the given analysis type.

        :param AnalysisType analysis_type: analysis type.
        """
        return self.hincrease_by(self.ANALYSIS_TYPE, analysis_type, 1)

    def unregister_analysis(self, analysis_type):
        """Unregister the given analysis type.

        :param AnalysisType analysis_type: analysis type.
        """
        if int(self.hget(self.ANALYSIS_TYPE, analysis_type)) > 0:
            return self.hincrease_by(self.ANALYSIS_TYPE, analysis_type, -1)
        return 0

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
