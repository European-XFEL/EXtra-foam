"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from datetime import datetime
import os.path as osp
from collections import OrderedDict

import redis
import yaml
from yaml.scanner import ScannerError
from yaml.parser import ParserError

from .base_proxy import _AbstractProxy
from .db_utils import redis_except_handler
from ..config import AnalysisType, config


class MetaMetadata(type):
    def __new__(mcs, name, bases, class_dict):
        proc_list = []
        proc_keys = []
        for k, v in class_dict.items():
            if isinstance(v, str):
                s = v.split(":")
                if len(s) == 3 and s[1] == 'proc':
                    proc_list.append(s[2])
                    proc_keys.append(v)

        class_dict['processors'] = proc_list
        class_dict['processor_keys'] = proc_keys
        cls = type.__new__(mcs, name, bases, class_dict)
        return cls


class Metadata(metaclass=MetaMetadata):

    SESSION = "meta:session"

    CONNECTION = "meta:connection"
    EXTENSION = "meta:extension"

    ANALYSIS_TYPE = "meta:analysis_type"

    # The key of processors' metadata must end with '_PROC'
    META_PROC = "meta:proc:meta"
    GLOBAL_PROC = "meta:proc:global"
    IMAGE_PROC = "meta:proc:image"
    IMAGE_TRANSFORM_PROC = "meta:proc:image_transform"
    GEOMETRY_PROC = "meta:proc:geometry"
    AZIMUTHAL_INTEG_PROC = "meta:proc:azimuthal_integration"
    PUMP_PROBE_PROC = "meta:proc:pump_probe"
    ROI_PROC = "meta:proc:roi"
    CORRELATION_PROC = "meta:proc:correlation"
    BIN_PROC = "meta:proc:bin"
    HISTOGRAM_PROC = "meta:proc:histogram"
    FOM_FILTER_PROC = "meta:proc:fom_filter"
    DARK_RUN_PROC = "meta:proc:dark_run"

    # The real key depends on the category of the data source. For example,
    # 'XGM' has the key 'meta:sources:XGM' and 'DSSC' has the key
    # 'meta:sources:DSSC'.
    # The value is an unordered set for each source.
    DATA_SOURCE = "meta:data_source"


class MetaProxy(_AbstractProxy):
    """Proxy for retrieving metadata."""

    def has_analysis(self, analysis_type):
        """Check if the given analysis type has been registered.

        :param AnalysisType analysis_type: analysis type.
        """
        return int(self.hget(Metadata.ANALYSIS_TYPE, analysis_type))

    def has_any_analysis(self, analysis_types):
        """Check if any of the listed analysis types has been registered.

        :param tuple/list analysis_types: a list of AnalysisType instances.
        """
        if not isinstance(analysis_types, (tuple, list)):
            raise TypeError("Input must be a tuple or list!")

        for analysis_type in analysis_types:
            if int(self.hget(Metadata.ANALYSIS_TYPE, analysis_type)) > 0:
                return True
        return False

    def has_all_analysis(self, analysis_types):
        """Check if all of the listed analysis types have been registered.

        :param tuple/list analysis_types: a list of AnalysisType instances.
        """
        if not isinstance(analysis_types, (tuple, list)):
            raise TypeError("Input must be a tuple or list!")

        for analysis_type in analysis_types:
            if int(self.hget(Metadata.ANALYSIS_TYPE, analysis_type)) <= 0:
                return False
        return True

    def get_all_analysis(self):
        """Query all the registered analysis types.

        :return: None if the connection failed;
                 otherwise, a dictionary of key-value pairs (
                 analysis type: number of registrations).
        """
        return self.hget_all(Metadata.ANALYSIS_TYPE)

    def register_analysis(self, analysis_type):
        """Register the given analysis type.

        :param AnalysisType analysis_type: analysis type.
        """
        return self.hincrease_by(Metadata.ANALYSIS_TYPE, analysis_type, 1)

    def unregister_analysis(self, analysis_type):
        """Unregister the given analysis type.

        :param AnalysisType analysis_type: analysis type.
        """
        if int(self.hget(Metadata.ANALYSIS_TYPE, analysis_type)) > 0:
            return self.hincrease_by(Metadata.ANALYSIS_TYPE, analysis_type, -1)
        return 0

    @redis_except_handler
    def add_data_source(self, item):
        """Add a data source.

        :param tuple item: a tuple which can be used to construct a SourceItem.
        """
        ctg, name, modules, ppt, slicer, vrange, ktype = item
        key = f"{name} {ppt}"
        return self._db.pipeline().execute_command(
            'HSET', key, 'category', ctg,
                         'name', name,
                         'modules', modules,
                         'property', ppt,
                         'slicer', slicer,
                         'vrange', vrange,
                         'ktype', ktype).execute_command(
            'PUBLISH', Metadata.DATA_SOURCE, key).execute()

    @redis_except_handler
    def remove_data_source(self, src):
        """Remove a data source.

        :param str src: data source.
        """
        return self._db.pipeline().execute_command(
            'DEL', src).execute_command(
            'PUBLISH', Metadata.DATA_SOURCE, src).execute()

    @redis_except_handler
    def take_snapshot(self, name):
        """Take a snapshot of the current metadata.

        :param str name: name of the snapshot.
        """
        # return (unix time in seconds, microseconds)
        timestamp = self._db.execute_command("TIME")
        datetime_str = datetime.fromtimestamp(timestamp[0]).strftime(
            "%m/%d/%Y, %H:%M:%S")

        cfg = self._read_configuration()
        cfg[Metadata.META_PROC]["timestamp"] = datetime_str
        cfg[Metadata.META_PROC]["description"] = ""
        self._write_configuration(cfg, None, name)
        return name, datetime_str, ""

    @redis_except_handler
    def load_snapshot(self, name):
        """Load metadata snapshot by name.

        :param str name: name of the snapshot.
        """
        self._write_configuration(self._read_configuration(name), name, None)

    @redis_except_handler
    def copy_snapshot(self, old, new):
        """Copy metadata snapshot.

        :param str old: name of the old snapshot.
        :param str new: name of the new snapshot.
        """
        self._write_configuration(self._read_configuration(old), old, new)

    @redis_except_handler
    def remove_snapshot(self, name):
        """Remove metadata snapshot by name.

        :param str name: name of the snapshot
        """
        pipe = self.pipeline()
        for k in Metadata.processor_keys:
            pipe.execute_command("DEL", f"{k}:{name}")
        pipe.execute()

    @redis_except_handler
    def rename_snapshot(self, old, new):
        """Rename a metadata snapshot.

        :param str old: old configuration name.
        :param str new: new configuration name.
        """
        for k in Metadata.processor_keys:
            try:
                self._db.execute_command("RENAME", f"{k}:{old}", f"{k}:{new}")
            except redis.ResponseError:
                pass

    def _read_configuration(self, name=None):
        """Read a configuration from Redis.

        :param str name: configuration name.
        """
        cfg = dict()
        for k in Metadata.processor_keys:
            if name is not None:
                k = f"{k}:{name}"
            cfg[k] = self.hget_all(k)
        return cfg

    def _write_configuration(self, cfg, old, new):
        """Write a configuration into Redis.

        :param dict cfg: configuration.
        :param str old: old configuration name.
        :param str new: new configuration name.
        """
        invalid_keys = []
        for k, v in cfg.items():
            if old is not None:
                k_root = k.rsplit(':', maxsplit=1)[0]
            else:
                k_root = k

            if new is not None:
                k_new = f"{k_root}:{new}"
            else:
                k_new = k_root

            if k_root in Metadata.processor_keys:
                if v:
                    self._db.hmset(k_new, v)
                else:
                    self._db.execute_command("DEL", k_new)
            else:
                invalid_keys.append(k)

        if invalid_keys:
            self.warning(
                f"Invalid keys when writing configuration: {invalid_keys}")

    def dump_configurations(self, lst):
        """Dump all GUI configurations into file.

        :param list lst: a list of (name, description) of configurations in the
            Configurator widget.
        """
        filepath = config.setup_file
        with open(filepath, 'w') as fp:
            configurations = OrderedDict()
            for name, description in lst:
                configurations[name] = self._read_configuration(name)
                meta_key = f"{Metadata.META_PROC}:{name}"
                configurations[name][meta_key]["description"] = description

            if configurations:
                try:
                    yaml.dump(configurations, fp, Dumper=yaml.Dumper)
                except (ScannerError, ParserError) as e:
                    self.error(f"Invalid setup file: {filepath}\n{repr(e)}")

    def load_configurations(self):
        """Load all GUI configurations from file."""
        filepath = config.setup_file
        lst = []
        if osp.isfile(filepath):
            with open(filepath, 'r') as fp:
                try:
                    configurations = yaml.load(fp, Loader=yaml.Loader)
                    for name, cfg in configurations.items():
                        meta_key = f"{Metadata.META_PROC}:{name}"
                        try:
                            timestamp = cfg[meta_key]["timestamp"]
                        except KeyError:
                            self.error(f"Invalid configuration: {name}! "
                                       f"timestamp is missing")
                            continue

                        try:
                            description = cfg[meta_key]["description"]
                        except KeyError:
                            description = ""

                        lst.append((name, timestamp, description))
                        self._write_configuration(cfg, name, name)

                except (ScannerError, ParserError) as e:
                    self.error(f"Invalid setup file: {filepath}\n{repr(e)}")
        return lst
