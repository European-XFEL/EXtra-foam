"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Proxy for metadata which is stored in Redis.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import functools

import redis

from karaboFAI.ipc import RedisConnection


class Metadata:

    DATA_SOURCE = "metadata:source"

    ANALYSIS_TYPE = "metadata:analysis_type"
    ANALYSIS_TYPE_PULSE = "metadata:analysis_type_pulse"

    GLOBAL_PROC = "metadata:proc:global"
    GEOMETRY_PROC = "metadata:proc:geometry"
    AZIMUTHAL_INTEG_PROC = "metadata:proc:azimuthal_integration"
    PUMP_PROBE_PROC = "metadata:proc:pump_probe"
    ROI_PROC = "metadata:proc:roi"
    CORRELATION_PROC = "metadata:proc:correlation"
    BIN_PROC = "metadata:proc:bin"
    IMAGE_PROC = "metadata:proc:image"
    STATISTICS_PROC = "metadata:proc:pulse_fom"
    DATA_REDUCTION_PROC = "metadata:proc:data_reduction"
    DARK_RUN = "metadata:proc:data_run"

    _meta = {
        SESSION: [
            "detector",
            "topic",
        ],
        ANALYSIS_TYPE: [
            "analysis_types",
        ],
        DATA_SOURCE: [
            "endpoint",
            "detector_source_name",
            "source_type",
            "xgm_source_name",
        ]
    }

    _proc_meta = {
        GLOBAL_PROC: [
            "sample_distance",
            "photon_energy",
            "selected_pulse_indices",
            "ma_window",
        ],
        IMAGE_PROC: [
            "threshold_mask",
            "ma_window",
            "background",
        ],
        GEOMETRY_PROC: [
            "geometry_file",
        ],
        # region and visibility must be give first since the image tool window
        # is not opened by default
        ROI_PROC: [
            "region1",
            "region2",
            "region3",
            "region4",
            "visibility1",
            "visibility2",
            "visibility3",
            "visibility4",
            "proj:direction",
            "proj:normalizer",
            "proj:auc_range",
            "proj:fom_integ_range",
        ],
        AZIMUTHAL_INTEG_PROC: [
            "integ_center_x",
            "integ_center_y",
            "integ_method",
            "integ_points",
            "integ_range",
            "normalizer",
            "auc_range",
            "fom_integ_range",
        ],
        PUMP_PROBE_PROC: [
            "analysis_type",
            "mode",
            "on_pulse_indices",
            "off_pulse_indices",
            "abs_difference",
        ],
        CORRELATION_PROC: [
            "analysis_type",
            "device_id1",
            "device_id2",
            "device_id3",
            "device_id4",
            "property1",
            "property2",
            "property3",
            "property4",
            "resolution1",
            "resolution2",
            "resolution3",
            "resolution4",
        ],
        BIN_PROC: [
            "analysis_type",
            "device_id_x",
            "device_id_y",
            "property_x",
            "property_y",
            "n_bins_x",
            "n_bins_y",
            "bin_range_x",
            "bin_range_y",
            "mode",
        ],
        STATISTICS_PROC: [
            "analysis_type",
            "n_bins",
            "pulse_resolved",
        ],
    }


def redis_except_handler(return_value):
    def wrap(f):
        @functools.wraps(f)
        def catched_f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except redis.exceptions.ConnectionError:
                return return_value
        return catched_f
    return wrap


class MetaProxy:
    _db = RedisConnection()

    def reset(self):
        self._db = None

    @redis_except_handler(-1)
    def set(self, name, key, value):
        """Set a Hash.

        :returns: -1 if connection fails;
                   1 if created a new field;
                   0 if set on a new field.
        """
        return self._db.hset(name, key, value)

    @redis_except_handler(-1)
    def mset(self, name, mapping):
        return self._db.hmset(name, mapping)

    @redis_except_handler(None)
    def get(self, name, key):
        return self._db.hget(name, key)

    @redis_except_handler(None)
    def mget(self, name, keys):
        return self._db.hmget(name, keys)

    @redis_except_handler(None)
    def delete(self, name, key):
        return self._db.hdel(name, key)

    @redis_except_handler(None)
    def get_all(self, name):
        return self._db.hgetall(name)

    @redis_except_handler(None)
    def increase_by(self, name, key, amount=1):
        return self._db.hincrby(name, key, amount)

    @redis_except_handler(None)
    def increase_by_float(self, name, key, amount=1):
        return self._db.hincrbyfloat(name, key, amount)
