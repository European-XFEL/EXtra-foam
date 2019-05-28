"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Proxy for metadata which is stored in Redis.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import redis

from .config import config, redis_connection


class Metadata:

    DATA_SOURCE = "metadata:source"

    GENERAL_PROC = "metadata:proc:general"
    GEOMETRY_PROC = "metadata:proc:geometry"
    AZIMUTHAL_INTEG_PROC = "metadata:proc:azimuthal_integration"
    PUMP_PROBE_PROC = "metadata:proc:pump_probe"
    ROI_PROC = "metadata:proc:roi"
    CORRELATION_PROC = "metadata:proc:correlation"
    XAS_PROC = "metadata:proc:xas"
    BIN_PROC = "metadata:proc:bin"
    IMAGE_PROC = "metadata:proc:image"

    _meta = {
        DATA_SOURCE: [
            "endpoint",
            "data_folder",
            "detector_source_name",
            "source_type",
            "xgm_source_name",
            "mono_source_name",
        ]
    }

    _proc_meta = {
        GENERAL_PROC: [
            "sample_distance",
            "photon_energy",
            "pulse_id_range",
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
            "fom_type",
            "proj1d:normalizer",
            "proj1d:auc_range",
            "proj1d:fom_integ_range",
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
            "enable_pulsed_ai",
        ],
        PUMP_PROBE_PROC: [
            "mode",
            "on_pulse_ids",
            "off_pulse_ids",
            "abs_difference",
            "analysis_type",
            "ma_window",
        ],
        CORRELATION_PROC: [
            "fom_type",
            "fom_integ_range",
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
        XAS_PROC: [
            "n_bins",
            "bin_range",
        ],
        BIN_PROC: [
            "device_id_x",
            "device_id_y",
            "property_x",
            "property_y",
            "n_bins_x",
            "n_bins_y",
            "bin_range_x",
            "bin_range_y",
            "analysis_type",
            "mode",
        ]
    }


class MetaProxy:
    def __init__(self):
        self._db = redis_connection()

    def set(self, name, key, value):
        """Set a Hash.

        :returns: -1 if connection fails;
                   1 if created a new field;
                   0 if set on a new field.
        """
        try:
            return self._db.hset(name, key, value)
        except redis.exceptions.ConnectionError:
            return -1

    def get(self, name, key):
        try:
            return self._db.hget(name, key)
        except redis.exceptions.ConnectionError:
            pass

    def delete(self, name, key):
        try:
            return self._db.hdel(name, key)
        except redis.exceptions.ConnectionError:
            pass

    def get_all(self, name):
        try:
            return self._db.hgetall(name)
        except redis.exceptions.ConnectionError:
            pass
