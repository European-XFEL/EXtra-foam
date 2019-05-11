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


class MetadataProxy:

    _redis = None

    @classmethod
    def redis(cls):
        if cls._redis is None:
            cls._redis = redis.Redis(decode_responses=True)

        return cls._redis

    @classmethod
    def reset(cls):
        cls.redis().hmset(
            "data_source", {
                "endpoint": "",
                "data_folder": "",
                "detector_source_name": "",
                "source_type": "",
                "xgm_source_name": "",
                "mono_source_name": "",
            }
        )

        cls.redis().hmset(
            "analysis:general", {
                "sample_distance": "",
                "photon_energy": "",
                "pulse_id_range": "",
            }
        )

        cls.redis().hmset(
            "analysis:geometry", {
                "geometry_file": ""
            }
        )

        # region and visibility must be give first since the image tool window
        # is not opened by default
        cls.redis().hmset(
            "analysis:ROI", {
                "region1": "[0, 0, 0, 0]",
                "region2": "[0, 0, 0, 0]",
                "region3": "[0, 0, 0, 0]",
                "region4": "[0, 0, 0, 0]",
                "visibility1": "False",
                "visibility2": "False",
                "visibility3": "False",
                "visibility4": "False",
                "fom_type": "",
                "proj1d:normalizer": "",
                "proj1d:auc_range": "",
                "proj1d:fom_integ_range": "",
            }
        )

        cls.redis().hmset(
            "analysis:azimuthal_integ", {
                "integ_center_x": "",
                "integ_center_y": "",
                "integ_method": "",
                "integ_points": "",
                "integ_range": "",
                "normalizer": "",
                "auc_range": "",
                "fom_integ_range": "",
                "enable_pulsed_ai": ""
            }
        )

        cls.redis().hmset(
            "analysis:pump_probe", {
                "mode": "",
                "on_pulse_ids": "",
                "off_pulse_ids": "",
                "abs_difference": "",
                "analysis_type": "",
                "ma_window": "",
            }
        )

        cls.redis().hmset(
            "analysis:correlation", {
                "fom_type": "",
                "fom_integ_range": "",
                "device_id1": "",
                "device_id2": "",
                "device_id3": "",
                "device_id4": "",
                "property1": "",
                "property2": "",
                "property3": "",
                "property4": "",
                "resolution1": "",
                "resolution2": "",
                "resolution3": "",
                "resolution4": "",
            }
        )

        cls.redis().hmset(
            "analysis:XAS", {
                "energy_bins": "",
            }
        )

    @classmethod
    def ds_set(cls, key, value):
        cls.redis().hset("data_source", key, value)

    @classmethod
    def ds_get(cls, key):
        return cls.redis().hget("data_source", key)

    @classmethod
    def ds_getall(cls):
        return cls.redis().hgetall("data_source")

    @classmethod
    def ga_set(cls, key, value):
        cls.redis().hset("analysis:general_analysis", key, value)

    @classmethod
    def ga_get(cls, key):
        return cls.redis().hget("analysis:general_analysis", key)

    @classmethod
    def ga_getall(cls):
        return cls.redis().hgetall("analysis:general_analysis")

    @classmethod
    def geom_set(cls, key, value):
        cls.redis().hset("analysis:geometry", key, value)

    @classmethod
    def geom_get(cls, key):
        return cls.redis().hget("analysis:geometry", key)

    @classmethod
    def geom_getall(cls):
        return cls.redis().hgetall("analysis:geometry")

    @classmethod
    def ai_set(cls, key, value):
        cls.redis().hset("analysis:azimuthal_integ", key, value)

    @classmethod
    def ai_get(cls, key):
        return cls.redis().hget("analysis:azimuthal_integ", key)

    @classmethod
    def ai_getall(cls):
        return cls.redis().hgetall("analysis:azimuthal_integ")

    @classmethod
    def pp_set(cls, key, value):
        cls.redis().hset("analysis:pump_probe", key, value)

    @classmethod
    def pp_get(cls, key):
        return cls.redis().hget("analysis:pump_probe", key)

    @classmethod
    def pp_getall(cls):
        return cls.redis().hgetall("analysis:pump_probe")

    @classmethod
    def roi_set(cls, key, value):
        cls.redis().hset("analysis:ROI", key, value)

    @classmethod
    def roi_get(cls, key):
        return cls.redis().hget("analysis:ROI", key)

    @classmethod
    def roi_getall(cls):
        return cls.redis().hgetall("analysis:ROI")

    @classmethod
    def corr_set(cls, key, value):
        cls.redis().hset("analysis:correlation", key, value)

    @classmethod
    def corr_get(cls, key):
        return cls.redis().hget("analysis:correlation", key)

    @classmethod
    def corr_getall(cls):
        return cls.redis().hgetall("analysis:correlation")

    @classmethod
    def xas_set(cls, key, value):
        cls.redis().hset("analysis:XAS", key, value)

    @classmethod
    def xas_get(cls, key):
        return cls.redis().hget("analysis:XAS", key)

    @classmethod
    def xas_getall(cls):
        return cls.redis().hgetall("analysis:XAS")
