"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Proxy for metadata which is stored in Redis.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from .config import redis_connection


class Metadata:

    DATA_SOURCE = "metadata:source"

    GENERAL_PROC = "metadata:proc:general"
    GEOMETRY_PROC = "metadata:proc:geometry"
    AZIMUTHAL_INTEG_PROC = "metadata:proc:azimuthal_integration"
    PUMP_PROBE_PROC = "metadata:proc:pump_probe"
    ROI_PROC = "metadata:proc:roi"
    CORRELATION_PROC = "metadata:proc:correlation"
    XAS_PROC = "metadata:proc:xas"
    BINNING_PROC = "metadata:proc:binning"

    _meta = {
        DATA_SOURCE: {
            "endpoint": "",
            "data_folder": "",
            "detector_source_name": "",
            "source_type": "",
            "xgm_source_name": "",
            "mono_source_name": "",
        }
    }

    _proc_meta = {
        GENERAL_PROC: {
            "sample_distance": "",
            "photon_energy": "",
            "pulse_id_range": "",
        },
        GEOMETRY_PROC: {
            "geometry_file": ""
        },
        # region and visibility must be give first since the image tool window
        # is not opened by default
        ROI_PROC: {
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
        },
        AZIMUTHAL_INTEG_PROC: {
            "integ_center_x": "",
            "integ_center_y": "",
            "integ_method": "",
            "integ_points": "",
            "integ_range": "",
            "normalizer": "",
            "auc_range": "",
            "fom_integ_range": "",
            "enable_pulsed_ai": ""
        },
        PUMP_PROBE_PROC: {
            "mode": "",
            "on_pulse_ids": "",
            "off_pulse_ids": "",
            "abs_difference": "",
            "analysis_type": "",
            "ma_window": "",
        },
        CORRELATION_PROC: {
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
        },
        XAS_PROC: {
            "energy_bins": "",
        },
        BINNING_PROC: {
            "n_bins": "",
            "bin_range": "",
            "analysis_type": "",
        }
    }

    @classmethod
    def reset(cls):
        redis = redis_connection()

        for key, value in cls._meta.items():
            redis.hmset(key, value)

        for key, value in cls._proc_meta.items():
            redis.hmset(key, value)
