"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Mediator class.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum
import json

from PyQt5.QtCore import pyqtSignal,  QObject

from ..metadata import MetadataProxy
from ..pipeline.data_model import DataManager
from ..config import CorrelationFom


class Mediator(QObject):
    """Mediator for GUI signal-slot connection."""

    start_file_server_sgn = pyqtSignal()
    stop_file_server_sgn = pyqtSignal()
    file_server_started_sgn = pyqtSignal()
    file_server_stopped_sgn = pyqtSignal()

    vip_pulse_id1_sgn = pyqtSignal(int)
    vip_pulse_id2_sgn = pyqtSignal(int)
    # tell the control widget to update VIP pulse IDs
    vip_pulse_ids_connected_sgn = pyqtSignal()

    reset_image_level_sgn = pyqtSignal()

    __instance = None

    def __new__(cls, *args, **kwargs):
        """Create a singleton."""
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
            cls.__instance._is_initialized = False
        return cls.__instance

    def __init__(self, *args, **kwargs):
        if self._is_initialized:
            return
        # this will reset all signal-slot connections
        super().__init__(*args, **kwargs)

        self._meta = MetadataProxy()
        self._data = DataManager()

        self._is_initialized = True

    def onBridgeEndpointChange(self, value: str):
        self._meta.ds_set("endpoint", value)

    def onDataFolderChange(self, value: str):
        self._meta.ds_set("data_folder", value)

    def onDetectorSourceNameChange(self, value: str):
        self._meta.ds_set("detector_source_name", value)

    def onXgmSourceNameChange(self, value: str):
        self._meta.ds_set("xgm_source_name", value)

    def onMonoSourceNameChange(self, value: str):
        self._meta.ds_set("mono_source_name", value)

    def onSourceTypeChange(self, value: IntEnum):
        self._meta.ds_set("source_type", int(value))

    def onGeometryFileChange(self, value: str):
        self._meta.geom_set("geometry_file", value)

    def onQuadPositionsChange(self, value: str):
        self._meta.geom_set("quad_positions", json.dumps(value))

    def onSampleDistanceChange(self, value: float):
        self._meta.ga_set('sample_distance', value)

    def onPhotonEnergyChange(self, value: float):
        self._meta.ga_set('photon_energy', value)

    def onPulseIdRangeChange(self, value: tuple):
        self._meta.ga_set('pulse_id_range', str(value))

    def onAiIntegCenterXChange(self, value: int):
        self._meta.ai_set('integ_center_x', value)

    def onAiIntegCenterYChange(self, value: int):
        self._meta.ai_set('integ_center_y', value)

    def onAiIntegMethodChange(self, value: str):
        self._meta.ai_set('integ_method', value)

    def onAiIntegPointsChange(self, value: int):
        self._meta.ai_set('integ_points', value)

    def onAiIntegRangeChange(self, value: tuple):
        self._meta.ai_set('integ_range', str(value))

    def onAiNormalizerChange(self, value: IntEnum):
        self._meta.ai_set('normalizer', int(value))

    def onAiAucChangeChange(self, value: tuple):
        self._meta.ai_set('auc_range', str(value))

    def onAiFomIntegRangeChange(self, value: tuple):
        self._meta.ai_set('fom_integ_range', str(value))

    def onAiPulsedIntegStateChange(self, value: bool):
        self._meta.ai_set('enable_pulsed_ai', str(value))

    def onPpModeChange(self, value: IntEnum):
        value = int(value)
        if self._meta.pp_get('mode') != value:
            self._data.reset_pp()
            if self._meta.corr_get('fom_type') == \
                    int(CorrelationFom.PUMP_PROBE_FOM):
                self._data.reset_correlation()
        self._meta.pp_set('mode', value)

    def onPpOnPulseIdsChange(self, value: list):
        self._meta.pp_set('on_pulse_ids', str(value))

    def onPpOffPulseIdsChange(self, value: list):
        self._meta.pp_set('off_pulse_ids', str(value))

    def onPpAnalysisTypeChange(self, value: IntEnum):
        self._meta.pp_set('analysis_type', int(value))
        self._data.reset_pp()
        if self._meta.corr_get('fom_type') == \
                int(CorrelationFom.PUMP_PROBE_FOM):
            self._data.reset_correlation()

    def onPpAbsDifferenceChange(self, value: bool):
        self._meta.pp_set("abs_difference", str(value))

    def onPpMaWindowChange(self, value: int):
        self._meta.pp_set("ma_window", value)

    def onPpReset(self):
        self._data.reset_pp()

    def onRoiRegionChange(self, value: tuple):
        rank, x, y, w, h = value
        self._meta.roi_set(f'region{rank}', str((x, y, w, h)))

    def onRoiVisibilityChange(self, value: tuple):
        rank, is_visible = value
        self._meta.roi_set(f'visibility{rank}', str(is_visible))

    def onRoiFomChange(self, value: IntEnum):
        self._meta.roi_set('fom_type', int(value))
        self._data.reset_roi()

    def onRoiReset(self):
        self._data.reset_roi()

    def onProj1dNormalizerChange(self, value: IntEnum):
        self._meta.roi_set("proj1d:normalizer", int(value))

    def onProj1dAucRangeChange(self, value: tuple):
        self._meta.roi_set("proj1d:auc_range", str(value))

    def onProj1dFomIntegRangeChange(self, value: tuple):
        self._meta.roi_set("proj1d:fom_integ_range", str(value))

    def onCorrelationFomChange(self, value: IntEnum):
        self._meta.corr_set("fom_type", int(value))
        self._data.reset_correlation()

    def onCorrelationParamChange(self, value: tuple):
        # index, device ID, property name, resolution
        # index starts from 1
        index, device_id, ppt, resolution = value
        self._data.add_correlation(index, device_id, ppt, resolution)
        self._meta.corr_set(f'device_id{index}', device_id)
        self._meta.corr_set(f'property{index}', ppt)
        self._meta.corr_set(f'resolution{index}', resolution)

    def onCorrelationReset(self):
        self._data.reset_correlation()

    def onXasEnergyBinsChange(self, value: int):
        self._meta.xas_set("energy_bins", value)

    def onXasReset(self):
        # FIXME
        # reset XAS processor?
        pass
