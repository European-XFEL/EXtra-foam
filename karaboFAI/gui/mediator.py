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

from ..metadata import Metadata as mt
from ..metadata import MetaProxy
from ..pipeline.data_model import DataManager
from ..config import CorrelationFom, redis_connection


class Mediator(QObject):
    """Mediator for GUI signal-slot connection."""

    start_file_server_sgn = pyqtSignal()
    stop_file_server_sgn = pyqtSignal()
    file_server_started_sgn = pyqtSignal()
    file_server_stopped_sgn = pyqtSignal()

    vip_pulse_index1_sgn = pyqtSignal(int)
    vip_pulse_index2_sgn = pyqtSignal(int)
    # tell the control widget to update VIP pulse indices
    vip_pulse_indices_connected_sgn = pyqtSignal()

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

        self._meta = MetaProxy()
        self._db = redis_connection()
        self._data = DataManager()

        self._is_initialized = True

    def onBridgeEndpointChange(self, value: str):
        self._meta.set(mt.DATA_SOURCE, "endpoint", value)

    def onDataFolderChange(self, value: str):
        self._meta.set(mt.DATA_SOURCE, "data_folder", value)

    def onDetectorSourceNameChange(self, value: str):
        self._meta.set(mt.DATA_SOURCE, "detector_source_name", value)

    def onXgmSourceNameChange(self, value: str):
        self._meta.set(mt.DATA_SOURCE, "xgm_source_name", value)

    def onSourceTypeChange(self, value: IntEnum):
        self._meta.set(mt.DATA_SOURCE, "source_type", int(value))

    def onImageThresholdMaskChange(self, value: tuple):
        self._meta.set(mt.IMAGE_PROC, "threshold_mask", str(value))

    def onImageMaWindowChange(self, value: int):
        self._meta.set(mt.IMAGE_PROC, "ma_window", value)

    def onImageBackgroundChange(self, value: float):
        self._meta.set(mt.IMAGE_PROC, "background", value)

    def onImageMaskRegionChange(self, value: tuple):
        self._db.publish("command:image_mask", str(value))

    def onGeomFilenameChange(self, value: str):
        self._meta.set(mt.GEOMETRY_PROC, "geometry_file", value)

    def onGeomQuadPositionsChange(self, value: str):
        self._meta.set(mt.GEOMETRY_PROC, "quad_positions", json.dumps(value))

    def onPulseIndexSelectorChange(self, value: list):
        self._meta.set(mt.GENERAL_PROC, 'selected_pulse_indices', str(value))

    def onSampleDistanceChange(self, value: float):
        self._meta.set(mt.GENERAL_PROC, 'sample_distance', value)

    def onPhotonEnergyChange(self, value: float):
        self._meta.set(mt.GENERAL_PROC, 'photon_energy', value)

    def onAiIntegCenterXChange(self, value: int):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'integ_center_x', value)

    def onAiIntegCenterYChange(self, value: int):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'integ_center_y', value)

    def onAiIntegMethodChange(self, value: str):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'integ_method', value)

    def onAiIntegPointsChange(self, value: int):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'integ_points', value)

    def onAiIntegRangeChange(self, value: tuple):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'integ_range', str(value))

    def onAiNormalizerChange(self, value: IntEnum):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'normalizer', int(value))

    def onAiAucChangeChange(self, value: tuple):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'auc_range', str(value))

    def onAiFomIntegRangeChange(self, value: tuple):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'fom_integ_range', str(value))

    def onAiPulsedIntegStateChange(self, value: bool):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'enable_pulsed_ai', str(value))

    def onPpModeChange(self, value: IntEnum):
        value = int(value)
        if self._meta.get(mt.PUMP_PROBE_PROC, 'mode') != value:
            self._data.reset_pp()
            if self._meta.get(mt.CORRELATION_PROC, 'fom_type') == \
                    int(CorrelationFom.PUMP_PROBE_FOM):
                self._data.reset_correlation()
        self._meta.set(mt.PUMP_PROBE_PROC, 'mode', value)

    def onPpOnPulseIdsChange(self, value: list):
        self._meta.set(mt.PUMP_PROBE_PROC, 'on_pulse_indices', str(value))

    def onPpOffPulseIdsChange(self, value: list):
        self._meta.set(mt.PUMP_PROBE_PROC, 'off_pulse_indices', str(value))

    def onPpAnalysisTypeChange(self, value: IntEnum):
        self._meta.set(mt.PUMP_PROBE_PROC, 'analysis_type', int(value))
        self._data.reset_pp()
        if self._meta.get(mt.CORRELATION_PROC, 'fom_type') == \
                int(CorrelationFom.PUMP_PROBE_FOM):
            self._data.reset_correlation()

    def onPpAbsDifferenceChange(self, value: bool):
        self._meta.set(mt.PUMP_PROBE_PROC, "abs_difference", str(value))

    def onPpMaWindowChange(self, value: int):
        self._meta.set(mt.PUMP_PROBE_PROC, "ma_window", value)

    def onPpReset(self):
        self._data.reset_pp()

    def onRoiRegionChange(self, value: tuple):
        rank, x, y, w, h = value
        self._meta.set(mt.ROI_PROC, f'region{rank}', str((x, y, w, h)))

    def onRoiVisibilityChange(self, value: tuple):
        rank, is_visible = value
        self._meta.set(mt.ROI_PROC, f'visibility{rank}', str(is_visible))

    def onRoiFomChange(self, value: IntEnum):
        self._meta.set(mt.ROI_PROC, 'fom_type', int(value))
        self._data.reset_roi()

    def onRoiReset(self):
        self._data.reset_roi()

    def onProj1dNormalizerChange(self, value: IntEnum):
        self._meta.set(mt.ROI_PROC, "proj1d:normalizer", int(value))

    def onProj1dAucRangeChange(self, value: tuple):
        self._meta.set(mt.ROI_PROC, "proj1d:auc_range", str(value))

    def onProj1dFomIntegRangeChange(self, value: tuple):
        self._meta.set(mt.ROI_PROC, "proj1d:fom_integ_range", str(value))

    def onCorrelationFomChange(self, value: IntEnum):
        self._meta.set(mt.CORRELATION_PROC, "fom_type", int(value))
        self._data.reset_correlation()

    def onCorrelationParamChange(self, value: tuple):
        # index, device ID, property name, resolution
        # index starts from 1
        index, device_id, ppt, resolution = value
        self._data.add_correlation(index, device_id, ppt, resolution)
        self._meta.set(mt.CORRELATION_PROC, f'device_id{index}', device_id)
        self._meta.set(mt.CORRELATION_PROC, f'property{index}', ppt)
        self._meta.set(mt.CORRELATION_PROC, f'resolution{index}', resolution)

    def onCorrelationReset(self):
        self._data.reset_correlation()

    def onXasMonoSourceNameChange(self, value: str):
        self._meta.set(mt.XAS_PROC, "mono_source_name", value)

    def onXasEnergyBinsChange(self, value: int):
        self._meta.set(mt.XAS_PROC, "n_bins", value)

    def onXasBinRangeChange(self, value: tuple):
        self._meta.set(mt.XAS_PROC, "bin_range", str(value))

    def onXasReset(self):
        self._data.reset_xas()

    def onBinGroupChange(self, value: tuple):
        # index, device ID, property name, bin_range, number of bins,
        # where the index starts from 1
        index, device_id, ppt, bin_range, n_bins = value
        if index == 1:
            suffix = '_x'
        else:
            suffix = '_y'

        self._meta.set(mt.BIN_PROC, f'device_id{suffix}', device_id)
        self._meta.set(mt.BIN_PROC, f'property{suffix}', ppt)
        self._meta.set(mt.BIN_PROC, f'bin_range{suffix}', str(bin_range))
        self._meta.set(mt.BIN_PROC, f'n_bins{suffix}', n_bins)

    def onBinReset(self):
        self._meta.set(mt.BIN_PROC, "reset", 1)

    def onBinAnalysisTypeChange(self, value: IntEnum):
        self._meta.set(mt.BIN_PROC, "analysis_type", int(value))

    def onBinModeChange(self, value: IntEnum):
        self._meta.set(mt.BIN_PROC, "mode", int(value))
