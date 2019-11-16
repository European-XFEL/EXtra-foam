"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum
import json

from PyQt5.QtCore import pyqtSignal,  QObject

from ..database import Metadata as mt
from ..database import MetaProxy
from ..ipc import RedisConnection


class Mediator(QObject):
    """Mediator for GUI signal-slot connection.

    The behavior of the code should not be affected by when the
    Mediator() is instantiated.
    """
    reset_image_level_sgn = pyqtSignal()

    __instance = None

    # TODO: make a command interface
    _db = RedisConnection()

    _meta = MetaProxy()

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

        self._is_initialized = True

    def onBridgeEndpointChange(self, value: str):
        self._meta.hset(mt.CONNECTION, "endpoint", value)

    def onSourceTypeChange(self, value: IntEnum):
        self._meta.hset(mt.CONNECTION, "source_type", int(value))

    def onDataSourceToggled(self, item, checked: bool):
        if checked:
            self._meta.add_data_source(item)
        else:
            self._meta.remove_data_source(item)

    def onImageThresholdMaskChange(self, value: tuple):
        self._meta.hset(mt.IMAGE_PROC, "threshold_mask", str(value))

    def onImageBackgroundChange(self, value: float):
        self._meta.hset(mt.IMAGE_PROC, "background", value)

    def onDarkSubtractionStateChange(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "dark_subtraction", str(value))

    def onGeomAssembleWithGeometryChange(self, value: bool):
        self._meta.hset(mt.GEOMETRY_PROC, "with_geometry", str(value))

    def onGeomFilenameChange(self, value: str):
        self._meta.hset(mt.GEOMETRY_PROC, "geometry_file", value)

    def onGeomQuadPositionsChange(self, value: str):
        self._meta.hset(mt.GEOMETRY_PROC, "quad_positions", json.dumps(value))

    def onPoiIndexChange(self, idx: int, value: int):
        self._meta.hset(mt.GLOBAL_PROC, f"poi{idx}_index", str(value))

    def onSampleDistanceChange(self, value: float):
        self._meta.hset(mt.GLOBAL_PROC, 'sample_distance', value)

    def onPhotonEnergyChange(self, value: float):
        self._meta.hset(mt.GLOBAL_PROC, 'photon_energy', value)

    def onMaWindowChange(self, value: int):
        self._meta.hset(mt.GLOBAL_PROC, "ma_window", value)

    def onResetMa(self):
        self._meta.hmset(mt.GLOBAL_PROC, {"reset_ma_ai": 1,
                                         "reset_ma_roi": 1,
                                         "reset_ma_xgm": 1})

    def onAiPixelSizeXChange(self, value: int):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'pixel_size_x', value)

    def onAiPixelSizeYChange(self, value: int):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'pixel_size_y', value)

    def onAiIntegCenterXChange(self, value: int):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_center_x', value)

    def onAiIntegCenterYChange(self, value: int):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_center_y', value)

    def onAiIntegMethodChange(self, value: str):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_method', value)

    def onAiIntegPointsChange(self, value: int):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_points', value)

    def onAiIntegRangeChange(self, value: tuple):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_range', str(value))

    def onCurveNormalizerChange(self, value: IntEnum):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'normalizer', int(value))

    def onAiAucChangeChange(self, value: tuple):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'auc_range', str(value))

    def onAiFomIntegRangeChange(self, value: tuple):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'fom_integ_range', str(value))

    def onPpModeChange(self, value: IntEnum):
        self._meta.hset(mt.PUMP_PROBE_PROC, 'mode', int(value))

    def onPpOnPulseIdsChange(self, value: list):
        self._meta.hset(mt.PUMP_PROBE_PROC, 'on_pulse_indices', str(value))

    def onPpOffPulseIdsChange(self, value: list):
        self._meta.hset(mt.PUMP_PROBE_PROC, 'off_pulse_indices', str(value))

    def onPpAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.PUMP_PROBE_PROC, 'analysis_type', int(value))

    def onPpAbsDifferenceChange(self, value: bool):
        self._meta.hset(mt.PUMP_PROBE_PROC, "abs_difference", str(value))

    def onPpReset(self):
        self._meta.hset(mt.PUMP_PROBE_PROC, "reset", 1)
        # reset moving average at the same time
        self.onResetMa()

    def onRoiGeometryChange(self, value: tuple):
        rank, x, y, w, h = value
        self._meta.hset(mt.ROI_PROC, f'region{rank}', str((x, y, w, h)))

    def onRoiVisibilityChange(self, value: tuple):
        rank, is_visible = value
        self._meta.hset(mt.ROI_PROC, f'visibility{rank}', str(is_visible))

    def onRoiProjDirectChange(self, value: str):
        self._meta.hset(mt.ROI_PROC, 'proj:direction', value)

    def onRoiProjNormalizerChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, "proj:normalizer", int(value))

    def onRoiProjAucRangeChange(self, value: tuple):
        self._meta.hset(mt.ROI_PROC, "proj:auc_range", str(value))

    def onRoiProjFomIntegRangeChange(self, value: tuple):
        self._meta.hset(mt.ROI_PROC, "proj:fom_integ_range", str(value))

    def onCorrelationAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.CORRELATION_PROC, "analysis_type", int(value))

    def onCorrelationParamChange(self, value: tuple):
        # index, device ID, property name, resolution
        # index starts from 1
        index, device_id, ppt, resolution = value
        self._meta.hset(mt.CORRELATION_PROC, f'device_id{index}', device_id)
        self._meta.hset(mt.CORRELATION_PROC, f'property{index}', ppt)
        self._meta.hset(mt.CORRELATION_PROC, f'resolution{index}', resolution)

    def onCorrelationReset(self):
        self._meta.hset(mt.CORRELATION_PROC, "reset", 1)

    def onBinGroupChange(self, value: tuple):
        # index, device ID, property name, bin_range, number of bins,
        # where the index starts from 1
        index, device_id, ppt, bin_range, n_bins = value

        self._meta.hset(mt.BIN_PROC, f'device_id{index}', device_id)
        self._meta.hset(mt.BIN_PROC, f'property{index}', ppt)
        self._meta.hset(mt.BIN_PROC, f'bin_range{index}', str(bin_range))
        self._meta.hset(mt.BIN_PROC, f'n_bins{index}', n_bins)

    def onBinAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.BIN_PROC, "analysis_type", int(value))

    def onBinModeChange(self, value: IntEnum):
        self._meta.hset(mt.BIN_PROC, "mode", int(value))

    def onBinReset(self):
        self._meta.hset(mt.BIN_PROC, "reset", 1)

    def onStAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.STATISTICS_PROC, "analysis_type", int(value))

    def onStNumBinsChange(self, value: int):
        self._meta.hset(mt.STATISTICS_PROC, "n_bins", int(value))

    def onStPulseOrTrainResolutionChange(self, value: bool):
        self._meta.hset(mt.STATISTICS_PROC, "pulse_resolved", str(value))

    def onStReset(self):
        self._meta.hset(mt.STATISTICS_PROC, "reset", 1)

    def onPfAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.PULSE_FILTER_PROC, "analysis_type", int(value))

    def onPfFomRangeChange(self, value: tuple):
        self._meta.hset(mt.PULSE_FILTER_PROC, "fom_range", str(value))

    def onRdStateChange(self, value: bool):
        self._meta.hset(mt.GLOBAL_PROC, "recording_dark", str(value))

    def onRdRemoveDark(self):
        self._meta.hset(mt.GLOBAL_PROC, "remove_dark", 1)

    def onTrXasScanStateToggled(self, value: IntEnum, state: bool):
        if state:
            self._meta.hset(mt.TR_XAS_PROC, "analysis_type", int(value))
        else:
            self._meta.hdel(mt.TR_XAS_PROC, "analysis_type")

    def onTrXasDelayDeviceChange(self, value: str):
        self._meta.hset(mt.TR_XAS_PROC, "delay_device", value)

    def onTrXasDelayPropertyChange(self, value: str):
        self._meta.hset(mt.TR_XAS_PROC, "delay_property", value)

    def onTrXasEnergyDeviceChange(self, value: str):
        self._meta.hset(mt.TR_XAS_PROC, "energy_device", value)

    def onTrXasEnergyPropertyChange(self, value: str):
        self._meta.hset(mt.TR_XAS_PROC, "energy_property", value)

    def onTrXasNoDelayBinsChange(self, value: int):
        self._meta.hset(mt.TR_XAS_PROC, "n_delay_bins", value)

    def onTrXasDelayRangeChange(self, value: tuple):
        self._meta.hset(mt.TR_XAS_PROC, "delay_range", str(value))

    def onTrXasNoEnergyBinsChange(self, value: int):
        self._meta.hset(mt.TR_XAS_PROC, "n_energy_bins", value)

    def onTrXasEnergyRangeChange(self, value: tuple):
        self._meta.hset(mt.TR_XAS_PROC, "energy_range", str(value))

    def onTrXasReset(self):
        self._meta.hset(mt.TR_XAS_PROC, "reset", 1)
