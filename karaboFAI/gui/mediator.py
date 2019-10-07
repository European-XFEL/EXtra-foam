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
from ..ipc import RedisConnection


class Mediator(QObject):
    """Mediator for GUI signal-slot connection.

    The behavior of the code should not be affected by when the
    Mediator() is instantiated.
    """

    poi_index1_sgn = pyqtSignal(int)
    poi_index2_sgn = pyqtSignal(int)
    # When pulsed azimuthal integration window is opened, it first connect
    # the above two signals to its two slots. Then it informs the
    # AnalysisCtrlWidget to update the POI indices.
    poi_indices_connected_sgn = pyqtSignal()

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
        self._meta.set(mt.DATA_SOURCE, "endpoint", value)

    def onDetectorSourceNameChange(self, value: str):
        self._meta.set(mt.DATA_SOURCE, "detector_source_name", value)

    def onXgmSourceNameChange(self, value: str):
        self._meta.set(mt.DATA_SOURCE, "xgm_source_name", value)

    def onSourceTypeChange(self, value: IntEnum):
        self._meta.set(mt.DATA_SOURCE, "source_type", int(value))

    def onImageThresholdMaskChange(self, value: tuple):
        self._meta.set(mt.IMAGE_PROC, "threshold_mask", str(value))

    def onImageBackgroundChange(self, value: float):
        self._meta.set(mt.IMAGE_PROC, "background", value)

    def onGeomFilenameChange(self, value: str):
        self._meta.set(mt.GEOMETRY_PROC, "geometry_file", value)

    def onGeomQuadPositionsChange(self, value: str):
        self._meta.set(mt.GEOMETRY_PROC, "quad_positions", json.dumps(value))

    def onPulseIndexSelectorChange(self, value: list):
        self._meta.set(mt.GLOBAL_PROC, 'selected_pulse_indices', str(value))

    def onPoiPulseIndexChange(self, vip_id: int, value: int):
        self._meta.set(mt.GLOBAL_PROC, f"poi{vip_id}_index", str(value))

        if vip_id == 1:
            self.poi_index1_sgn.emit(value)
        else:  # vip_id == 2:
            self.poi_index2_sgn.emit(value)

    def onSampleDistanceChange(self, value: float):
        self._meta.set(mt.GLOBAL_PROC, 'sample_distance', value)

    def onPhotonEnergyChange(self, value: float):
        self._meta.set(mt.GLOBAL_PROC, 'photon_energy', value)

    def onMaWindowChange(self, value: int):
        self._meta.set(mt.GLOBAL_PROC, "ma_window", value)

    def onMaReset(self):
        # TODO: merge into one set
        self._meta.set(mt.GLOBAL_PROC, "reset_ai", 1)
        self._meta.set(mt.GLOBAL_PROC, "reset_roi", 1)

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

    def onCurveNormalizerChange(self, value: IntEnum):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'normalizer', int(value))

    def onAiAucChangeChange(self, value: tuple):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'auc_range', str(value))

    def onAiFomIntegRangeChange(self, value: tuple):
        self._meta.set(mt.AZIMUTHAL_INTEG_PROC, 'fom_integ_range', str(value))

    def onPpModeChange(self, value: IntEnum):
        self._meta.set(mt.PUMP_PROBE_PROC, 'mode', int(value))

    def onPpOnPulseIdsChange(self, value: list):
        self._meta.set(mt.PUMP_PROBE_PROC, 'on_pulse_indices', str(value))

    def onPpOffPulseIdsChange(self, value: list):
        self._meta.set(mt.PUMP_PROBE_PROC, 'off_pulse_indices', str(value))

    def onPpAnalysisTypeChange(self, value: IntEnum):
        self._meta.set(mt.PUMP_PROBE_PROC, 'analysis_type', int(value))

    def onPpAbsDifferenceChange(self, value: bool):
        self._meta.set(mt.PUMP_PROBE_PROC, "abs_difference", str(value))

    def onPpReset(self):
        self._meta.set(mt.PUMP_PROBE_PROC, "reset", 1)
        # reset moving average at the same time
        self.onMaReset()

    def onRoiRegionChange(self, value: tuple):
        rank, x, y, w, h = value
        self._meta.set(mt.ROI_PROC, f'region{rank}', str((x, y, w, h)))

    def onRoiVisibilityChange(self, value: tuple):
        rank, is_visible = value
        self._meta.set(mt.ROI_PROC, f'visibility{rank}', str(is_visible))

    def onRoiProjDirectChange(self, value: str):
        self._meta.set(mt.ROI_PROC, 'proj:direction', value)

    def onRoiProjNormalizerChange(self, value: IntEnum):
        self._meta.set(mt.ROI_PROC, "proj:normalizer", int(value))

    def onRoiProjAucRangeChange(self, value: tuple):
        self._meta.set(mt.ROI_PROC, "proj:auc_range", str(value))

    def onRoiProjFomIntegRangeChange(self, value: tuple):
        self._meta.set(mt.ROI_PROC, "proj:fom_integ_range", str(value))

    def onCorrelationAnalysisTypeChange(self, value: IntEnum):
        self._meta.set(mt.CORRELATION_PROC, "analysis_type", int(value))

    def onCorrelationParamChange(self, value: tuple):
        # index, device ID, property name, resolution
        # index starts from 1
        index, device_id, ppt, resolution = value
        self._meta.set(mt.CORRELATION_PROC, f'device_id{index}', device_id)
        self._meta.set(mt.CORRELATION_PROC, f'property{index}', ppt)
        self._meta.set(mt.CORRELATION_PROC, f'resolution{index}', resolution)

    def onCorrelationReset(self):
        self._meta.set(mt.CORRELATION_PROC, "reset", 1)

    def onBinGroupChange(self, value: tuple):
        # index, device ID, property name, bin_range, number of bins,
        # where the index starts from 1
        index, device_id, ppt, bin_range, n_bins = value

        self._meta.set(mt.BIN_PROC, f'device_id{index}', device_id)
        self._meta.set(mt.BIN_PROC, f'property{index}', ppt)
        self._meta.set(mt.BIN_PROC, f'bin_range{index}', str(bin_range))
        self._meta.set(mt.BIN_PROC, f'n_bins{index}', n_bins)

    def onBinAnalysisTypeChange(self, value: IntEnum):
        self._meta.set(mt.BIN_PROC, "analysis_type", int(value))

    def onBinModeChange(self, value: IntEnum):
        self._meta.set(mt.BIN_PROC, "mode", int(value))

    def onBinReset(self):
        self._meta.set(mt.BIN_PROC, "reset", 1)

    def onStAnalysisTypeChange(self, value: IntEnum):
        self._meta.set(mt.STATISTICS_PROC, "analysis_type", int(value))

    def onStNumBinsChange(self, value: int):
        self._meta.set(mt.STATISTICS_PROC, "n_bins", int(value))

    def onStPulseOrTrainResolutionChange(self, value: bool):
        self._meta.set(mt.STATISTICS_PROC, "pulse_resolved", str(value))

    def onStReset(self):
        self._meta.set(mt.STATISTICS_PROC, "reset", 1)

    def onDrAnalysisTypeChange(self, value: IntEnum):
        self._meta.set(mt.DATA_REDUCTION_PROC, "analysis_type", int(value))

    def onDrFomRangeChange(self, value: tuple):
        self._meta.set(mt.DATA_REDUCTION_PROC, "fom_range", str(value))

    def onRdStateChange(self, value: bool):
        self._meta.set(mt.GLOBAL_PROC, "recording_dark", str(value))

    def onRdResetDark(self):
        self._meta.set(mt.GLOBAL_PROC, "reset_dark", 1)

    def onRdProcessStateChange(self, value: bool):
        self._meta.set(mt.GLOBAL_PROC, "process_dark", str(value))
