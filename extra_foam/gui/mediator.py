"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum
import json

from PyQt5.QtCore import pyqtSignal,  QObject

from ..database import Metadata as mt
from ..database import MetaProxy


class Mediator(QObject):
    """Mediator for GUI signal-slot connection.

    The behavior of the code should not be affected by when the
    Mediator() is instantiated.
    """
    reset_image_level_sgn = pyqtSignal()
    # POI index, pulse index
    poi_index_change_sgn = pyqtSignal(int, int)
    poi_window_initialized_sgn = pyqtSignal()

    bin_heatmap_autolevel_sgn = pyqtSignal()

    __instance = None

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

    def registerAnalysis(self, analysis_type):
        self._meta.register_analysis(analysis_type)

    def unregisterAnalysis(self, analysis_type):
        self._meta.unregister_analysis(analysis_type)

    def onBridgeConnectionsChange(self, connections: dict):
        # key = endpoint, value = source type
        pipe = self._meta.pipeline()
        pipe.delete(mt.CONNECTION)
        pipe.hmset(mt.CONNECTION, connections)
        pipe.execute()

    def onSourceItemToggled(self, checked: bool, item: object):
        if checked:
            self._meta.add_data_source(item)
        else:
            self._meta.remove_data_source(item)

    def onCalGainCorrection(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "correct gain", str(value))

    def onCalOffsetCorrection(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "correct offset", str(value))

    def onCalGainSlicerChange(self, value: list):
        self._meta.hset(mt.IMAGE_PROC, "gain slicer", str(value))

    def onCalOffsetSlicerChange(self, value: list):
        self._meta.hset(mt.IMAGE_PROC, "offset slicer", str(value))

    def onCalDarkAsOffset(self, value:bool):
        self._meta.hset(mt.IMAGE_PROC, "dark as offset", str(value))

    def onCalDarkRecording(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "recording dark", str(value))

    def onCalDarkRemove(self):
        self._meta.hset(mt.IMAGE_PROC, "remove dark", 1)

    def onImageThresholdMaskChange(self, value: tuple):
        self._meta.hset(mt.IMAGE_PROC, "threshold_mask", str(value))

    def onGeomStackOnlyChange(self, value: bool):
        self._meta.hset(mt.GEOMETRY_PROC, "stack_only", str(value))

    def onGeomAssemblerChange(self, value: IntEnum):
        self._meta.hset(mt.GEOMETRY_PROC, "assembler", int(value))

    def onGeomFilenameChange(self, value: str):
        self._meta.hset(mt.GEOMETRY_PROC, "geometry_file", value)

    def onGeomQuadPositionsChange(self, value: str):
        self._meta.hset(mt.GEOMETRY_PROC, "quad_positions", json.dumps(value))

    def onPoiIndexChange(self, idx: int, value: int):
        self._meta.hset(mt.GLOBAL_PROC, f"poi{idx+1}_index", str(value))
        self.poi_index_change_sgn.emit(idx, value)

    def onSampleDistanceChange(self, value: float):
        self._meta.hset(mt.GLOBAL_PROC, 'sample_distance', value)

    def onPhotonEnergyChange(self, value: float):
        self._meta.hset(mt.GLOBAL_PROC, 'photon_energy', value)

    def onMaWindowChange(self, value: int):
        self._meta.hset(mt.GLOBAL_PROC, "ma_window", value)

    def onResetMa(self):
        self._meta.hmset(mt.GLOBAL_PROC, {
            "reset_ma_ai": 1,
            "reset_ma_roi": 1,
            "reset_ma_xgm": 1,
            "reset_ma_digitizer": 1})

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
        idx, x, y, w, h = value
        self._meta.hset(mt.ROI_PROC, f'geom{idx}', str((x, y, w, h)))

    def onRoiFomTypeChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'fom:type', int(value))

    def onRoiFomComboChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'fom:combo', int(value))

    def onRoiFomNormChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, "fom:norm", int(value))

    def onRoiHistComboChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, "hist:combo", int(value))

    def onRoiHistNumBinsChange(self, value: int):
        self._meta.hset(mt.ROI_PROC, "hist:n_bins", int(value))

    def onRoiHistBinRangeChange(self, value: tuple):
        self._meta.hset(mt.ROI_PROC, "hist:bin_range", str(value))

    def onRoiNormTypeChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'norm:type', int(value))

    def onRoiNormComboChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'norm:combo', int(value))

    def onRoiProjComboChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'proj:combo', int(value))

    def onRoiProjDirectChange(self, value: str):
        self._meta.hset(mt.ROI_PROC, 'proj:direct', value)

    def onRoiProjNormChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, "proj:norm", int(value))

    def onRoiProjAucRangeChange(self, value: tuple):
        self._meta.hset(mt.ROI_PROC, "proj:auc_range", str(value))

    def onRoiProjFomIntegRangeChange(self, value: tuple):
        self._meta.hset(mt.ROI_PROC, "proj:fom_integ_range", str(value))

    def onCorrelationAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.CORRELATION_PROC, "analysis_type", int(value))

    def onCorrelationParamChange(self, value: tuple):
        # index, source, resolution
        # index starts from 1
        index, src, resolution = value
        pipe = self._meta.pipeline()
        pipe.hset(mt.CORRELATION_PROC, f'source{index}', src)
        pipe.hset(mt.CORRELATION_PROC, f'resolution{index}', resolution)
        pipe.execute()

    def onCorrelationReset(self):
        self._meta.hset(mt.CORRELATION_PROC, "reset", 1)

    def onBinParamChange(self, value: tuple):
        # index, source, bin_range, number of bins,
        # where the index starts from 1
        index, src, bin_range, n_bins = value

        pipe = self._meta.pipeline()
        pipe.hset(mt.BIN_PROC, f'source{index}', src)
        pipe.hset(mt.BIN_PROC, f'bin_range{index}', str(bin_range))
        pipe.hset(mt.BIN_PROC, f'n_bins{index}', n_bins)
        pipe.execute()

    def onBinAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.BIN_PROC, "analysis_type", int(value))

    def onBinModeChange(self, value: IntEnum):
        self._meta.hset(mt.BIN_PROC, "mode", int(value))

    def onBinReset(self):
        self._meta.hset(mt.BIN_PROC, "reset", 1)

    def onHistAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.HISTOGRAM_PROC, "analysis_type", int(value))

    def onHistBinRangeChange(self, value: tuple):
        self._meta.hset(mt.HISTOGRAM_PROC, "bin_range", str(value))

    def onHistNumBinsChange(self, value: int):
        self._meta.hset(mt.HISTOGRAM_PROC, "n_bins", int(value))

    def onHistPulseResolvedChange(self, value: bool):
        self._meta.hset(mt.HISTOGRAM_PROC, "pulse_resolved", str(value))

    def onHistReset(self):
        self._meta.hset(mt.HISTOGRAM_PROC, "reset", 1)

    def onPfAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.PULSE_FILTER_PROC, "analysis_type", int(value))

    def onPfFomRangeChange(self, value: tuple):
        self._meta.hset(mt.PULSE_FILTER_PROC, "fom_range", str(value))

    def onTrXasScanStateToggled(self, value: IntEnum, state: bool):
        if state:
            self._meta.hset(mt.TR_XAS_PROC, "analysis_type", int(value))
        else:
            self._meta.hdel(mt.TR_XAS_PROC, "analysis_type")

    def onTrXasDelaySourceChange(self, value: str):
        self._meta.hset(mt.TR_XAS_PROC, "delay_source", value)

    def onTrXasEnergySourceChange(self, value: str):
        self._meta.hset(mt.TR_XAS_PROC, "energy_source", value)

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
