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

from ..config import ExtensionType
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
    # geometry assembler
    assembler_change_sgn = pyqtSignal(object)

    connection_change_sgn = pyqtSignal(object)
    file_stream_initialized_sgn = pyqtSignal()

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

    def onExtensionEndpointChange(self, endpoint: str):
        self._meta.hset(mt.EXTENSION, ExtensionType.ALL_OUTPUT.value, endpoint)

    def onDetectorExtensionEndpointChange(self, endpoint: str):
        self._meta.hset(mt.EXTENSION, ExtensionType.DETECTOR_OUTPUT.value, endpoint)

    def onBridgeConnectionsChange(self, connections: dict):
        # key = endpoint, value = source type
        pipe = self._meta.pipeline()
        pipe.execute_command('DEL', mt.CONNECTION)
        pipe.hset(mt.CONNECTION, mapping=connections)
        pipe.execute()

        self.connection_change_sgn.emit(connections)

    def onSourceItemToggled(self, checked: bool, item):
        if checked:
            self._meta.add_data_source(item)
        else:
            self._meta.remove_data_source(item)

    def onPhotonBinningChange(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "photon_binning", str(value))

    def onAduThresholdChanged(self, value: int):
        self._meta.hset(mt.IMAGE_PROC, "adu_count", value)

    def onCalGainCorrection(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "correct_gain", str(value))

    def onCalOffsetCorrection(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "correct_offset", str(value))

    def onCalOffsetPolicyChange(self, value: IntEnum):
        self._meta.hset(mt.IMAGE_PROC, "offset_policy", int(value))

    def onCalGainMemoCellsChange(self, value: list):
        self._meta.hset(mt.IMAGE_PROC, "gain_cells", str(value))

    def onCalOffsetMemoCellsChange(self, value: list):
        self._meta.hset(mt.IMAGE_PROC, "offset_cells", str(value))

    def onCalDarkAsOffset(self, value:bool):
        self._meta.hset(mt.IMAGE_PROC, "dark_as_offset", str(value))

    def onCalDarkRecording(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "recording_dark", str(value))

    def onCalDarkRemove(self):
        self._meta.hset(mt.IMAGE_PROC, "remove_dark", 1)

    def onImageMovingAverageChange(self, value: int):
        self._meta.hset(mt.IMAGE_PROC, "ma_window", value)

    def onImageThresholdMaskChange(self, value: tuple):
        self._meta.hset(mt.IMAGE_PROC, "threshold_mask", str(value))

    def onImageMaskTileEdgeChange(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "mask_tile", str(value))

    def onImageMaskAsicEdgeChange(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "mask_asic", str(value))

    def onImageMaskSaveInModulesToggled(self, value: bool):
        self._meta.hset(mt.IMAGE_PROC, "mask_save_in_modules", str(value))

    def onGeomStackOnlyChange(self, value: bool):
        self._meta.hset(mt.GEOMETRY_PROC, "stack_only", str(value))

    def onGeomAssemblerChange(self, value: IntEnum):
        self._meta.hset(mt.GEOMETRY_PROC, "assembler", int(value))
        self.assembler_change_sgn.emit(value)

    def onGeomFileChange(self, value: str):
        self._meta.hset(mt.GEOMETRY_PROC, "geometry_file", value)

    def onGeomCoordinatesChange(self, value: str):
        self._meta.hset(mt.GEOMETRY_PROC, "coordinates", json.dumps(value))

    def onPoiIndexChange(self, idx: int, value: int):
        self._meta.hset(mt.GLOBAL_PROC, f"poi{idx+1}_index", str(value))
        self.poi_index_change_sgn.emit(idx, value)

    def onSampleDistanceChange(self, value: float):
        self._meta.hset(mt.GLOBAL_PROC, 'sample_distance', value)

    def onPhotonEnergyChange(self, value: float):
        self._meta.hset(mt.GLOBAL_PROC, 'photon_energy', value)

    def onMaWindowChange(self, value: int):
        self._meta.hset(mt.GLOBAL_PROC, "ma_window", value)

    def onResetAll(self):
        pipe = self._meta.pipeline()
        pipe.hset(mt.GLOBAL_PROC, "reset_ma", 1)
        pipe.hset(mt.PUMP_PROBE_PROC, "reset", 1)
        pipe.hset(mt.CORRELATION_PROC, "reset1", 1)
        pipe.hset(mt.CORRELATION_PROC, "reset2", 1)
        pipe.hset(mt.HISTOGRAM_PROC, "reset", 1)
        pipe.hset(mt.BINNING_PROC, "reset", 1)
        pipe.execute()

    def onResetMa(self):
        self._meta.hset(mt.GLOBAL_PROC, "reset_ma", 1)

    def onAiPixelSizeXChange(self, value: int):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'pixel_size_x', value)

    def onAiPixelSizeYChange(self, value: int):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'pixel_size_y', value)

    def onAiIntegCenterXChange(self, value: float):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_center_x', value)

    def onAiIntegCenterYChange(self, value: float):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_center_y', value)

    def onAiIntegMethodChange(self, value: str):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_method', value)

    def onAiIntegPointsChange(self, value: int):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_points', value)

    def onAiIntegRangeChange(self, value: tuple):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'integ_range', str(value))

    def onAiNormChange(self, value: IntEnum):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'normalizer', int(value))

    def onAiAucRangeChange(self, value: tuple):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'auc_range', str(value))

    def onAiFomIntegRangeChange(self, value: tuple):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, 'fom_integ_range', str(value))

    def onAiPeakFindingChange(self, value: bool):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, "peak_finding", str(value))

    def onAiPeakProminenceChange(self, value: float):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, "peak_prominence", str(value))

    def onAiPeakSlicerChange(self, value: list):
        self._meta.hset(mt.AZIMUTHAL_INTEG_PROC, "peak_slicer", str(value))

    def onPpModeChange(self, value: IntEnum):
        self._meta.hset(mt.PUMP_PROBE_PROC, 'mode', int(value))

    def onPpOnPulseSlicerChange(self, value: list):
        self._meta.hset(mt.PUMP_PROBE_PROC, 'on_pulse_slicer', str(value))

    def onPpOffPulseSlicerChange(self, value: list):
        self._meta.hset(mt.PUMP_PROBE_PROC, 'off_pulse_slicer', str(value))

    def onPpAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.PUMP_PROBE_PROC, 'analysis_type', int(value))

    def onPpAbsDifferenceChange(self, value: bool):
        self._meta.hset(mt.PUMP_PROBE_PROC, "abs_difference", str(value))

    def onPpReset(self):
        self._meta.hset(mt.PUMP_PROBE_PROC, "reset", 1)

    def onRoiGeometryChange(self, value: tuple):
        idx, activated, locked, x, y, w, h = value
        self._meta.hset(mt.ROI_PROC, f'geom{idx}',
                        str((int(activated), int(locked), x, y, w, h)))

    def onRoiFomTypeChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'fom:type', int(value))

    def onRoiFomComboChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'fom:combo', int(value))

    def onRoiFomNormChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, "fom:norm", int(value))

    def onRoiFomMasterSlaveModeChange(self, value: bool):
        self._meta.hset(mt.ROI_PROC, "fom:master_slave", str(value))

    def onRoiHistComboChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, "hist:combo", int(value))

    def onRoiHistNumBinsChange(self, value: int):
        self._meta.hset(mt.ROI_PROC, "hist:n_bins", int(value))

    def onRoiHistBinRangeChange(self, value: tuple):
        self._meta.hset(mt.ROI_PROC, "hist:bin_range", str(value))

    def onRoiNormSourceChange(self, value: str):
        self._meta.hset(mt.ROI_PROC, "norm:source", value)

    def onRoiNormTypeChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'norm:type', int(value))

    def onRoiNormComboChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'norm:combo', int(value))

    def onRoiProjComboChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'proj:combo', int(value))

    def onRoiProjTypeChange(self, value: IntEnum):
        self._meta.hset(mt.ROI_PROC, 'proj:type', int(value))

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
        pipe = self._meta.pipeline()
        pipe.hset(mt.CORRELATION_PROC, "reset1", 1)
        pipe.hset(mt.CORRELATION_PROC, "reset2", 1)
        pipe.execute()

    def onCorrelationAutoResetMaChange(self, value: bool):
        self._meta.hset(mt.CORRELATION_PROC, 'auto_reset_ma', str(value))

    def onBinParamChange(self, value: tuple):
        # index, source, bin_range, number of bins,
        # where the index starts from 1
        index, src, bin_range, n_bins = value

        pipe = self._meta.pipeline()
        pipe.hset(mt.BINNING_PROC, f'source{index}', src)
        pipe.hset(mt.BINNING_PROC, f'bin_range{index}', str(bin_range))
        pipe.hset(mt.BINNING_PROC, f'n_bins{index}', n_bins)
        pipe.execute()

    def onBinAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.BINNING_PROC, "analysis_type", int(value))

    def onBinModeChange(self, value: IntEnum):
        self._meta.hset(mt.BINNING_PROC, "mode", int(value))

    def onBinReset(self):
        self._meta.hset(mt.BINNING_PROC, "reset", 1)

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

    def onFomFilterAnalysisTypeChange(self, value: IntEnum):
        self._meta.hset(mt.FOM_FILTER_PROC, "analysis_type", int(value))

    def onFomFilterRangeChange(self, value: tuple):
        self._meta.hset(mt.FOM_FILTER_PROC, "fom_range", str(value))

    def onFomFilterPulseResolvedChange(self, value: bool):
        self._meta.hset(mt.FOM_FILTER_PROC, "pulse_resolved", str(value))

    def onItCrCxChange(self, value: float):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "cr:cx", value)

    def onItCrCyChange(self, value: float):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "cr:cy", value)

    def onItCrProminenceChange(self, value: float):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "cr:prominence", value)

    def onItCrDistanceChange(self, value: int):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "cr:distance", value)

    def onItCrMinCountChange(self, value: int):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "cr:min_count", value)

    def onItTransformTypeChange(self, value: int):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "transform_type", value)

    def onItFftLogrithmicScaleChange(self, value: bool):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "fft:logrithmic", str(value))

    def onItEdKernelSizeChange(self, value: int):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "ed:kernel_size", value)

    def onItEdSigmaChange(self, value: int):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "ed:sigma", value)

    def onItEdThresholdChange(self, value: tuple):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, "ed:threshold", str(value))

    def onItBraggPeakRoiChange(self, label: str, value: tuple):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, f"bp:roi_{label}", str(value))

    def onItBraggPeakWindowSizeChange(self, value: int):
        self._meta.hset(mt.IMAGE_TRANSFORM_PROC, f"bp:window_size", value)

    def onItBraggPeakRoiDeletion(self, value: str):
        self._meta.hdel(mt.IMAGE_TRANSFORM_PROC, f"bp:roi_{value}")
