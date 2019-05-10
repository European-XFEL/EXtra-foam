"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Mediator class.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject


class Mediator(QObject):
    """Mediator for GUI signal-slot connection."""

    bridge_endpoint_sgn = pyqtSignal(str)

    port_change_sgn = pyqtSignal(str)
    data_folder_change_sgn = pyqtSignal(str)

    start_file_server_sgn = pyqtSignal()
    stop_file_server_sgn = pyqtSignal()
    file_server_started_sgn = pyqtSignal()
    file_server_stopped_sgn = pyqtSignal()

    vip_pulse_id1_sgn = pyqtSignal(int)
    vip_pulse_id2_sgn = pyqtSignal(int)
    # tell the control widget to update VIP pulse IDs
    vip_pulse_ids_connected_sgn = pyqtSignal()

    roi_displayed_range_sgn = pyqtSignal(int)

    # index, device ID, property name, resolution
    correlation_param_change_sgn = pyqtSignal(int, str, str, float)
    correlation_fom_change_sgn = pyqtSignal(object)

    reset_image_level_sgn = pyqtSignal()

    source_type_change_sgn = pyqtSignal(int)
    detector_source_change_sgn = pyqtSignal(str)
    xgm_source_change_sgn = pyqtSignal(str)
    mono_source_change_sgn = pyqtSignal(str)

    pp_ma_window_change_sgn = pyqtSignal(int)
    pp_abs_difference_sgn = pyqtSignal(bool)
    pp_analysis_type_sgn = pyqtSignal(object)

    proj1d_normalizer_change_sgn = pyqtSignal(object)
    proj1d_auc_x_range_change_sgn = pyqtSignal(float, float)
    proj1d_fom_integ_range_change_sgn = pyqtSignal(float, float)

    reset_xas_sgn = pyqtSignal()
    energy_bins_change_sgn = pyqtSignal(int)

    roi_region_change_sgn = pyqtSignal(int, bool, int, int, int, int)
    roi_fom_change_sgn = pyqtSignal(object)
    roi_hist_clear_sgn = pyqtSignal()

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

        self._is_initialized = True

    def connect_scheduler(self, scheduler):
        # with the scheduler
        self.source_type_change_sgn.connect(scheduler.onSourceTypeChange)
        self.detector_source_change_sgn.connect(
            scheduler.onDetectorSourceChange)
        self.xgm_source_change_sgn.connect(scheduler.onXgmSourceChange)
        self.mono_source_change_sgn.connect(scheduler.onMonoSourceChange)
        self.pp_ma_window_change_sgn.connect(
            scheduler.onPumpProbeMAWindowChange)
        self.reset_xas_sgn.connect(scheduler.onXasReset)
        self.energy_bins_change_sgn.connect(scheduler.onXasEnergyBinsChange)
        self.roi_region_change_sgn.connect(scheduler.onRoiRegionChange)
        self.roi_fom_change_sgn.connect(scheduler.onRoiFomChange)
        self.roi_hist_clear_sgn.connect(scheduler.onRoiHistClear)

        self.pp_abs_difference_sgn.connect(scheduler.onPpDifferenceTypeChange)
        self.pp_analysis_type_sgn.connect(scheduler.onPpAnalysisTypeChange)

        self.proj1d_normalizer_change_sgn.connect(
            scheduler.onProj1dNormalizerChange)
        self.proj1d_auc_x_range_change_sgn.connect(
            scheduler.onProj1dAucXRangeChange)
        self.proj1d_fom_integ_range_change_sgn.connect(
            scheduler.onProj1dFomIntegRangeChange)

        self.correlation_fom_change_sgn.connect(
            scheduler.onCorrelationFomChange)

    def connect_bridge(self, bridge):
        self.bridge_endpoint_sgn.connect(bridge.onEndpointChange)
        self.source_type_change_sgn.connect(bridge.onSourceTypeChange)
