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
    update_vip_pulse_ids_sgn = pyqtSignal()

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

    @pyqtSlot(int)
    def onPulseID1Updated(self, v):
        self.vip_pulse_id1_sgn.emit(v)

    @pyqtSlot(int)
    def onPulseID2Updated(self, v):
        self.vip_pulse_id2_sgn.emit(v)

    @pyqtSlot()
    def onRoiDisplayedRangeChange(self):
        v = int(self.sender().text())
        self.roi_displayed_range_sgn.emit(v)

    def updateVipPulseIds(self):
        self.update_vip_pulse_ids_sgn.emit()

    def onAutoLevel(self):
        self.reset_image_level_sgn.emit()
