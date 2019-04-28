"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Mediator class.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .pyqtgraph import QtCore


class Mediator(QtCore.QObject):
    """Mediator for GUI signal-slot connection."""

    tcp_host_change_sgn = QtCore.pyqtSignal(str)
    tcp_port_change_sgn = QtCore.pyqtSignal(int)

    vip_pulse_id1_sgn = QtCore.pyqtSignal(int)
    vip_pulse_id2_sgn = QtCore.pyqtSignal(int)
    update_vip_pulse_ids_sgn = QtCore.pyqtSignal()

    roi_displayed_range_sgn = QtCore.pyqtSignal(int)

    # index, device ID, property name, resolution
    correlation_param_change_sgn = QtCore.pyqtSignal(int, str, str, float)

    reset_image_level_sgn = QtCore.pyqtSignal()

    source_type_change_sgn = QtCore.pyqtSignal(int)
    detector_source_change_sgn = QtCore.pyqtSignal(str)
    xgm_source_change_sgn = QtCore.pyqtSignal(str)
    mono_source_change_sgn = QtCore.pyqtSignal(str)

    pp_ma_window_change_sgn = QtCore.pyqtSignal(int)

    reset_xas_sgn = QtCore.pyqtSignal()
    energy_bins_change_sgn = QtCore.pyqtSignal(int)

    roi_region_change_sgn = QtCore.pyqtSignal(int, bool, int, int, int, int)
    roi_fom_change_sgn = QtCore.pyqtSignal(object)
    roi_hist_clear_sgn = QtCore.pyqtSignal()

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
        self.source_type_change_sgn.connect(scheduler.onSourceTypeChange)
        self.detector_source_change_sgn.connect(scheduler.onDetectorSourceChange)
        self.xgm_source_change_sgn.connect(scheduler.onXgmSourceChange)
        self.mono_source_change_sgn.connect(scheduler.onMonoSourceChange)

        self.pp_ma_window_change_sgn.connect(scheduler.onPumpProbeMAWindowChange)

        self.reset_xas_sgn.connect(scheduler.onXasReset)
        self.energy_bins_change_sgn.connect(scheduler.onXasEnergyBinsChange)

        self.roi_region_change_sgn.connect(scheduler.onRoiRegionChange)
        self.roi_fom_change_sgn.connect(scheduler.onRoiFomChange)
        self.roi_hist_clear_sgn.connect(scheduler.clear_roi_hist)

    def connect_bridge(self, bridge):
        self.tcp_host_change_sgn.connect(bridge.onTcpHostChange)
        self.tcp_port_change_sgn.connect(bridge.onTcpPortChange)

    @QtCore.pyqtSlot(int)
    def onPulseID1Updated(self, v):
        self.vip_pulse_id1_sgn.emit(v)

    @QtCore.pyqtSlot(int)
    def onPulseID2Updated(self, v):
        self.vip_pulse_id2_sgn.emit(v)

    @QtCore.pyqtSlot()
    def onRoiDisplayedRangeChange(self):
        v = int(self.sender().text())
        self.roi_displayed_range_sgn.emit(v)

    def updateVipPulseIds(self):
        self.update_vip_pulse_ids_sgn.emit()

    def onAutoLevel(self):
        self.reset_image_level_sgn.emit()
