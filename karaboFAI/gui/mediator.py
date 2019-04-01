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
    tcp_host_change_sgn = QtCore.pyqtSignal(str)
    tcp_port_change_sgn = QtCore.pyqtSignal(int)

    xas_state_toggle_sgn = QtCore.pyqtSignal(int)
    reset_xas_sgn = QtCore.pyqtSignal()
    energy_bins_change_sgn = QtCore.pyqtSignal(int)

    __instance = None

    def __new__(cls, *args, **kwargs):
        """Create a singleton."""
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self, *args, **kwargs):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__(*args, **kwargs)

        self._pipe = None
        self._daq = None

    def setPipeline(self, pipe):
        self._pipe = pipe
        self.initPipeConnections()

    def initPipeConnections(self):
        self.source_type_change_sgn.connect(self._pipe.onSourceTypeChange)
        self.detector_source_change_sgn.connect(self._pipe.onDetectorSourceChange)
        self.xgm_source_change_sgn.connect(self._pipe.onXgmSourceChange)
        self.mono_source_change_sgn.connect(self._pipe.onMonoSourceChange)

        self.xas_state_toggle_sgn.connect(self._pipe.onXasStateToggle)
        self.reset_xas_sgn.connect(self._pipe.onXasClear)
        self.energy_bins_change_sgn.connect(self._pipe.onXasEnergyBinsChange)

    def setDaq(self, daq):
        self._daq = daq
        self.initDaqConnections()

    def initDaqConnections(self):
        self.tcp_host_change_sgn.connect(self._daq.onTcpHostChange)
        self.tcp_port_change_sgn.connect(self._daq.onTcpPortChange)

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

    @QtCore.pyqtSlot()
    def onRoiHistClear(self):
        self._pipe.clear_roi_hist()

    @QtCore.pyqtSlot(object)
    def onRoiFomChange(self, state):
        self._pipe.update_roi_fom(state)

    @QtCore.pyqtSlot(int, bool, int, int, int, int)
    def onRoiChange(self, rank, activated, w, h, px, py):
        self._pipe.update_roi_region(rank, activated, w, h, px, py)

    def updateVipPulseIds(self):
        self.update_vip_pulse_ids_sgn.emit()

    def onAutoLevel(self):
        self.reset_image_level_sgn.emit()
