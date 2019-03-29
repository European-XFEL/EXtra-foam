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

        self._proc = None

    def setProcessor(self, proc):
        self._proc = proc

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
        self._proc.clear_roi_hist()

    @QtCore.pyqtSlot(object)
    def onRoiFomChange(self, state):
        self._proc.update_roi_fom(state)

    @QtCore.pyqtSlot(int, bool, int, int, int, int)
    def onRoiChange(self, rank, activated, w, h, px, py):
        self._proc.update_roi_region(rank, activated, w, h, px, py)

    def updateVipPulseIds(self):
        self.update_vip_pulse_ids_sgn.emit()

    def onAutoLevel(self):
        self.reset_image_level_sgn.emit()
