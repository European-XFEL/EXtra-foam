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

    __instance = None

    def __new__(cls, *args, **kwargs):
        """Create a singleton."""
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self, processor=None, *args, **kwargs):
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__(*args, **kwargs)

        self._proc = processor

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
    def onRoiValueTypeChange(self, state):
        self._proc.update_roi_value_type(state)

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onRoi1Change(self, activated, w, h, px, py):
        self._proc.update_roi1_region(activated, w, h, px, py)

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onRoi2Change(self, activated, w, h, px, py):
        self._proc.update_roi2_region(activated, w, h, px, py)

    def updateVipPulseIds(self):
        self.update_vip_pulse_ids_sgn.emit()
