"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout, QLabel, QMainWindow, QVBoxLayout, QWidget
)

from extra_foam.gui.gui_helpers import create_icon_button

from . import __version__
from .cam_view_w import CamViewWindow
from .gotthard_pump_probe_w import GotthardPumpProbeWindow
from .gotthard_w import GotthardWindow
from .module_scan_w import ModuleScanWindow
from .multicam_view_w import MultiCamViewWindow
from .trxas_w import TrXasWindow
from .vector_view import VectorViewWindow
from .xas_tim_w import XasTimWindow
from .xas_tim_xmcd_w import XasTimXmcdWindow
from .xes_timing_w import XesTimingWindow


class SpecialSuiteController:
    def __init__(self):
        self._facade = None
        self._window = None

    def showFacade(self, topic):
        self._facade = create_special_suite(topic)
        self._facade.open_analysis_sgn.connect(self._showAnalysis)
        self._facade.show()

    def _showAnalysis(self, analysis_type, topic):
        self._window = analysis_type(topic)
        self._facade.close()
        self._window.show()


class _SpecialSuiteFacadeBase(QMainWindow):
    """Base class for special analysis suite."""
    _ICON_WIDTH = 160
    _ROW_HEIGHT = 220
    _WIDTH = 720

    open_analysis_sgn = pyqtSignal(object, str)

    def __init__(self, topic):
        super().__init__()

        self.setWindowTitle(
            f"EXtra-foam {__version__} - special analysis suite")

        # StatusBar to display topic name
        self.statusBar().showMessage(f"TOPIC: {topic}")
        self.statusBar().setStyleSheet("QStatusBar{font-weight:bold;}")

        self._topic = topic

        self._buttons = OrderedDict()

        self._cw = QWidget()
        self.setCentralWidget(self._cw)

    def initUI(self):
        layout = QVBoxLayout()
        layout_row = None
        for i, (title, btn) in enumerate(self._buttons.items()):
            if i % 4 == 0:
                layout_row = QGridLayout()
                layout.addLayout(layout_row)
            layout_row.addWidget(QLabel(title), 0, i % 4)
            layout_row.addWidget(btn, 1, i % 4)
            layout_row.setColumnStretch(3, 2)
            layout_row.setRowStretch(2, 2)
        self._cw.setLayout(layout)

        h = len(self._buttons) // 4
        if len(self._buttons) % 4 != 0:
            h += 1
        self.setFixedSize(self._WIDTH, h * self._ROW_HEIGHT)

    def addSpecial(self, instance_type):
        """Add a button for the given analysis."""
        btn = create_icon_button(instance_type.icon, self._ICON_WIDTH)
        btn.clicked.connect(lambda: self.open_analysis_sgn.emit(
            instance_type, self._topic))
        btn.clicked.connect(lambda: btn.setEnabled(False))

        title = instance_type._title
        if title in self._buttons:
            raise RuntimeError(f"Duplicated special analysis title: {title}")
        self._buttons[title] = btn

    def addCommonSpecials(self):
        self.addSpecial(CamViewWindow)
        self.addSpecial(VectorViewWindow)
        self.addSpecial(MultiCamViewWindow)


class SpbSpecialSuiteFacade(_SpecialSuiteFacadeBase):
    def __init__(self):
        super().__init__("SPB")

        self.addSpecial(GotthardWindow)
        self.addSpecial(GotthardPumpProbeWindow)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class FxeSpecialSuiteFacade(_SpecialSuiteFacadeBase):
    def __init__(self):
        super().__init__("FXE")

        self.addSpecial(XesTimingWindow)
        self.addSpecial(GotthardWindow)
        self.addSpecial(GotthardPumpProbeWindow)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class ScsSpecialSuiteFacade(_SpecialSuiteFacadeBase):
    def __init__(self):
        super().__init__("SCS")

        self.addSpecial(XasTimWindow)
        self.addSpecial(XasTimXmcdWindow)
        self.addSpecial(TrXasWindow)
        self.addSpecial(GotthardWindow)
        self.addSpecial(GotthardPumpProbeWindow)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class MidSpecialSuiteFacade(_SpecialSuiteFacadeBase):
    def __init__(self):
        super().__init__("MID")

        self.addSpecial(GotthardWindow)
        self.addSpecial(GotthardPumpProbeWindow)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class HedSpecialSuiteFacade(_SpecialSuiteFacadeBase):
    def __init__(self):
        super().__init__("HED")

        self.addSpecial(GotthardWindow)
        self.addSpecial(GotthardPumpProbeWindow)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class DetSpecialSuiteFacade(_SpecialSuiteFacadeBase):
    def __init__(self):
        super().__init__("DET")

        self.addSpecial(ModuleScanWindow)
        self.addCommonSpecials()

        self.initUI()
        self.show()


def create_special_suite(topic):
    if topic == "SPB":
        return SpbSpecialSuiteFacade()

    if topic == "FXE":
        return FxeSpecialSuiteFacade()

    if topic == "SCS":
        return ScsSpecialSuiteFacade()

    if topic == "MID":
        return MidSpecialSuiteFacade()

    if topic == "HED":
        return HedSpecialSuiteFacade()

    if topic == "DET":
        return DetSpecialSuiteFacade()

    raise ValueError(f"{topic} does not have a special analysis suite")
